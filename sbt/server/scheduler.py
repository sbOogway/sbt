"""Scheduler daemon for managing backtest jobs and worker processes.

Durability model:
- Startup reconciliation re-enqueues pending/stale-running jobs from the DB.
- Workers must ACK job receipt; un-ACKed dispatches are killed + requeued.
- Busy workers are heartbeated (PING/PONG); workers missing several pongs or
  whose process dies are killed, their job requeued (or failed after
  MAX_ATTEMPTS), and a replacement worker spawned.
- Each job has a hard wall-clock budget (`BacktestJob.timeout_seconds`).

Timing knobs are env-overridable for tests: SBT_ACK_TIMEOUT,
SBT_HEARTBEAT_INTERVAL, SBT_MAX_ATTEMPTS.
"""

import json
import logging
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import zmq

from ..core.config import RunConfig
from ..core.db import ResultStore
from ..core.job import BacktestJob, BacktestResult, JobStatus
from .worker import ensure_worktree

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("Scheduler")

ACK_TIMEOUT_S = float(os.environ.get("SBT_ACK_TIMEOUT", "30"))
HEARTBEAT_INTERVAL_S = float(os.environ.get("SBT_HEARTBEAT_INTERVAL", "10"))
MAX_MISSED_PONGS = int(os.environ.get("SBT_MAX_MISSED_PONGS", "3"))
MAX_ATTEMPTS = int(os.environ.get("SBT_MAX_ATTEMPTS", "2"))


class Scheduler:
    """Central scheduler managing client requests, job queue, and worker pool."""

    def __init__(
        self,
        num_workers: int = 4,
        client_endpoint: str = "tcp://127.0.0.1:5555",
        worker_endpoint: str = "tcp://127.0.0.1:5556",
        db_path: str = "sbt.db",
        worktree_root: str = ".worktrees",
        repo_root: str = ".",
    ) -> None:
        self.num_workers = num_workers
        self.client_endpoint = client_endpoint
        self.worker_endpoint = worker_endpoint
        self.db_path = db_path
        self.worktree_root = Path(worktree_root).resolve()
        self.repo_root = Path(repo_root).resolve()

        self.db = ResultStore(self.db_path)
        self.job_queue: deque[BacktestJob] = deque()
        self.idle_workers: set[str] = set()
        self.busy_workers: dict[str, str] = {}  # worker_id -> job_id (post-ACK)
        self.worker_procs: dict[str, subprocess.Popen] = {}  # worker_id -> proc
        self.jobs_by_id: dict[str, BacktestJob] = {}  # every live job
        self.awaiting_ack: dict[
            str, tuple[str, float]
        ] = {}  # job_id -> (worker_id, deadline)
        self.dispatched_at: dict[str, float] = {}  # job_id -> monotonic
        self.last_pong: dict[str, float] = {}  # worker_id -> monotonic
        self._last_heartbeat = 0.0
        self._running = False

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    def _spawn_worker(self, worker_id: str) -> subprocess.Popen:
        """Create the worktree (if needed) and spawn one worker subprocess."""
        wt_path = (self.worktree_root / worker_id).resolve()
        isolated = ensure_worktree(wt_path, self.repo_root)
        if not isolated:
            logger.warning(
                "Worker %s: %s is NOT a git worktree (mkdir fallback). "
                "Code isolation DEGRADED: falling back to repo root %s on PYTHONPATH.",
                worker_id,
                wt_path,
                self.repo_root,
            )
        child_env = dict(os.environ)
        pythonpath = [str(wt_path)]
        if not isolated:
            pythonpath.append(str(self.repo_root))
        existing = child_env.get("PYTHONPATH", "")
        if existing:
            pythonpath.append(existing)
        child_env["PYTHONPATH"] = os.pathsep.join(pythonpath)

        cmd = [
            sys.executable,
            "-m",
            "sbt.server.worker",
            "--worker-id",
            worker_id,
            "--worktree",
            str(wt_path),
            "--repo-root",
            str(self.repo_root),
            "--endpoint",
            self.worker_endpoint,
        ]
        proc = subprocess.Popen(cmd, cwd=str(wt_path), env=child_env)
        self.worker_procs[worker_id] = proc
        logger.info("Spawned %s (PID %d, cwd %s)", worker_id, proc.pid, wt_path)
        return proc

    def _spawn_workers(self) -> None:
        """Spawn the fixed worker subprocesses."""
        logger.info(
            "Spawning %d worker processes in %s...",
            self.num_workers,
            self.worktree_root,
        )
        for i in range(self.num_workers):
            self._spawn_worker(f"worker-{i}")

    def _kill_proc(self, worker_id: str) -> None:
        proc = self.worker_procs.get(worker_id)
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=2.0)
            except Exception as e:
                logger.warning("Failed to terminate %s: %s", worker_id, e)

    def _handle_worker_death(
        self, worker_id: str, reason: str, respawn: bool = True
    ) -> None:
        """Kill a worker, requeue-or-fail its job, optionally respawn it."""
        logger.warning("Worker %s down (%s)", worker_id, reason)
        self._kill_proc(worker_id)
        self.worker_procs.pop(worker_id, None)
        self.idle_workers.discard(worker_id)
        self.last_pong.pop(worker_id, None)

        # Job already ACKed and running there…
        job_id = self.busy_workers.pop(worker_id, None)
        # …or dispatched but never ACKed.
        if job_id is None:
            for jid, (wid, _deadline) in list(self.awaiting_ack.items()):
                if wid == worker_id:
                    job_id = jid
                    break
        if job_id is not None:
            self.awaiting_ack.pop(job_id, None)
            self.dispatched_at.pop(job_id, None)
            self._requeue_or_fail(job_id, reason=f"worker {worker_id} {reason}")

        if respawn:
            self._spawn_worker(worker_id)

    # ------------------------------------------------------------------
    # Job state transitions
    # ------------------------------------------------------------------

    def _requeue_or_fail(self, job_id: str, reason: str) -> None:
        """Put a job back in the queue, or fail it once attempts are exhausted."""
        job = self.jobs_by_id.get(job_id)
        if job is None:
            return
        self.jobs_by_id.pop(job_id, None)
        if job.attempts >= MAX_ATTEMPTS:
            logger.error(
                "Job %s failed after %d attempts (%s); giving up.",
                job_id,
                job.attempts,
                reason,
            )
            self.db.update_job_status(job_id, JobStatus.FAILED, None)
            return
        logger.info(
            "Requeueing job %s (attempt %d/%d; %s)",
            job_id,
            job.attempts + 1,
            MAX_ATTEMPTS,
            reason,
        )
        job.status = JobStatus.PENDING
        job.worker_id = None
        self.db.update_job_status(job_id, JobStatus.PENDING, None)
        self.job_queue.appendleft(job)
        self.jobs_by_id[job_id] = job

    def _dispatch(self, worker_sock: zmq.Socket) -> None:
        """Dispatch pending jobs to available idle workers."""
        while self.job_queue and self.idle_workers:
            worker_id = self.idle_workers.pop()
            job = self.job_queue.popleft()
            job.status = JobStatus.RUNNING
            job.worker_id = worker_id
            job.attempts += 1

            self.db.update_job_dispatch(job)
            self.busy_workers[worker_id] = job.id
            self.awaiting_ack[job.id] = (
                worker_id,
                time.monotonic() + ACK_TIMEOUT_S,
            )
            self.dispatched_at[job.id] = time.monotonic()
            self.jobs_by_id[job.id] = job

            logger.info(
                "Dispatching job %s (%s) to %s (attempt %d)",
                job.id,
                job.config.strategy_name,
                worker_id,
                job.attempts,
            )
            worker_sock.send_multipart(
                [
                    worker_id.encode("utf-8"),
                    json.dumps({"type": "JOB", "job": job.to_dict()}).encode("utf-8"),
                ]
            )

    def _check_acks(self) -> None:
        """Kill+requeue for dispatches whose ACK never arrived."""
        now = time.monotonic()
        expired = [jid for jid, (_w, dl) in self.awaiting_ack.items() if now > dl]
        for job_id in expired:
            worker_id = self.awaiting_ack[job_id][0]
            logger.warning(
                "Job %s: no ACK from %s within %.0fs", job_id, worker_id, ACK_TIMEOUT_S
            )
            # Kill before requeue so a late-starting worker can't double-run it;
            # _handle_worker_death finds the job via awaiting_ack/busy maps.
            self._handle_worker_death(
                worker_id, reason=f"ACK timeout for job {job_id}", respawn=True
            )

    def _heartbeat(self, worker_sock: zmq.Socket) -> None:
        """PING busy workers; reap those that miss too many PONGs."""
        now = time.monotonic()
        if now - self._last_heartbeat < HEARTBEAT_INTERVAL_S:
            return
        self._last_heartbeat = now
        busy = list(self.busy_workers)
        for worker_id in busy:
            last = self.last_pong.get(worker_id)
            if (
                last is not None
                and now - last > MAX_MISSED_PONGS * HEARTBEAT_INTERVAL_S
            ):
                self._handle_worker_death(
                    worker_id, reason=f"missed >{MAX_MISSED_PONGS} heartbeats"
                )
                continue
            try:
                worker_sock.send_multipart(
                    [worker_id.encode("utf-8"), b'{"type": "PING"}'],
                    flags=zmq.NOBLOCK,
                )
            except zmq.ZMQError:
                pass

    def _check_job_timeouts(self) -> None:
        """Enforce the hard wall-clock budget of each running job."""
        now = time.monotonic()
        for job_id, started in list(self.dispatched_at.items()):
            job = self.jobs_by_id.get(job_id)
            if job is None:
                self.dispatched_at.pop(job_id, None)
                continue
            if now - started <= job.timeout_seconds:
                continue
            worker_id = (
                self.busy_workers.get(job_id)
                or self.awaiting_ack.get(job_id, ("?", 0))[0]
            )
            logger.error(
                "Job %s exceeded timeout of %ds on %s",
                job_id,
                job.timeout_seconds,
                worker_id,
            )
            if worker_id in self.worker_procs:
                self._handle_worker_death(
                    worker_id, reason=f"job {job_id} timeout", respawn=True
                )

    def _check_child_procs(self) -> None:
        """Respawn workers whose processes exited on their own."""
        for worker_id, proc in list(self.worker_procs.items()):
            rc = proc.poll()
            if rc is not None:
                self._handle_worker_death(
                    worker_id, reason=f"process exited with code {rc}"
                )

    # ------------------------------------------------------------------
    # Client / worker message handling
    # ------------------------------------------------------------------

    def _handle_client_req(
        self, client_sock: zmq.Socket, worker_sock: zmq.Socket
    ) -> None:
        """Process requests arriving from client CLI / optimizer."""
        msg_parts = client_sock.recv_multipart()
        if len(msg_parts) < 2:
            return
        client_id = msg_parts[0]
        # REQ clients arrive as [identity, b'', payload]; DEALER-style
        # callers may omit the empty delimiter frame.
        raw = msg_parts[2] if len(msg_parts) >= 3 else msg_parts[1]
        try:
            req = json.loads(raw.decode("utf-8"))
        except Exception as e:
            client_sock.send_multipart(
                [
                    client_id,
                    json.dumps({"status": "error", "error": str(e)}).encode("utf-8"),
                ]
            )
            return

        action = req.get("action")

        if action == "submit":
            config_dict = req.get("config", {})
            study_name = req.get("study_name")
            config = RunConfig.from_dict(config_dict)
            job = BacktestJob(config=config, study_name=study_name)
            self.db.save_job(job)
            self.job_queue.append(job)
            self.jobs_by_id[job.id] = job
            logger.info("Enqueued job %s (%s)", job.id, job.config.strategy_name)
            self._dispatch(worker_sock)
            resp = {"status": "ok", "job_id": job.id}

        elif action == "submit_batch":
            configs = req.get("configs", [])
            study_name = req.get("study_name")
            job_ids = []
            for c_dict in configs:
                config = RunConfig.from_dict(c_dict)
                job = BacktestJob(config=config, study_name=study_name)
                self.db.save_job(job)
                self.job_queue.append(job)
                self.jobs_by_id[job.id] = job
                job_ids.append(job.id)
            logger.info("Enqueued batch of %d jobs", len(job_ids))
            self._dispatch(worker_sock)
            resp = {"status": "ok", "job_ids": job_ids}

        elif action == "status":
            jobs = self.db.list_jobs()
            jobs_data = [j.to_dict() for j in jobs]
            resp = {
                "status": "ok",
                "workers_total": self.num_workers,
                "workers_idle": len(self.idle_workers),
                "workers_busy": len(self.busy_workers),
                "queue_length": len(self.job_queue),
                "jobs": jobs_data,
            }

        elif action == "get_result":
            job_id = req.get("job_id")
            result = self.db.get_result(job_id)
            if result:
                resp = {"status": "ok", "result": result.to_dict()}
            else:
                job = self.db.get_job(job_id)
                if job:
                    resp = {"status": "pending", "job_status": job.status.value}
                else:
                    resp = {"status": "not_found", "error": f"Job {job_id} not found"}

        elif action == "list_results":
            results = self.db.list_results(study_name=req.get("study_name"))
            resp = {"status": "ok", "results": [r.to_dict() for r in results]}

        elif action == "ping":
            resp = {"status": "ok", "message": "pong"}

        else:
            resp = {"status": "error", "error": f"Unknown action: {action}"}

        client_sock.send_multipart([client_id, json.dumps(resp).encode("utf-8")])

    def _handle_worker_msg(self, worker_sock: zmq.Socket) -> None:
        """Process messages arriving from workers."""
        msg_parts = worker_sock.recv_multipart()
        if len(msg_parts) < 2:
            return
        worker_id = msg_parts[0].decode("utf-8")
        try:
            msg = json.loads(msg_parts[1].decode("utf-8"))
        except Exception as e:
            logger.error("Failed to parse worker message: %s", e)
            return

        msg_type = msg.get("type")

        if msg_type == "READY":
            logger.info("Worker %s is READY", worker_id)
            self.busy_workers.pop(worker_id, None)
            self.idle_workers.add(worker_id)
            self.last_pong[worker_id] = time.monotonic()
            self._dispatch(worker_sock)

        elif msg_type == "ACK":
            job_id = msg.get("job_id")
            expected = self.awaiting_ack.get(job_id)
            if expected and expected[0] == worker_id:
                self.awaiting_ack.pop(job_id)
                logger.info("Job %s ACKed by %s", job_id, worker_id)
            else:
                logger.warning(
                    "Stale/unknown ACK for job %s from %s (expected %s)",
                    job_id,
                    worker_id,
                    expected[0] if expected else "nothing",
                )
            self.last_pong[worker_id] = time.monotonic()

        elif msg_type == "PONG":
            self.last_pong[worker_id] = time.monotonic()

        elif msg_type == "RESULT":
            job_id = msg.get("job_id")

            # Drop late/duplicate results (e.g. after a requeue raced).
            owner = self.active_owner(job_id)
            if owner != worker_id:
                logger.warning(
                    "Dropping RESULT for job %s from %s (expected owner: %s)",
                    job_id,
                    worker_id,
                    owner,
                )
                return

            result_dict = msg.get("result", {})
            result = BacktestResult.from_dict(result_dict)
            self.db.complete_job(job_id, result, worker_id)

            logger.info(
                "Job %s completed by %s (Sharpe: %s, Trades: %s, PnL: %s)",
                job_id,
                worker_id,
                f"{result.sharpe_ratio:.2f}"
                if result.sharpe_ratio is not None
                else "N/A",
                result.num_trades,
                f"${result.pnl:,.2f}" if result.pnl is not None else "N/A",
            )

            self.busy_workers.pop(worker_id, None)
            self.jobs_by_id.pop(job_id, None)
            self.dispatched_at.pop(job_id, None)
            self.last_pong[worker_id] = time.monotonic()
            self.idle_workers.add(worker_id)
            self._dispatch(worker_sock)

    def active_owner(self, job_id: str) -> str | None:
        """The worker believed to own *job_id* right now, if any."""
        wid = self.busy_workers_inverse(job_id)
        if wid is not None:
            return wid
        entry = self.awaiting_ack.get(job_id)
        return entry[0] if entry else None

    def busy_workers_inverse(self, job_id: str) -> str | None:
        for wid, jid in self.busy_workers.items():
            if jid == job_id:
                return wid
        return None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the scheduler event loop."""
        self._running = True

        backlog = self.db.reconcile_backlog()
        if backlog:
            logger.info(
                "Recovered %d unfinished job(s) from previous run:", len(backlog)
            )
            for job in backlog:
                logger.info(
                    "  - %s (%s, attempts=%d)",
                    job.id,
                    job.config.strategy_name,
                    job.attempts,
                )
                self.job_queue.append(job)
                self.jobs_by_id[job.id] = job

        self._spawn_workers()

        ctx = zmq.Context()
        client_sock = ctx.socket(zmq.ROUTER)
        client_sock.bind(self.client_endpoint)

        worker_sock = ctx.socket(zmq.ROUTER)
        worker_sock.bind(self.worker_endpoint)

        logger.info(
            "Scheduler listening on client=%s, worker=%s",
            self.client_endpoint,
            self.worker_endpoint,
        )

        poller = zmq.Poller()
        poller.register(client_sock, zmq.POLLIN)
        poller.register(worker_sock, zmq.POLLIN)

        try:
            while self._running:
                events = dict(poller.poll(timeout=500))

                if client_sock in events and events[client_sock] == zmq.POLLIN:
                    self._handle_client_req(client_sock, worker_sock)

                if worker_sock in events and events[worker_sock] == zmq.POLLIN:
                    self._handle_worker_msg(worker_sock)

                # Durability sweeps (cheap; run each tick)
                self._check_acks()
                self._check_job_timeouts()
                self._check_child_procs()
                self._heartbeat(worker_sock)

        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        finally:
            self.stop(client_sock, worker_sock, ctx)

    def stop(
        self,
        client_sock: zmq.Socket | None = None,
        worker_sock: zmq.Socket | None = None,
        ctx: zmq.Context | None = None,
    ) -> None:
        """Gracefully stop scheduler and all workers."""
        logger.info("Stopping scheduler and workers...")
        self._running = False

        if worker_sock:
            for worker_id in list(self.worker_procs):
                try:
                    worker_sock.send_multipart(
                        [
                            worker_id.encode("utf-8"),
                            json.dumps({"type": "SHUTDOWN"}).encode("utf-8"),
                        ],
                        flags=zmq.NOBLOCK,
                    )
                except Exception:
                    pass

        # Terminate worker processes
        for proc in self.worker_procs.values():
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        if client_sock:
            client_sock.close(linger=0)
        if worker_sock:
            worker_sock.close(linger=0)
        if ctx:
            ctx.term()

        self.db.close()
        logger.info("Scheduler stopped cleanly.")
