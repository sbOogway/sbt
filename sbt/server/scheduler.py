"""Scheduler daemon for managing backtest jobs and worker processes."""

import json
import logging
import subprocess
import sys
from collections import deque
from pathlib import Path

import zmq

from ..core.config import RunConfig
from ..core.db import ResultStore
from ..core.job import BacktestJob, BacktestResult, JobStatus

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("Scheduler")


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
        self.busy_workers: dict[str, str] = {}  # worker_id -> job_id
        self.worker_procs: list[subprocess.Popen] = []
        self._running = False

    def _spawn_workers(self) -> None:
        """Spawn the fixed worker subprocesses."""
        logger.info(
            "Spawning %d worker processes in %s...",
            self.num_workers,
            self.worktree_root,
        )
        for i in range(self.num_workers):
            worker_id = f"worker-{i}"
            wt_path = self.worktree_root / worker_id
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
            proc = subprocess.Popen(cmd)
            self.worker_procs.append(proc)
            logger.info("Spawned %s (PID %d)", worker_id, proc.pid)

    def _dispatch(self, worker_sock: zmq.Socket) -> None:
        """Dispatch pending jobs to available idle workers."""
        while self.job_queue and self.idle_workers:
            worker_id = self.idle_workers.pop()
            job = self.job_queue.popleft()
            job.status = JobStatus.RUNNING
            job.worker_id = worker_id

            self.db.update_job_status(job.id, JobStatus.RUNNING, worker_id)
            self.busy_workers[worker_id] = job.id

            logger.info(
                "Dispatching job %s (%s) to %s",
                job.id,
                job.config.strategy_name,
                worker_id,
            )
            worker_sock.send_multipart(
                [
                    worker_id.encode("utf-8"),
                    json.dumps({"type": "JOB", "job": job.to_dict()}).encode("utf-8"),
                ]
            )

    def _handle_client_req(
        self, client_sock: zmq.Socket, worker_sock: zmq.Socket
    ) -> None:
        """Process requests arriving from client CLI / optimizer."""
        msg_parts = client_sock.recv_multipart()
        if len(msg_parts) < 2:
            return
        client_id = msg_parts[0]
        try:
            req = json.loads(msg_parts[1].decode("utf-8"))
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
            study_name = req.get("study_name")
            if study_name:
                results = self.db.get_study_results(study_name)
            else:
                jobs = self.db.list_jobs(JobStatus.DONE)
                results = []
                for j in jobs:
                    r = self.db.get_result(j.id)
                    if r:
                        results.append(r)
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
            self._dispatch(worker_sock)

        elif msg_type == "RESULT":
            job_id = msg.get("job_id")
            result_dict = msg.get("result", {})
            result = BacktestResult.from_dict(result_dict)
            self.db.save_result(result)
            self.db.update_job_status(job_id, result.status, worker_id)

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
            self.idle_workers.add(worker_id)
            self._dispatch(worker_sock)

    def start(self) -> None:
        """Start the scheduler event loop."""
        self._running = True
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
            for i in range(self.num_workers):
                worker_id = f"worker-{i}"
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
        for proc in self.worker_procs:
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
