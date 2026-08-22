"""Worker process running in an isolated git worktree."""

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import zmq

from ..core.job import BacktestJob, BacktestResult, JobStatus
from ..core.runner import BacktestRunner

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("Worker")


def ensure_worktree(worktree_path: Path, repo_root: Path) -> bool:
    """Create a git worktree for this worker and symlink the data dir.

    Returns True when the directory holds its own checkout (linked git
    worktree or standalone clone, detected via a `.git` entry), i.e. code
    isolation is real. Returns False when only a plain mkdir fallback was
    possible — callers must warn loudly because imports will fall through
    to another tree.
    """
    worktree_path = worktree_path.resolve()
    repo_root = repo_root.resolve()

    if not worktree_path.exists():
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Creating git worktree at %s", worktree_path)
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree_path), "HEAD"],
                cwd=str(repo_root),
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as e:
            logger.warning(
                "git worktree creation failed: %s; falling back to plain mkdir "
                "(DEGRADED code isolation)",
                e,
            )
            worktree_path.mkdir(parents=True, exist_ok=True)

    # Symlink data dir
    data_target = repo_root / "data"
    worktree_data = worktree_path / "data"
    if data_target.exists() and not worktree_data.exists():
        try:
            worktree_data.symlink_to(data_target)
            logger.info("Symlinked data dir: %s -> %s", worktree_data, data_target)
        except Exception as e:
            logger.warning("Failed to symlink data dir: %s", e)

    # Ensure reports dir exists
    (worktree_path / "reports").mkdir(parents=True, exist_ok=True)

    return (worktree_path / ".git").exists()


def cleanup_worktree(worktree_path: Path, repo_root: Path) -> None:
    """Remove git worktree if exists."""
    worktree_path = worktree_path.resolve()
    repo_root = repo_root.resolve()
    if worktree_path.exists():
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
            )
        except Exception as e:
            logger.warning("Failed to remove git worktree %s: %s", worktree_path, e)


class Worker:
    """Worker node communicating with Scheduler via ZMQ DEALER socket."""

    def __init__(
        self,
        worker_id: str,
        worktree_path: Path,
        repo_root: Path,
        scheduler_endpoint: str = "tcp://127.0.0.1:5556",
    ) -> None:
        self.worker_id = worker_id
        self.worktree_path = worktree_path.resolve()
        self.repo_root = repo_root.resolve()
        self.scheduler_endpoint = scheduler_endpoint
        self._running = True

    def run_job(self, job: BacktestJob) -> BacktestResult:
        """Execute a backtest job in the worker's worktree context."""
        logger.info(
            "[%s] Executing job %s (%s)",
            self.worker_id,
            job.id,
            job.config.strategy_name,
        )
        # Test hook: artificial per-job latency so durability tests can
        # interrupt jobs deterministically (real backtests may finish in ms).
        delay = float(os.environ.get("SBT_JOB_DELAY_S", "0"))
        if delay > 0:
            logger.info(
                "[%s] Job %s: sleeping %.1fs (SBT_JOB_DELAY_S)",
                self.worker_id,
                job.id,
                delay,
            )
            time.sleep(delay)
        # Ensure we point to the worktree data / config
        orig_cwd = os.getcwd()
        try:
            if self.worktree_path.exists():
                os.chdir(self.worktree_path)

            # Ensure data_dir is resolved
            config = job.config
            if not config.feather_path and self.worktree_path.exists():
                # Allow runner to find data in worktree/data (symlinked to repo)
                config.data_dir = str(self.worktree_path / "data")

            runner = BacktestRunner(config)
            result = runner.run(job_id=job.id)
            return result
        except Exception as e:
            logger.exception(
                "[%s] Job %s failed with exception: %s", self.worker_id, job.id, e
            )
            return BacktestResult(
                job_id=job.id,
                status=JobStatus.FAILED,
                error=str(e),
            )
        finally:
            os.chdir(orig_cwd)

    def start(self) -> None:
        """Main worker event loop."""
        ensure_worktree(self.worktree_path, self.repo_root)
        ctx = zmq.Context()
        sock = ctx.socket(zmq.DEALER)
        sock.setsockopt_string(zmq.IDENTITY, self.worker_id)
        sock.connect(self.scheduler_endpoint)
        logger.info(
            "[%s] Connected to scheduler at %s", self.worker_id, self.scheduler_endpoint
        )

        # Notify scheduler we are ready
        sock.send_json({"type": "READY", "worker_id": self.worker_id})

        poller = zmq.Poller()
        poller.register(sock, zmq.POLLIN)

        try:
            while self._running:
                events = dict(poller.poll(timeout=1000))
                if sock in events and events[sock] == zmq.POLLIN:
                    msg = sock.recv_json()
                    msg_type = msg.get("type")

                    if msg_type == "SHUTDOWN":
                        logger.info("[%s] Received SHUTDOWN signal", self.worker_id)
                        break
                    elif msg_type == "JOB":
                        job_dict = msg.get("job")
                        job = BacktestJob.from_dict(job_dict)
                        # ACK receipt *before* running so the scheduler's ACK
                        # timer measures delivery, not execution time.
                        sock.send_json(
                            {
                                "type": "ACK",
                                "worker_id": self.worker_id,
                                "job_id": job.id,
                            }
                        )
                        result = self.run_job(job)
                        # default=str: report records may carry Timestamps
                        sock.send_string(
                            json.dumps(
                                {
                                    "type": "RESULT",
                                    "worker_id": self.worker_id,
                                    "job_id": job.id,
                                    "result": result.to_dict(),
                                },
                                default=str,
                            )
                        )
                    elif msg_type == "PING":
                        sock.send_json({"type": "PONG", "worker_id": self.worker_id})
                    else:
                        logger.warning(
                            "[%s] Unknown message type: %s", self.worker_id, msg_type
                        )

        except KeyboardInterrupt:
            logger.info("[%s] Worker interrupted", self.worker_id)
        finally:
            sock.close(linger=0)
            ctx.term()
            logger.info("[%s] Worker stopped", self.worker_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SBT Backtest Worker")
    parser.add_argument("--worker-id", required=True, help="Unique worker identifier")
    parser.add_argument("--worktree", required=True, help="Path to worker git worktree")
    parser.add_argument("--repo-root", default=".", help="Root repo directory")
    parser.add_argument(
        "--endpoint", default="tcp://127.0.0.1:5556", help="Scheduler worker endpoint"
    )
    args = parser.parse_args()

    worker = Worker(
        worker_id=args.worker_id,
        worktree_path=Path(args.worktree),
        repo_root=Path(args.repo_root),
        scheduler_endpoint=args.endpoint,
    )
    worker.start()
