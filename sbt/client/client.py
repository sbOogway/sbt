"""ZMQ Client for communicating with the SBT Scheduler."""

import json
import logging
import uuid
import zmq

from ..core.config import RunConfig

logger = logging.getLogger("Client")


class SbtClient:
    """Client for dispatching and tracking backtests via the Scheduler daemon."""

    def __init__(self, endpoint: str = "tcp://127.0.0.1:5555", timeout_ms: int = 5000) -> None:
        self.endpoint = endpoint
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context()

    def _request(self, payload: dict) -> dict:
        """Send request to scheduler and await JSON response."""
        sock = self.ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        sock.connect(self.endpoint)

        try:
            sock.send_json(payload)
            resp = sock.recv_json()
            return resp
        except zmq.Again:
            raise TimeoutError(f"Request to scheduler at {self.endpoint} timed out.")
        finally:
            sock.close(linger=0)

    def ping(self) -> bool:
        """Check if scheduler is alive."""
        try:
            resp = self._request({"action": "ping"})
            return resp.get("status") == "ok"
        except Exception:
            return False

    def submit(self, config: RunConfig, study_name: str | None = None) -> str:
        """Submit a single backtest job."""
        resp = self._request({
            "action": "submit",
            "config": config.to_dict(),
            "study_name": study_name,
        })
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("error", "Failed to submit job"))
        return resp["job_id"]

    def submit_batch(self, configs: list[RunConfig], study_name: str | None = None) -> list[str]:
        """Submit a batch of backtest jobs."""
        resp = self._request({
            "action": "submit_batch",
            "configs": [c.to_dict() for c in configs],
            "study_name": study_name,
        })
        if resp.get("status") != "ok":
            raise RuntimeError(resp.get("error", "Failed to submit batch"))
        return resp["job_ids"]

    def get_status(self) -> dict:
        """Get scheduler and worker status summary."""
        return self._request({"action": "status"})

    def get_result(self, job_id: str) -> dict:
        """Get execution result for a job."""
        return self._request({"action": "get_result", "job_id": job_id})

    def list_results(self, study_name: str | None = None) -> list[dict]:
        """List completed results."""
        resp = self._request({"action": "list_results", "study_name": study_name})
        return resp.get("results", [])

    def close(self) -> None:
        self.ctx.term()
