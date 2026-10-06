"""Explicit external executor bridge; the DSE optimizer never silently starts GPUs."""
from __future__ import annotations

import subprocess
import fcntl
from pathlib import Path

from .campaign import ask, read_json, tell, write_json


def complete_feedback(request, result):
    """Raw collectors may return a hashed feedback manifest instead of labels."""
    if result.get("status") == "ok" and result.get("feedback_manifest"):
        from .feedback import attach_feedback
        from .campaign import file_hash
        item = result["feedback_manifest"]
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("feedback manifest hash mismatch")
        return attach_feedback(request, result, item["path"])
    return result


def run_one(directory, command, timeout_seconds=1800):
    """Executor receives --request PATH --result PATH and owns cleanup/receipts.

    A crash leaves the trial pending and blocks further asks until its complete
    result or charged failure is recorded. Existing results resume without rerun.
    """
    request = ask(directory)
    if request.get("stopped"):
        return request
    trial = Path(directory).resolve() / "trials" / request["trial_id"]
    trial.mkdir(parents=True, exist_ok=True)
    request_path, result_path = trial / "request.json", trial / "result.json"
    with (trial / ".executor.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if request_path.exists():
            if read_json(request_path) != request:
                raise ValueError("trial request artifact changed")
        else:
            write_json(request_path, request)
        if result_path.exists():
            return tell(directory, complete_feedback(request, read_json(result_path)))
        if not command:
            raise ValueError("an explicit evaluator argv is required")
        # Never interpret evaluator arguments through a shell.
        with (trial / "executor.log").open("a") as log:
            completed = subprocess.run([*command, "--request", str(request_path), "--result", str(result_path)],
                                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout_seconds, check=False)
        if completed.returncode != 0 and not result_path.exists():
            raise RuntimeError(f"executor failed ({completed.returncode}); record a charged failure for {request['trial_id']} before continuing; see {trial / 'executor.log'}")
        if not result_path.exists():
            raise RuntimeError("executor did not write a receipt; trial remains pending")
        result = read_json(result_path)
        if completed.returncode != 0:
            result["status"] = "failed"
            result["failure_reason"] = f"executor exited {completed.returncode}, including possible cleanup failure"
            write_json(result_path, result)
        result = complete_feedback(request, result)
        write_json(result_path, result)
        return tell(directory, result)
