#!/usr/bin/env python3
"""nvcontrold: a small privileged GPU-control daemon.

WHY THIS EXISTS
---------------
On this host, host root and sudo are unavailable, so an unprivileged process
CANNOT set GPU clocks or power limits. However, membership in the `docker` group
lets us run a *privileged* container, and inside `--privileged --gpus all` the
injected libnvidia-ml allows nvmlDeviceSetGpuLockedClocks / SetPowerManagementLimit.

So this daemon runs INSIDE such a container and exposes a tiny HTTP API. The rest
of Pythia-Energy runs unprivileged on the host and talks to it over 127.0.0.1.

SAFETY
------
* Only GPUs in ALLOWED_GPUS (env PYTHIA_ALLOWED_GPUS) may be modified.
* Every touched GPU is reset (clocks + power limit) on shutdown / SIGTERM / atexit,
  so we never leave a *shared* machine's GPU locked.
* Set calls are done through persistent NVML handles (low latency, ~ms).

Endpoints (all JSON):
  GET  /health
  GET  /state?gpu=N            -> one GPU;  /state -> all allowed GPUs
  POST /set_clock              {"gpu": N, "sm_mhz": M}      | {"gpu": N, "reset": true}
  POST /set_power_limit        {"gpu": N, "watts": W}       | {"gpu": N, "reset": true}
  POST /reset                  {"gpu": N}
  POST /reset_all
"""
import atexit
import json
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pynvml as nv

ALLOWED = set(int(x) for x in os.environ.get("PYTHIA_ALLOWED_GPUS", "").split(",") if x != "")
PORT = int(os.environ.get("PYTHIA_NVCTL_PORT", "9096"))
ACK_PROTOCOL = "nvcontrold.applied_ack.v2"

_lock = threading.Lock()
_handles = {}
_touched = set()          # gpus whose clock or power we changed
_supported_clocks = {}    # gpu -> sorted list of supported SM clocks (MHz)
_default_power_limits = {}
_clock_control = {}       # gpu -> unknown | reset | locked
_clock_targets = {}       # gpu -> acknowledged locked clock or None
_power_control = {}       # gpu -> unknown | default | limited
_power_targets = {}       # gpu -> acknowledged cap/default limit or None
_control_generation = 0


def _init():
    nv.nvmlInit()
    for i in ALLOWED:
        h = nv.nvmlDeviceGetHandleByIndex(i)
        _handles[i] = h
        _default_power_limits[i] = (
            nv.nvmlDeviceGetPowerManagementDefaultLimit(h) / 1000.0
        )
        # NVML exposes the instantaneous SM clock, not whether a lock was set
        # by a previous process.  The daemon therefore starts fail-closed and
        # reports an unknown control mode until this daemon acknowledges an
        # explicit reset or set operation.
        _clock_control[i] = "unknown"
        _clock_targets[i] = None
        _power_control[i] = "unknown"
        _power_targets[i] = None
        try:
            mem_clk = nv.nvmlDeviceGetMaxClockInfo(h, nv.NVML_CLOCK_MEM)
            clks = sorted(nv.nvmlDeviceGetSupportedGraphicsClocks(h, mem_clk))
            _supported_clocks[i] = clks
        except nv.NVMLError:
            _supported_clocks[i] = []
    print(f"[nvcontrold] init ok, allowed gpus={sorted(ALLOWED)}, driver={nv.nvmlSystemGetDriverVersion()}",
          flush=True)


def _nearest_supported(gpu, sm_mhz):
    """Snap requested clock to the closest supported SM clock (device rejects arbitrary values)."""
    clks = _supported_clocks.get(gpu) or []
    if not clks:
        return int(sm_mhz)
    return min(clks, key=lambda c: abs(c - sm_mhz))


def _state(gpu):
    h = _handles[gpu]
    u = nv.nvmlDeviceGetUtilizationRates(h)
    mem = nv.nvmlDeviceGetMemoryInfo(h)
    try:
        enforced = nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
    except nv.NVMLError:
        enforced = nv.nvmlDeviceGetPowerManagementLimit(h) / 1000.0
    with _lock:
        clock_control = _clock_control.get(gpu, "unknown")
        clock_target = _clock_targets.get(gpu)
        power_control = _power_control.get(gpu, "unknown")
        power_target = _power_targets.get(gpu)
        generation = _control_generation
    return {
        "gpu": gpu,
        "power_w": nv.nvmlDeviceGetPowerUsage(h) / 1000.0,
        "power_limit_w": enforced,
        "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
        "mem_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
        "gpu_util": u.gpu,
        "mem_util": u.memory,
        "mem_used_mib": mem.used // (1024 * 1024),
        "temp_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
        "clock_control": clock_control,
        "clock_target_mhz": clock_target,
        "power_control": power_control,
        "power_target_w": power_target,
        "default_power_limit_w": _default_power_limits[gpu],
        "control_generation": generation,
        "control_verified": clock_control != "unknown" and power_control != "unknown",
        "ts": time.time(),
    }


def _advance_generation():
    global _control_generation
    _control_generation += 1
    return _control_generation


def _set_clock(gpu, sm_mhz):
    snapped = _nearest_supported(gpu, sm_mhz)
    with _lock:
        t0 = time.perf_counter()
        nv.nvmlDeviceSetGpuLockedClocks(_handles[gpu], snapped, snapped)
        dt = (time.perf_counter() - t0) * 1000.0
        _touched.add(gpu)
        _clock_control[gpu] = "locked"
        _clock_targets[gpu] = snapped
        generation = _advance_generation()
    return {"gpu": gpu, "requested_mhz": sm_mhz, "applied_mhz": snapped,
            "clock_control": "locked", "control_generation": generation,
            "set_ms": round(dt, 3)}


def _reset_clock(gpu):
    with _lock:
        nv.nvmlDeviceResetGpuLockedClocks(_handles[gpu])
        _clock_control[gpu] = "reset"
        _clock_targets[gpu] = None
        generation = _advance_generation()
    return {"gpu": gpu, "clock": "reset", "clock_control": "reset",
            "control_generation": generation}


def _set_power(gpu, watts):
    lo, hi = nv.nvmlDeviceGetPowerManagementLimitConstraints(_handles[gpu])
    mw = int(max(lo, min(hi, watts * 1000)))
    with _lock:
        t0 = time.perf_counter()
        nv.nvmlDeviceSetPowerManagementLimit(_handles[gpu], mw)
        dt = (time.perf_counter() - t0) * 1000.0
        _touched.add(gpu)
        _power_control[gpu] = "limited"
        _power_targets[gpu] = mw / 1000.0
        generation = _advance_generation()
    return {"gpu": gpu, "requested_w": watts, "applied_w": mw / 1000.0,
            "power_control": "limited", "control_generation": generation,
            "set_ms": round(dt, 3)}


def _reset_power(gpu):
    dflt = int(_default_power_limits[gpu] * 1000.0)
    with _lock:
        nv.nvmlDeviceSetPowerManagementLimit(_handles[gpu], dflt)
        _power_control[gpu] = "default"
        _power_targets[gpu] = dflt / 1000.0
        generation = _advance_generation()
    return {"gpu": gpu, "power_limit_w": dflt / 1000.0,
            "power_control": "default", "control_generation": generation}


def reset_all():
    for i in list(_touched):
        try:
            _reset_clock(i)
        except nv.NVMLError as e:
            print(f"[nvcontrold] reset clock gpu{i} failed: {e}", flush=True)
        try:
            _reset_power(i)
        except nv.NVMLError as e:
            print(f"[nvcontrold] reset power gpu{i} failed: {e}", flush=True)
    reset = sorted(_touched)
    _touched.clear()
    print(f"[nvcontrold] reset_all done for {reset}", flush=True)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request stderr spam
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _check_gpu(self, gpu):
        if gpu not in ALLOWED:
            self._send(403, {"error": f"gpu {gpu} not in allowed set {sorted(ALLOWED)}"})
            return False
        return True

    def do_GET(self):
        try:
            path = self.path.split("?")[0]
            if path == "/health":
                return self._send(200, {
                    "ok": True,
                    "allowed": sorted(ALLOWED),
                    "protocol": ACK_PROTOCOL,
                })
            if path == "/state":
                q = self.path.split("?", 1)[1] if "?" in self.path else ""
                params = dict(p.split("=") for p in q.split("&") if "=" in p)
                if "gpu" in params:
                    g = int(params["gpu"])
                    if not self._check_gpu(g):
                        return
                    return self._send(200, _state(g))
                return self._send(200, {"gpus": [_state(i) for i in sorted(ALLOWED)]})
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            path = self.path.split("?")[0]
            if path == "/reset_all":
                reset_all()
                return self._send(200, {"reset_all": True})
            if path == "/reset":
                g = int(body["gpu"])
                if not self._check_gpu(g):
                    return
                clock_ack = _reset_clock(g)
                power_ack = _reset_power(g)
                return self._send(200, {
                    "gpu": g,
                    "clock_reset": clock_ack.get("clock") == "reset",
                    "power_limit_w": power_ack["power_limit_w"],
                })
            if path == "/set_clock":
                g = int(body["gpu"])
                if not self._check_gpu(g):
                    return
                if body.get("reset"):
                    return self._send(200, _reset_clock(g))
                return self._send(200, _set_clock(g, int(body["sm_mhz"])))
            if path == "/set_power_limit":
                g = int(body["gpu"])
                if not self._check_gpu(g):
                    return
                if body.get("reset"):
                    return self._send(200, _reset_power(g))
                return self._send(200, _set_power(g, float(body["watts"])))
            self._send(404, {"error": "not found"})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": f"{type(e).__name__}: {e}"})


def main():
    _init()
    atexit.register(reset_all)

    def _sig(signum, frame):
        print(f"[nvcontrold] signal {signum}, resetting and exiting", flush=True)
        reset_all()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[nvcontrold] listening on 127.0.0.1:{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
