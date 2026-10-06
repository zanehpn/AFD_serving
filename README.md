# AFD_serving

ECO configuration search for attention-FFN disaggregated serving. The default
implementation uses the paper's **measured four-stage model, finite-microbatch
FIFO scheduling, and Gaussian-process residual correction**.

## Energy model

1. Collect Attention, dispatch, FFN, and combine timing records from calibration
   requests, together with aligned NVML power and operating-state measurements.
2. Fit nonnegative stage-time regressions and structure-specific responses to
   frequency and power controls. Missing or unidentifiable stage feedback fails
   the trial; it is not replaced with replay-duration labels.
3. Simulate finite microbatches through FIFO resources. Each microbatch finishes
   combine before entering the next layer. Dispatch and combine share a resource
   by default; unavailable per-layer records use uniform stage-service weights.
4. Compute the log-energy prior as
   `log(N * predicted_power / min(arrival_rate, predicted_capacity))`.
   A Matern-5/2 GP fits standardized log-energy residuals. Refitting the mechanism
   recomputes residual targets before the GP is fitted again.
5. Use feasibility- and cost-aware acquisition, measured service constraints, and
   calibration-only feedback. Freeze the lowest-energy measured feasible point.
   Held-out requests never update the search model.

The campaign identifies this contract as `four_stage_fifo_v1` with
`require_four_stage=true`. Successful observations must include fitted stage
models and a finite-microbatch FIFO schedule. Unsupported independent-replica
feedback fails instead of falling back to an end-to-end energy prior.

## Entry points

Run commands from the repository root. A dry run does not load models, allocate
GPUs, create a campaign, or claim execution compatibility:

```bash
python3 bo_dse/native.py start --directory results/paper-preview --rps 8 --dry-run
```

`bo_dse/official.py` also defaults to this paper implementation. The paper path
requires the instrumented runtime, installed with `bash migration/setup_native.sh`
(or explicitly `--backend paper`) on a suitable GPU host. Installation changes
that environment; inspect the script before running it. CPU dependencies are
listed in `bo_dse/requirements-cpu.txt`.

A physical campaign needs model weights, disjoint calibration/held-out traces,
and the instrumented runtime. Supply traces through `--calibration` and
`--heldout`. Use a new campaign directory after a source/model change. Total
`--evaluations` includes preparation and failed measurements; inspect dry-run
accounting rather than treating that option as an uncharged search budget.
The native deployment checks remain platform-specific; CPU tests do not certify
physical execution on every platform in the paper.

## Historical observable-only backend

The uninstrumented upstream runtime cannot provide four-stage timing data.
Its historical `external_power_duration_v1` prior remains available only through
an explicit backend selection:

```bash
python3 bo_dse/native.py --backend legacy-observable start \
  --directory results/legacy-preview --dry-run
```

This mode uses replay duration and role power, not the paper's FIFO prior.
`--backend official` is a compatibility alias for this historical mode;
`--backend customized` aliases the instrumented paper path. Upstream-only
installation also requires explicit `--backend legacy-observable`.
Historical archive/queue scripts retain their original experiment definitions
and are not the entry point for new paper-method runs.

## Source layout

- `bo_dse/native.py`: instrumented calibration, search, and measurement feedback.
- `bo_dse/scripts/afd/static_dse/`: candidate space, GP acquisition, BO/GA/Random,
  stage feedback, and configuration freezing.
- `bo_dse/scripts/afd/four_stage_dse_v6/`: stage models and FIFO simulator.
- `migration/`, `scripts/`, `services/`: installation, replay, telemetry, and runtime tools.
- `environment/`, `inputs/`: dependency locks and runtime configuration templates.
- `patches/`, `runtime-patches/`: instrumented runtime changes.
- `tests/`, `bo_dse/tests/`: CPU and integration tests with simulated measurements.

The original source snapshot came from MOE_DVFS commit
`9193339ae91495d979d424bdc76d0775a5883f96`. This repository contains code and
configuration, not the manuscript, plotting data, archived results, request
traces, model weights, or historical calibration inputs. The original
`analyze_m1_repetitions.py` is retained.

## Verification and publication

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 bash bo_dse/test_cpu.sh
PYTHONDONTWRITEBYTECODE=1 python3 tools/check_public_code.py
```

Comments, docstrings, READMEs, and generated report text are in English.
Project paths are relative or derived from script locations. CUDA tools use
`PATH`; set `ECODEP_CUDA_COMPATIBILITY_PATH` explicitly when needed. OS interfaces,
interpreter shebangs, and container mount paths are runtime conventions, not
private host directories.

The publication scan reports paths, line numbers, and rule names without printing
sensitive values. Git history and binary files need separate review. Do not commit
credentials, real request data, or generated runtime logs. Custom bundle commit
identities are anonymized; its runtime source is preserved and test paths are
relative. Current source/configuration hashes and bundle commit IDs differ from
historical experiments and must not be presented as their original frozen evidence.

The default-model correction does not regenerate historical measurements or
establish the paper's multi-seed, held-out, or ablation results. Those require
matching raw records or new experiments under the declared protocol.
