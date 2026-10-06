#!/usr/bin/env python3
"""Launch a persistent Attention/Expert vLLM pair inside one container."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/model")
    parser.add_argument("--attention-gpus", default="0,1,2,3")
    parser.add_argument("--expert-gpus", default="4,5,6,7")
    parser.add_argument("--attention-ranks", type=int, default=4)
    parser.add_argument("--expert-ranks", type=int, default=4)
    parser.add_argument("--attention-tp", type=int, default=1)
    parser.add_argument("--expert-tp", type=int, default=1)
    parser.add_argument("--api-port", type=int, default=18000)
    parser.add_argument("--expert-api-port", type=int, default=18001)
    parser.add_argument("--afd-port", type=int, default=16239)
    parser.add_argument("--served-model-name", default="ecodep-model")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--cpu-offload-gb", type=float, default=0.0)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--safetensors-load-strategy",
        choices=("lazy", "eager", "prefetch"),
        default="lazy",
        help=(
            "Use lazy mmap by default so four AFD workers do not each prefetch "
            "the full MoE checkpoint into a memory-limited container."
        ),
    )
    parser.add_argument("--enable-prefix-caching", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--cuda-graph-full-decode-only",
        type=int,
        choices=(0, 1),
        default=0,
        help=(
            "Enable the GPU AFD-supported FULL_DECODE_ONLY CUDA graph mode "
            "instead of forcing eager execution."
        ),
    )
    parser.add_argument("--cudagraph-capture-size", type=int, default=32)
    parser.add_argument("--enable-dbo", type=int, choices=(0, 1), default=0)
    parser.add_argument(
        "--compute-gate-on-attention",
        type=int,
        choices=(0, 1),
        default=0,
    )
    parser.add_argument(
        "--p2p-wire-dtype",
        choices=("native_bf16", "int8_per_tensor"),
        default="native_bf16",
        help="P2P hidden-state wire format; INT8 is an accuracy-changing ablation.",
    )
    parser.add_argument("--dbo-decode-token-threshold", type=int, default=2)
    parser.add_argument("--dbo-prefill-token-threshold", type=int, default=12)
    parser.add_argument(
        "--limit-mm-per-prompt",
        default="",
        help="Optional vLLM JSON limit; text-only Qwen runs use image/video=0.",
    )
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument(
        "--language-model-only",
        action="store_true",
        help="Use vLLM's text-only language-model lane (required by Qwen3.6).",
    )
    parser.add_argument("--results", type=Path, default=Path("/results"))
    return parser.parse_args()


def command(args: argparse.Namespace, role: str) -> list[str]:
    ranks = args.attention_ranks if role == "attention" else args.expert_ranks
    tp = args.attention_tp if role == "attention" else args.expert_tp
    dp = ranks // tp
    config = {
        "afd": {
            "role": "ffn" if role == "expert" else "attention",
            "connector": "P2pNcclAFDConnector",
            "host": "127.0.0.1",
            "port": args.afd_port,
            "num_attention_ranks": args.attention_ranks,
            "num_ffn_ranks": args.expert_ranks,
            "compute_gate_on_attention": bool(args.compute_gate_on_attention),
        }
    }
    if args.p2p_wire_dtype != "native_bf16":
        config["afd"]["connector_extra_config"] = {
            "wire_dtype": args.p2p_wire_dtype,
        }
    port = args.api_port if role == "attention" else args.expert_api_port
    served = args.served_model_name if role == "attention" else f"{args.served_model_name}-expert"
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        served,
        "--data-parallel-size",
        str(dp),
        "--tensor-parallel-size",
        str(tp),
        "--enable-expert-parallel",
        "--additional-config",
        json.dumps(config, separators=(",", ":")),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--cpu-offload-gb",
        str(args.cpu_offload_gb),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--safetensors-load-strategy",
        args.safetensors_load_strategy,
        "--trust-remote-code",
    ]
    if args.cuda_graph_full_decode_only:
        capture_size = str(args.cudagraph_capture_size)
        cmd.extend(
            [
                "--max-cudagraph-capture-size",
                capture_size,
                "--cudagraph-capture-sizes",
                capture_size,
                "--compilation-config",
                json.dumps(
                    {"cudagraph_mode": "FULL_DECODE_ONLY"},
                    separators=(",", ":"),
                ),
            ]
        )
    else:
        cmd.append("--enforce-eager")
    if args.enable_dbo:
        cmd.extend(
            [
                "--enable-dbo",
                "--dbo-decode-token-threshold",
                str(args.dbo_decode_token_threshold),
                "--dbo-prefill-token-threshold",
                str(args.dbo_prefill_token_threshold),
            ]
        )
    mm_limit = args.limit_mm_per_prompt
    config_path = Path(args.model) / "config.json"
    if args.text_only and config_path.exists():
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        if "vision_config" in model_config:
            mm_limit = json.dumps({"image": 0, "video": 0}, separators=(",", ":"))
    if mm_limit:
        cmd.extend(["--limit-mm-per-prompt", mm_limit])
    if args.language_model_only:
        cmd.append("--language-model-only")
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    else:
        cmd.append("--no-enable-prefix-caching")
    return cmd


def main() -> int:
    args = parse_args()
    if args.attention_ranks % args.attention_tp or args.expert_ranks % args.expert_tp:
        raise ValueError("rank count must be divisible by TP size")
    if args.cudagraph_capture_size <= 0:
        raise ValueError("CUDA graph capture size must be positive")
    if args.dbo_decode_token_threshold <= 0 or args.dbo_prefill_token_threshold <= 0:
        raise ValueError("DBO thresholds must be positive")
    args.results.mkdir(parents=True, exist_ok=True)
    children: list[subprocess.Popen[str]] = []
    logs = []
    stopping = False

    def stop_children(*_: object) -> None:
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGINT)

    signal.signal(signal.SIGTERM, stop_children)
    signal.signal(signal.SIGINT, stop_children)

    for role, gpu_list in (
        ("expert", args.expert_gpus),
        ("attention", args.attention_gpus),
    ):
        log = (args.results / f"{role}.log").open("w", encoding="utf-8")
        logs.append(log)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list
        cmd = command(args, role)
        print(f"launching {role}: {' '.join(cmd)}", flush=True)
        children.append(
            subprocess.Popen(
                cmd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )

    try:
        while not stopping:
            for role, child in zip(("expert", "attention"), children, strict=True):
                code = child.poll()
                if code is not None:
                    print(f"{role} exited with status {code}", file=sys.stderr)
                    stop_children()
                    return code or 1
            time.sleep(1)
    finally:
        stop_children()
        deadline = time.monotonic() + 30
        for child in children:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                child.kill()
        for log in logs:
            log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
