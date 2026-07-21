#!/usr/bin/env bash
# Laguna S 2.1 NVFP4 launch contract for one GB10 / SM121.
set -euo pipefail

AUDIT_BIN=/usr/local/bin/audit_runtime.py

if [[ "${1:-}" == "audit" ]]; then
    exec "$AUDIT_BIN"
fi

KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
joined_args=" ${*,,} "
if [[ "${KV_CACHE_DTYPE,,}" == "nvfp4" ]] \
    || [[ "$joined_args" == *" --kv-cache-dtype nvfp4 "* ]] \
    || [[ "$joined_args" == *" --kv-cache-dtype=nvfp4 "* ]]; then
    printf '%s\n' 'R0B0TLAB_LAUNCH_REJECTED: NVFP4 KV is not qualified; use fp8' >&2
    exit 64
fi
if [[ "$joined_args" == *"flashinfer_b12x"* ]]; then
    printf '%s\n' 'R0B0TLAB_LAUNCH_REJECTED: do not force flashinfer_b12x for this release' >&2
    exit 65
fi

"$AUDIT_BIN"

if (( $# > 0 )); then
    exec "$@"
fi

MODEL_PATH="${MODEL_PATH:-/models/Laguna-S-2.1-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-216d1f13878dd4e715bc7412848d0f330e95bba6}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/Laguna-S-2.1-DFlash-NVFP4}"
DRAFT_MODEL_REVISION="${DRAFT_MODEL_REVISION:-723794750422b3efbf3a7b3af76dffb4ba035943}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-poolside/Laguna-S-2.1-NVFP4}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DFLASH_TOKENS="${DFLASH_TOKENS:-0}"

args=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --tensor-parallel-size 1
    --dtype bfloat16
    --trust-remote-code
    --attention-backend FLASHINFER
    --kv-cache-dtype "$KV_CACHE_DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --reasoning-parser poolside_v1
    --enable-auto-tool-choice
    --tool-call-parser poolside_v1
    --override-generation-config '{"temperature":0.7,"top_p":0.95,"top_k":20}'
)

if [[ "$MODEL_PATH" != /* ]] && [[ -n "$MODEL_REVISION" ]]; then
    args+=(--revision "$MODEL_REVISION")
fi

if [[ "$DFLASH_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    spec_json="$(DRAFT_MODEL_PATH="$DRAFT_MODEL_PATH" DRAFT_MODEL_REVISION="$DRAFT_MODEL_REVISION" DFLASH_TOKENS="$DFLASH_TOKENS" python3 -c 'import json,os; value={"method":"dflash","model":os.environ["DRAFT_MODEL_PATH"],"num_speculative_tokens":int(os.environ["DFLASH_TOKENS"])}; value.update({"revision":os.environ["DRAFT_MODEL_REVISION"]} if not os.environ["DRAFT_MODEL_PATH"].startswith("/") else {}); print(json.dumps(value,separators=(",",":")))')"
    args+=(--speculative-config "$spec_json")
elif [[ "$DFLASH_TOKENS" != "0" ]]; then
    printf 'R0B0TLAB_LAUNCH_REJECTED: DFLASH_TOKENS must be 0 or a positive integer, got %q\n' "$DFLASH_TOKENS" >&2
    exit 66
fi

printf 'R0B0TLAB_LAUNCH='
printf '%q ' "${args[@]}"
printf '\n'
exec "${args[@]}"
