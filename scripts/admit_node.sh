#!/usr/bin/env bash
# Fail-closed admission for Laguna S 2.1 heavy work.
set -euo pipefail

EXPECTED_HOST="${EXPECTED_HOST:-gn100-2eea}"
MIN_FREE_GIB="${MIN_FREE_GIB:-180}"
EVIDENCE_DIR="${EVIDENCE_DIR:-$HOME/laguna-s21-work/evidence/admission}"
ALLOW_PROJECT_CONTAINERS="${ALLOW_PROJECT_CONTAINERS:-0}"
mkdir -p "$EVIDENCE_DIR"

if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
    printf 'ADMISSION_FAIL: expected host %s, got %s\n' "$EXPECTED_HOST" "$(hostname)" >&2
    exit 40
fi

mapfile -t containers < <(docker ps --no-trunc --format '{{.ID}}|{{.Names}}|{{.Image}}|{{.Labels}}')
if (( ${#containers[@]} > 0 )); then
    if [[ "$ALLOW_PROJECT_CONTAINERS" != "1" ]]; then
        printf 'ADMISSION_FAIL: running containers exist\n' >&2
        printf '%s\n' "${containers[@]}" >&2
        exit 41
    fi
    for row in "${containers[@]}"; do
        if [[ "$row" != *"io.r0b0tlab.project=laguna-s-2.1-nvfp4"* ]]; then
            printf 'ADMISSION_FAIL: unowned running container: %s\n' "$row" >&2
            exit 42
        fi
    done
fi

python3 - "$MIN_FREE_GIB" "$EVIDENCE_DIR" <<'PY'
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys

minimum = int(sys.argv[1]) * 1024**3
evidence = Path(sys.argv[2])
free = shutil.disk_usage('/').free
if free < minimum:
    raise SystemExit(f'ADMISSION_FAIL: disk_free_bytes={free} minimum={minimum}')

ancestors = set()
pid = os.getpid()
while pid > 1:
    ancestors.add(pid)
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split()
        pid = int(fields[3])
    except Exception:
        break
forbidden = ('vllm', 'sglang', 'lm_eval', 'benchmark', 'ninja', 'nvcc', 'ptxas', 'cicc')
violations = []
for item in Path('/proc').iterdir():
    if not item.name.isdigit() or int(item.name) in ancestors:
        continue
    try:
        cmd = (item / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
    except OSError:
        continue
    lowered = cmd.lower()
    if cmd and any(word in lowered for word in forbidden):
        violations.append({'pid': int(item.name), 'cmd': cmd[:500]})
if violations:
    print(json.dumps(violations, indent=2), file=sys.stderr)
    raise SystemExit('ADMISSION_FAIL: competing build/model/benchmark processes exist')

meminfo = {}
for line in Path('/proc/meminfo').read_text().splitlines():
    key, _, value = line.partition(':')
    if value:
        meminfo[key] = int(value.strip().split()[0]) * 1024
cap = subprocess.check_output(
    ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], text=True
).strip().splitlines()
if cap != ['12.1']:
    raise SystemExit(f'ADMISSION_FAIL: expected one SM121 GPU, got {cap!r}')
report = {
    'schema_version': 1,
    'status': 'PASS',
    'captured_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    'hostname': socket.gethostname(),
    'disk_free_bytes': free,
    'mem_available_bytes': meminfo.get('MemAvailable'),
    'swap_total_bytes': meminfo.get('SwapTotal'),
    'swap_free_bytes': meminfo.get('SwapFree'),
    'compute_capability': cap,
    'containers': [],
    'competing_processes': [],
}
path = evidence / 'admission.json'
path.write_text(json.dumps(report, indent=2) + '\n')
print(f'ADMISSION_PASS evidence={path}')
PY
