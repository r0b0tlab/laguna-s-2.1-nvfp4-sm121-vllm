#!/usr/bin/env bash
# Durable, single-owner source build wrapper.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIVATE_ROOT="${PRIVATE_ROOT:-$HOME/laguna-s21-work/evidence/build}"
LOCK="$ROOT/docker/dependency-manifest.json"
IMAGE_REPOSITORY="${IMAGE_REPOSITORY:-laguna-s21-sm121-vllm}"
mkdir -p "$PRIVATE_ROOT"

if [[ "$(hostname)" != "gn100-2eea" ]]; then
    printf '%s\n' 'BUILD_FAIL: this build is admitted only on gn100-2eea' >&2
    exit 40
fi
if tmux has-session -t laguna-s21-download 2>/dev/null; then
    printf '%s\n' 'BUILD_FAIL: model download session is still active' >&2
    exit 41
fi
if [[ ! -f "$HOME/laguna-s21-work/evidence/phase0/download-pass.timestamp" ]]; then
    printf '%s\n' 'BUILD_FAIL: verified model download PASS is absent' >&2
    exit 42
fi
EVIDENCE_DIR="$PRIVATE_ROOT/admission" "$ROOT/scripts/admit_node.sh"

python3 "$ROOT/scripts/resolve_vllm_release.py" --write "$PRIVATE_ROOT/vllm-release-lock.json"
python3 - "$LOCK" "$PRIVATE_ROOT/vllm-release-lock.json" <<'PY'
import json,sys
manifest=json.load(open(sys.argv[1]))
lock=json.load(open(sys.argv[2]))
for key in ('version','tag','commit'):
    if manifest['vllm'][key] != lock['vllm'][key]:
        raise SystemExit(f'BUILD_FAIL: dependency manifest {key} is stale: {manifest["vllm"][key]} != {lock["vllm"][key]}')
print('LATEST_VLLM_LOCK_PASS')
PY
python3 - "$LOCK" "$ROOT/docker/runtime-manifest.production.json" <<'PY'
import json,sys
dependency=json.load(open(sys.argv[1]))
runtime=json.load(open(sys.argv[2]))
expected=dependency['vllm']
actual=(runtime.get('vllm_version'), runtime.get('vllm_tag'), runtime.get('vllm_commit'))
wanted=(expected['version'], expected['tag'], expected['commit'])
if actual != wanted:
    raise SystemExit(f'BUILD_FAIL: runtime manifest vLLM identity is stale: {actual!r} != {wanted!r}')
print('RUNTIME_MANIFEST_LOCK_PASS')
PY

cd "$ROOT"
if [[ -n "$(git status --porcelain)" ]]; then
    printf '%s\n' 'BUILD_FAIL: source tree must be clean' >&2
    git status --short >&2
    exit 43
fi
REVISION="$(git rev-parse HEAD)"
VLLM_VERSION="$(python3 -c 'import json; print(json.load(open("docker/dependency-manifest.json"))["vllm"]["version"])')"
VLLM_TAG="$(python3 -c 'import json; print(json.load(open("docker/dependency-manifest.json"))["vllm"]["tag"])')"
VLLM_COMMIT="$(python3 -c 'import json; print(json.load(open("docker/dependency-manifest.json"))["vllm"]["commit"])')"
FLASHINFER_VERSION="$(python3 -c 'import json; print(json.load(open("docker/dependency-manifest.json"))["flashinfer"]["version"])')"
IMAGE="${IMAGE_REPOSITORY}:v${VLLM_VERSION}-candidate"

sha256sum .dockerignore docker/Dockerfile.production docker/dependency-manifest.json docker/runtime-manifest.production.json scripts/entrypoint.sh scripts/audit_runtime.py scripts/check_dependencies.py > "$PRIVATE_ROOT/build-inputs.sha256"
printf '%s\n' "$REVISION" > "$PRIVATE_ROOT/source-revision.txt"
printf '%s\n' "$IMAGE" > "$PRIVATE_ROOT/image-tag.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$PRIVATE_ROOT/build-start.timestamp"
rm -f "$PRIVATE_ROOT/build-pass.timestamp" "$PRIVATE_ROOT/build-fail.timestamp"

set +e
MAX_JOBS=4 NVCC_THREADS=2 docker buildx build --load --progress=plain \
    --build-arg "VLLM_VERSION=$VLLM_VERSION" \
    --build-arg "VLLM_TAG=$VLLM_TAG" \
    --build-arg "VLLM_COMMIT=$VLLM_COMMIT" \
    --build-arg "FLASHINFER_VERSION=$FLASHINFER_VERSION" \
    --build-arg "IMAGE_REVISION=$REVISION" \
    -f docker/Dockerfile.production -t "$IMAGE" . 2>&1 | tee "$PRIVATE_ROOT/build.log"
rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$rc" > "$PRIVATE_ROOT/build.rc"
if (( rc == 0 )); then
    date -u +%Y-%m-%dT%H:%M:%SZ > "$PRIVATE_ROOT/build-pass.timestamp"
    docker image inspect "$IMAGE" > "$PRIVATE_ROOT/image-inspect.json"
else
    date -u +%Y-%m-%dT%H:%M:%SZ > "$PRIVATE_ROOT/build-fail.timestamp"
fi
exit "$rc"
