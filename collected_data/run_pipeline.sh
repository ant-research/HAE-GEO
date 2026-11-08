#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output}"
URL_INPUT="${1:-${SCRIPT_DIR}/manifests/clean_urls.jsonl}"

if [[ ! -f "${URL_INPUT}" ]]; then
  if [[ $# -gt 0 ]]; then
    printf 'Input file does not exist: %s\n' "${URL_INPUT}" >&2
    exit 1
  fi
  python3 "${SCRIPT_DIR}/corpus_tools.py" manifest
fi

COLLECT_ARGS=(
  --input "${URL_INPUT}"
  --output "${OUTPUT_DIR}/clean.jsonl"
  --manifest "${OUTPUT_DIR}/crawl_manifest.jsonl"
  --limit "${LIMIT:-50}"
  --delay-ms "${DELAY_MS:-300}"
)
if [[ "${RETRY_FAILED:-0}" == "1" ]]; then
  COLLECT_ARGS+=(--retry-failed)
fi

node "${SCRIPT_DIR}/collect_clean_corpus.mjs" "${COLLECT_ARGS[@]}"

python3 "${SCRIPT_DIR}/corpus_tools.py" build \
  --clean "${OUTPUT_DIR}/clean.jsonl" \
  --output-dir "${OUTPUT_DIR}/environments"

printf 'Environment config: %s\n' "${OUTPUT_DIR}/environments/environments.json"
