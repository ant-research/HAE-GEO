# Xunsearch retrieval backend

This optional backend indexes the existing clean and poisoned corpora with
Xunsearch, retrieves up to the requested top-k documents using BM25, and
connects the results to `search_geo_knowledge_base` and `scrape_geo_webpage`.
Search runs within the indexed corpus. It does not query a public Web search
engine or download webpages during an agent trajectory.

The implementation lives in this directory. `GEO_eval/geo_qwen_tools.py`
selects it when an environment entry sets `backend` to `xunsearch`.
Environments without that setting keep the existing lexical backend.

## Architecture

```text
Clean/L1/L2/L3 JSONL corpora
  -> title/body indexing in separate Xunsearch databases
  -> SCWS tokenization + BM25 -> ranked document IDs
  -> environment-scoped SQLite document store
  -> local HTTP adapter -> Search metadata + cached Scrape body
```

Only an opaque document ID, title, and body enter the search index. Hidden
provenance, source type, brand labels, and dataset metadata never influence
ranking. The document store retains them for raw tool traces and offline
evaluation. The agent's existing projection exposes only title/URL/timestamp
through Search, and page text through Scrape.

The schema gives titles weight 5 and uses Xunsearch BM25 with OR matching and
automatic synonyms disabled. These settings stay identical across levels.


## Docker quick start

Install Docker with Compose. The configuration targets Linux amd64; Docker
on Apple Silicon requires amd64 emulation. The image compiles the official
Xunsearch 1.4.17 source distribution with a pinned archive SHA256. The image
also contains PHP CLI and Python for the adapter.

From the repository root, start the engine:

```bash
docker compose -f deep_researchbehind/compose.yaml up -d --build xunsearch
```

Build four small demonstration indexes, using the existing demo corpora:

```bash
docker compose -f deep_researchbehind/compose.yaml run --rm gateway \
  python3 manage.py \
  --config /workspace/configs/environments.example.json \
  --root /workspace --output-dir /data/index

docker compose -f deep_researchbehind/compose.yaml up -d gateway
```

For corpora built by `collected_data/run_pipeline.sh`, replace the indexing
command above with:

```bash
docker compose -f deep_researchbehind/compose.yaml run --rm gateway \
  python3 manage.py \
  --config /workspace/collected_data/output/environments/environments.json \
  --corpus-dir /workspace/collected_data/output/environments \
  --root /workspace --output-dir /data/index
```

`--corpus-dir` explicitly remaps the corpus filenames into the container.
Each corpus must contain nonempty full-page bodies and unique tool URLs.
Use the collection builder to preserve distinct poisoned snapshots that
share an original URL. Metadata/snippets alone are not a full-page corpus.

The indexer writes a document store, schema, index inputs, source hashes,
and revision information under `deep_researchbehind/output/index/`. It waits
until all database document counts match before publishing `manifest.json`
and `environments.json`. A failed build does not publish a ready manifest.
An existing ready output directory is protected from accidental rebuilding;
use a new directory, then restart the adapter with the new manifest. The
Compose default adapter uses `/data/index/manifest.json`.

The generated index paths refer to the build environment. Use the manifest
in that same container/mount layout; the host-side agent only needs the
generated HTTP runtime configuration below.

## Use the existing agent tools

From the repository root, in the Python environment with the repository's
requirements installed:

```bash
export GEO_ENVIRONMENTS_CONFIG="$PWD/deep_researchbehind/output/index/environments.json"
export GEO_TOOL_CACHE_DIR="$PWD/.cache/xunsearch-tools"

python3 - <<'PY'
import json
from GEO_eval.geo_qwen_tools import search_geo_knowledge_base, scrape_geo_webpage

result = json.loads(search_geo_knowledge_base("laundry detergent", knowledge_name="L1", top_k=10))
print(json.dumps(result, ensure_ascii=False, indent=2))
if result["documents"]:
    print(scrape_geo_webpage(result["documents"][0]["url"], knowledge_name="L1"))
PY
```

The demo has only a few documents, so it may return fewer than ten matches.
The same environment variable works with the existing evaluation runners.
Keep query sets, prompts, corpora, index settings, and interaction budgets
fixed across comparisons. Rebuilding an index creates a revision-specific
cache namespace, so a different corpus or endpoint cannot reuse old pages.
The adapter fails explicitly on backend errors or revision mismatches; it
does not silently fall back to the lexical score.

## Existing Xunsearch installation

With PHP CLI and an existing Xunsearch service, Docker is optional:

```bash
python3 deep_researchbehind/manage.py \
  --config configs/environments.example.json \
  --output-dir deep_researchbehind/output/native-index \
  --index-server 127.0.0.1:8383 --search-server 127.0.0.1:8384

python3 deep_researchbehind/server.py \
  --manifest deep_researchbehind/output/native-index/manifest.json
```

Set `GEO_ENVIRONMENTS_CONFIG` to that directory's `environments.json`.
The vendored PHP SDK and its notices are described in [UPSTREAM.md](UPSTREAM.md).
Keep the gateway on loopback or a trusted private network: its internal HTTP
response carries full stored documents for the tool cache. The SDK engine
ports stay inside the Docker network; Compose publishes only the gateway on
`127.0.0.1:8765`.

## API and validation

`GET /health` reports the adapter revision and available environments.
`POST /search` accepts `environment`, `query`, `top_k`, `dataset`, and
`index_revision`; it returns ranked stored documents and scores. The
`index_revision` must match the generated configuration. Dataset filtering
uses the existing dataset/all convention after ranking and continues through
ranked pages until it finds top-k accepted documents or exhausts the results.

```bash
python3 -m unittest discover -s deep_researchbehind/tests -p 'test_*.py'
python3 deep_researchbehind/smoke_test.py \
  --config deep_researchbehind/output/index/environments.json
```

The unit/integration tests use an injected ranking fixture for the PHP boundary;
they validate indexing inputs, API behavior, isolation, and Search-to-Scrape
contracts without claiming to execute Xunsearch. 
