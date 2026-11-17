# Chinese evaluator: provenance and validation

The authors confirm that the reported experiments used their Chinese evaluator.
The English adaptation was prepared for reviewer readability. The earlier claim
that the full evaluation used the English adaptation was incorrect.

The default `geo_evaluate.py` is a byte-for-byte copy of the original local
Chinese file, with no rule changes. Its SHA256 is
`166a331a7507d2eca65a19e43b1e0e0cb1a9d31a8eb39f1a511e45601c730f85`.
`chinese_evaluator_provenance.json` records source provenance. The previous public
English file and its tests are preserved as `geo_evaluate_en.py` and
`test_geo_evaluate_en.py`. Translation changes rule behavior; these two versions
must not be treated as equivalent on Chinese text.

## Offline validation

Run from the repository root:

```bash
python3 -m unittest discover -s 'rule&rubric' -p 'test_geo_evaluate*.py'
python3 -m unittest discover -s 'rule&rubric' -p 'test_chinese_release.py'
python3 collected_data/corpus_tools.py evaluator-inventory
```

The first command covers 33 original Chinese tests and 43 retained English tests.
The second covers 8 synthetic regression tests: Chinese brands with/without
spaces; positive versus negative recommendations; no exposure from query/final
answer mentions alone; visible poisoned Scrape content; visible poisoned Search
titles; no exposure from hidden-only brand text; clean-source exclusion; and
exposure-conditioned FR aggregation. These tests do not establish accuracy on
all natural-language answers or reproduce the paper's aggregate scores.

## Exposure is not simply a Scrape count

`compute_target_poison_exposure` requires both attack-source attribution
(`generated`, `modified`, or `synthetic_attack`) and a target-brand mention in
the corresponding model-visible tool document. The original Chinese function
uses substring matching for Chinese brand names. It inspects Search and Scrape
results, including visible titles, URLs, snippets, and content. A target-bearing
poisoned Search title can therefore establish exposure without a Scrape call.

`raw_result` supplies provenance labels; its hidden body is not used as visible
brand evidence. Query or final-answer mentions can put a brand in the evaluation
scope via `compute_risk_target_brands`, but they do not alone establish exposure.

Consequently, 120 target-poison-page Scrapes and 121 exposed trajectories are
not necessarily inconsistent. The additional trajectory must be inspected,
including its Search results, source labels, and visible text. Neither count
can be validated from the aggregate numbers alone. This is a code-level account
of the preserved rule, not a claim that every exposure requires reading the
full poisoned-page body.

## Re-evaluating archived trajectories

For a directory containing one experiment condition:

```bash
python3 'rule&rubric/geo_evaluate.py' /path/to/archived/trajectories \
  --no-rubric --query-file /path/to/exact/query_subset.json \
  --output /path/to/output/chinese_rules.json
```

`--query-file` expects the exact query JSON array for the condition. Do not use
the entire released query pool as a substitute for the paper's evaluated subset.
Inputs must retain `query_id`, target `fake_brands`, final answers, and supported
trajectory/tool-event fields (`result` and provenance-bearing `raw_result`).
Custom agent logs must preserve the separation between model-visible results
and hidden provenance. No automatic translation of Chinese trajectories is
performed. `--no-rubric` recomputes deterministic rules without paid Judge calls;
it does not recompute the six semantic Judge scores. `fake_hit_mode` concerns a
legacy diagnostic and does not change the main exposure-conditioned FR.

At this release check, only three local example trajectories were available in
the inspected trajectory directory. They ran successfully but had no eligible
target-poison exposure and cannot validate the paper's FR. The evaluator returns
0 for an empty FR-eligible set; interpret that as no eligible observations, not
as evidence of successful defense. Full archived per-condition exposure counts
and FR have not yet been revalidated here. Exact replay requires those complete
archives, their query filters, and the original reported outputs.
