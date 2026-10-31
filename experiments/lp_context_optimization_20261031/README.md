# MRAGE LP semantic-context preparation optimization

Only `src/models/mrage_lp.py` changes runtime behavior. MRAGE/task defaults,
the main LinkNeighborLoader, training, Validation and Test code stay unchanged.

Changes:

- Build the extra Train-graph NeighborSampler/CSC index once per graph binding.
  Keep the per-batch NeighborLoader wrapper and iterator RNG draw, preserving
  the original seeded sampling results and order.
- Sample context IDs/edges without gathering an unused second feature matrix.
- Gather only source IDs for the first semantic-closure query.
- Vectorize first-occurrence EID deduplication while retaining sampler order.
- Reuse one global-to-local node mapping within each batch.

No semantic candidates are truncated. Both modalities' weights/source flags,
the `[1,1,1]` extra physical fanouts, and supervision-positive edge exclusion
are retained. The cached sampler is CPU-only runtime state, absent from
`state_dict`, and rebuilt when the Train graph is rebound.

Validation:

- `PYTHONPATH=. python -m pytest -q tests`: **53 passed** in `yhf_env`.
- 11 new targeted cases cover seeds 42/43/44, fresh-vs-cached sampling,
  complete semantic closure, flags/weights/order, duplicate EIDs, isolated
  nodes, empty semantics, CPU RNG preservation and graph rebinding.
- `sports_cpu_context_comparison.json` records a CPU-only before/after check
  using the existing Sports semantic cache and three real sampled supervision
  batches (`batch_size=2048`, fanouts `[5,5,5]`, 8 Torch threads).
  There is one warmup per path; cloning and supervision sampling are excluded
  from the timing. Measurement order alternates between paths.
- Every expanded feature, node ID, EID, physical edge, label, semantic edge,
  weight, source flag and context metadata matched exactly for the three
  context seeds. Caller CPU RNG states were unchanged.
- Context-preparation medians: **139.3 ms before / 103.4 ms after**, approximately
  **1.35x** for this stage. This is a small CPU microbenchmark, not an end-to-end
  training speedup claim. Graph-binding times are recorded separately.
- The three expanded batches contain 50,233–50,241 of 50,250 Sports nodes.
  Their model compute is unchanged; no reduction of this closure was attempted.

No GPU training, full benchmark, or Test evaluation was started. Cloth has no
existing semantic cache here and was not benchmarked or used to build a cache.
Original configuration hashes and before/after source hashes are in the JSON.
