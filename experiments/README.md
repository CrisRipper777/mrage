# Planned paper extensions

The production encoder in `src/models/mrage.py` contains only the frozen full
model. Ablations and instrumented analyses are intentionally separate from
that API. No extension is advertised as implemented until it has its own
reproducible entry point and parity/behavior checks.

Planned ablations: Full MRAGE, Single Filter, Unshared Experts, View-global
Routing, without Semantic Propagation, without Topological Collaboration, and
without Semantic Refinement.

Planned analyses: relational neighborhood/response differences; shared expert
filter coefficients, routing and responses; Private/Collaborative routing and
Semantic/Same-node/None selection; optional topology perturbation robustness.

No ablation or analysis training has been launched as part of the core project
migration.
