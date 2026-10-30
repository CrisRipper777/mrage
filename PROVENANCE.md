# Provenance

- Source repository: <https://github.com/CrisRipper777/model_research>
- Source branch: `exp/mrcge_final_benchmark_nc5_lp2`
- Source commit used for this migration: `8f1f9c490907583b003beb73b72e873d62f16c3d`
- Previous requested source commit: `72c18fd80a65b671ee3664dbcb810dfc6896786c`

The source commit was updated by the project owner during this task; the
explicit instruction to use `8f1f9c4` superseded the initial pinned commit.
The migrated model is based on the frozen MR-CGE V1.3-A Final implementation.
The work here is an engineering migration, removal of unrelated historical
research code, and automatic semantic graph initialization. The formal model
was checked against the selected source commit with paired CPU runs.

The current MRAGE repository HEAD is tracked separately from that source
snapshot. NC uses the fixed MAGB split identity for split seed 42 while model
training seeds remain independent. The release keeps hidden dimension 256 as
the default, permits other positive hidden dimensions, and preserves the
four-expert/Top-2 routing contract. Formal runs use a two-process Validation
and Final Test workflow; see the README for the runner and result collector.

Only runtime-required data/model/task/configuration files and the ten selected
baselines were extracted. Historical model variants, generated caches,
checkpoints, experiment results, and research logs were not migrated. The
source repository was treated as read-only; no remote or push was configured.

Common graph operators used by MRAGE were extracted into `src/models/ops/`.
The formal MRAGE import closure does not import `cg_cge_v1.py`,
`psce_mag_v7a.py`, `rmcge_v2.py`, or `mvcge_mag_v21.py`. Those historical model
files are not part of this project.

The standard V7B.1 candidate builder does not include degree-preserving random
graph construction. Ablation and Analysis pipelines are still future work and
were not implemented as part of this release.
