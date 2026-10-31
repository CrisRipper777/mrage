# MRAGE LP recommended starting configuration

Use mrage_lp_default_h256.yaml for both sports-copurchase and cloth-copurchase.
This preserves the current original MRAGE model and shared LP task protocol; it
is a conservative recommendation, not an optimum measured on these LP datasets.
No LP tuning or training was run when preparing this recommendation.
Training seeds42/43/44 reuse each dataset's single official lp-edge-split.pt.
Use scripts/run_paper_benchmark.py --models mrage --task lp --stage all.
The ten-model run_lp_baseline_benchmark.py wrapper excludes MRAGE.
