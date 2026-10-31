# Validation search progress

Generated from completed Validation runs. Each cell reports Accuracy percent and the number of available training seeds. Test is excluded.

| Candidate | Movies | Toys | Grocery | ele-fashion | Reddit-S | Representative mean |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 55.549 (n=3) | 79.971 (n=3) | 83.641 (n=3) | 87.327 (n=3) | 96.162 (n=3) | 75.506 |
| lr3e4 | 55.609 (n=1) | 79.971 (n=1) | 83.631 (n=1) | 86.847 (n=1) | pending | 75.362 |
| lr6e4 | 55.489 (n=1) | pending | 83.485 (n=1) | 87.583 (n=1) | pending | 75.519 |
| a01_weight_decay_1em05 | 55.849 (n=1) | pending | 83.631 (n=1) | 87.307 (n=1) | pending | 75.596 |
| a01_weight_decay_0p001 | 55.879 (n=1) | pending | 83.455 (n=1) | 87.286 (n=1) | pending | 75.540 |
| a02_dropout_0p1 | 55.429 (n=1) | pending | 83.280 (n=1) | 87.481 (n=1) | pending | 75.396 |
| a02_dropout_0p3 | 55.939 (n=1) | pending | 83.543 (n=1) | 87.379 (n=1) | pending | 75.620 |
| a03_gamma_init_0p04 | 55.609 (n=1) | pending | 83.602 (n=1) | 87.399 (n=1) | pending | 75.537 |
| a03_gamma_init_0p16 | 56.359 (n=1) | pending | 83.338 (n=1) | 87.327 (n=1) | pending | 75.675 |
| a03_gamma_init_0p25 | 56.269 (n=1) | pending | 83.280 (n=1) | 87.450 (n=1) | pending | 75.666 |
| a04_beta_init_0p05 | 56.209 (n=1) | pending | 83.280 (n=1) | 87.726 (n=1) | pending | 75.738 |
| a04_beta_init_0p15 | 55.789 (n=1) | pending | 83.104 (n=1) | 87.512 (n=1) | pending | 75.468 |
| a04_beta_init_0p4 | 55.789 (n=1) | pending | 83.543 (n=1) | 87.348 (n=1) | pending | 75.560 |
| a05_rho_init_0p15 | 55.399 (n=1) | pending | 83.221 (n=1) | 87.522 (n=1) | pending | 75.381 |
| a05_rho_init_0p4 | 56.209 (n=1) | pending | 83.367 (n=1) | 87.389 (n=1) | pending | 75.655 |
| a06_router_hidden_dim_128 | 55.969 (n=1) | pending | 83.367 (n=1) | 87.583 (n=1) | pending | 75.640 |
| a06_router_hidden_dim_512 | 55.519 (n=1) | pending | 83.045 (n=1) | 87.491 (n=1) | pending | 75.352 |
| a07_attention_dim_16 | 55.489 (n=1) | pending | 83.221 (n=1) | 87.573 (n=1) | pending | 75.428 |
| a07_attention_dim_64 | 55.759 (n=1) | pending | 83.133 (n=1) | 87.665 (n=1) | pending | 75.519 |
| a08_router_embed_dim_4 | 55.759 (n=1) | pending | 83.075 (n=1) | 87.512 (n=1) | pending | 75.448 |
| a08_router_embed_dim_16 | 55.939 (n=1) | pending | 83.514 (n=1) | 87.471 (n=1) | pending | 75.641 |
| a09_channel_gate_hidden_64 | 55.699 (n=1) | pending | 83.280 (n=1) | 87.798 (n=1) | pending | 75.592 |
| a09_channel_gate_hidden_256 | 55.909 (n=1) | pending | 83.221 (n=1) | 87.542 (n=1) | pending | 75.557 |
| a10_nc_unimodal_weight_0p1 | 55.849 (n=1) | pending | 83.426 (n=1) | 87.276 (n=1) | pending | 75.517 |
| a10_nc_unimodal_weight_0p5 | 55.699 (n=1) | pending | 83.485 (n=1) | 87.583 (n=1) | pending | 75.589 |
| a11_balance_weight_0p001 | 55.699 (n=1) | pending | 83.397 (n=1) | 87.563 (n=1) | pending | 75.553 |
| a11_balance_weight_0p03 | 55.369 (n=1) | pending | 82.928 (n=1) | 87.552 (n=1) | pending | 75.283 |
| a12_expert_bottleneck_32 | 55.879 (n=1) | pending | 83.455 (n=1) | 87.552 (n=1) | pending | 75.629 |
| a12_expert_bottleneck_128 | 55.879 (n=1) | pending | 83.455 (n=1) | 87.655 (n=1) | pending | 75.663 |
| a13_expert_eta_0p05 | 56.059 (n=1) | pending | 83.309 (n=1) | 87.501 (n=1) | pending | 75.623 |
| a13_expert_eta_0p2 | 56.009 (n=3) | 80.060 (n=3) | 83.260 (n=3) | 87.450 (n=3) | 96.099 (n=3) | 75.573 |
| a14_hidden_dim_128 | 55.979 (n=3) | 80.430 (n=3) | 83.446 (n=3) | 87.539 (n=3) | 96.341 (n=3) | 75.654 |
| a14_hidden_dim_512 | 55.009 (n=1) | pending | 82.753 (n=1) | 87.685 (n=1) | pending | 75.149 |
