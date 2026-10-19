# Paper ablations

| Config | Paper interpretation | Avg. mIoU, seeds 0 / 1 / 2 (%) |
|---|---|---|
| `full` | RoleHOS (SFSC + RICD) | 73.0333 / 72.1076 / 72.4734 |
| `without_both_zero` | Anonymous only; no SFSC and zero global context | 52.2857 / 52.9182 / 52.8537 |
| `without_ricd` | Anonymous + SFSC (Hungarian queries) | 67.7607 / 69.1056 / 68.0867 |
| `without_sfsc_zero` | Without SFSC; indexed tokens, zero global context | 70.0859 / 68.8458 / 69.1541 |
| `without_foreground_conditions` | Spatial 0 / global 0, foreground auxiliary loss retained | 71.3365 / 71.4850 / 70.3009 |
| `p_only_zero` | Spatial P / global 0 | 72.3448 / 71.7999 / 70.2288 |
| `dense_only` | Spatial P,U,R / global 0 | 71.1268 / 71.2697 / 71.7892 |
| `global_only` | Spatial 0 / foreground-weighted pooling | 72.4242 / 71.6668 / 72.3399 |
| `p_only_weighted` | Spatial P / foreground-weighted pooling | 72.0343 / 71.9620 / 71.7910 |
| `token_ffn` | Token-wise FFN instead of token attention | 73.0735 / 72.1543 / 71.1187 |
| `Q1_fixed5_noagg` | Unconditioned indexed | 71.2226 / 71.8441 / 71.4767 |
| `Q0_hungarian5_noagg` | Unconditioned anonymous | 54.4459 / 54.4070 / 53.9284 |

These are the final manuscript's 12 configurations, 36 final-20k checkpoints. Superseded `without_sfsc` / `without_both` variants with ordinary global pooling are excluded; the final table uses their zero-context controls. Full, main removal, foreground interface and query-indexing ablations retain their exact original model configuration and separate seed identity. This release does not bundle five unrelated baseline implementations or device-specific TensorRT engines.

The released values are archived original evaluations, not a claim that 36 full trainings were repeated during packaging. A batch-1 CPU check can differ slightly from historical GPU/batched arithmetic near a decision threshold. `docs/expected_metrics.csv` and the matching logs are the reference, not an automatic tolerance-free equality claim across hardware.
