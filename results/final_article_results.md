# Final article result summary
> Aggregation convention: per-experiment prompt summaries use the released `torch.median` lower-middle value for even N; medians across atlas rows use the conventional median.
## Global executable replacement
| Experiment | N | logit rel median % | p95 % | max % | KL median | top-1 preserved |
|---|---:|---:|---:|---:|---:|---:|
| global_attention_program | 64 | 0.254006 | 0.476551 | 0.885164 | 3.37841e-05 | 100.00% |
| global_mlp_program | 64 | 0.240513 | 0.418546 | 0.883226 | 2.6832e-05 | 100.00% |
| global_backbone_program | 64 | 0.258383 | 0.56624 | 0.923447 | 3.09369e-05 | 100.00% |

## Complete attention-head atlas
- Heads: 336
- Single-head program replacement discrepancy, conventional median: 0.137008%
- QK-off effect, conventional median: 3.27985%
- VO-off effect, conventional median: 2.88301%
- QK-off / replacement-discrepancy ratio, conventional median: 24.89×
- VO-off / replacement-discrepancy ratio, conventional median: 20.62×
- QK delay to peak, conventional median: 10 layers
- VO delay to peak, conventional median: 9 layers

## Complete MLP atlas
- MLP layers: 24
- Program replacement discrepancy, conventional median: 0.138715%
- MLP-off effect, conventional median: 23.4362%
- MLP-off / replacement-discrepancy ratio, conventional median: 152×

## Notes
- The atom files are retained for transparency but remain exploratory because the original archived random controls reused a seed.
- The paper uses `replacement discrepancy`; it does not interpret the native identity control as an independent numerical lower-bound estimate.
