# Aggregation conventions

- Per-experiment prompt summaries in `all_experiment_summaries.csv` were generated with `torch.median`. For even N, this is the lower of the two central sorted values. This convention produces the headline global-backbone value 0.258383%, reported as 0.258%.
- Medians reported across the 336 head-atlas rows and 24 MLP-atlas rows use the conventional sample median, averaging the two central values for even N. This produces 20.62x for VO and 152.05x for MLP effect-to-replacement-discrepancy ratios.
- No numeric experiment values were changed during release cleanup; only legacy column names containing `noise` were renamed.
