# Result files

These files are the machine-readable outputs used by the paper tables and figures.

Primary quantitative artifacts:

- `all_experiment_summaries.csv`
- `all_prompt_metrics.csv`
- `head_causal_atlas.csv`
- `mlp_causal_atlas.csv`
- `layer_group_atlas.csv`
- `propagation_profiles.csv`
- `experiment_manifest.json`
- `final_prompts.json`
- `summary.json`

The `atom_*` files are retained for transparency but are exploratory. In the original archived run, the three random controls were duplicated because a random seed was reused. The corrected v4.1 experiment driver is included in `experiments/`, and a rerun-only wrapper is included in `exploratory/atom_split_fixed_seeds.py`.

## Aggregation and terminology

Per-experiment summaries over prompts use `torch.median`, which returns the lower middle value for an even number of observations. Medians computed across rows of the released head and MLP atlas CSVs use the conventional median. Legacy internal column names containing `noise` have been renamed to `replacement_discrepancy`; numeric values are unchanged.
