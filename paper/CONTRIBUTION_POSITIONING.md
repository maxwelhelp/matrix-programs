# Contribution positioning audit

This note records the intended novelty boundary for the paper.

## Claimed contributions

1. A unified executable, weight-derived representation for the modern RoPE/GQA/RMSNorm/bias/SwiGLU stack, integrating relative-distance-indexed affine QK targets, an affine VO map, and an exact SwiGLU atom program.
2. Faithful reinsertion of that representation into the native forward pass, including global replacement of all attention and MLP sublayers.
3. Full factorized QK/VO and MLP intervention atlases with downstream propagation profiles.
4. Effect-to-replacement-discrepancy calibration as an empirical reporting practice.
5. A verified extractor and the no-bias control establishing the practical necessity of affine bias folding for this model.

## Explicitly not claimed as novel

- The general QK/OV circuit decomposition (Elhage et al., 2021).
- RoPE and its relative-position identity (Su et al., 2021).
- Homogeneous coordinates as a mathematical device.
- The algebraic expansion of SwiGLU itself (Shazeer, 2020).
- Input-specific locally linear/Jacobian representations (Golden, 2025).
- Trained sparse component surrogates such as transcoders (Dunefsky et al., 2024).
- Activation patching, causal scrubbing, or causal abstraction in general.

## Exploratory observation

The gate lens is presented only as a qualitative probe motivated by the gate factor in SwiGLU. It is not listed as a validated main contribution until a quantitative benchmark is performed.
