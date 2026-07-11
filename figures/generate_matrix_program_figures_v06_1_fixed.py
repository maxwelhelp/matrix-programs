from pathlib import Path
import argparse
import json
import re

import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


def save_both(fig, out_base: Path) -> None:
    fig.savefig(out_base.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_base.with_suffix('.png'), dpi=220, bbox_inches='tight')
    plt.close(fig)


def load_tables(results_dir: Path) -> dict[str, pd.DataFrame]:
    required = {
        'summaries': 'all_experiment_summaries.csv',
        'prompts': 'all_prompt_metrics.csv',
        'heads': 'head_causal_atlas.csv',
        'mlps': 'mlp_causal_atlas.csv',
        'layers': 'layer_group_atlas.csv',
        'prop': 'propagation_profiles.csv',
    }
    tables: dict[str, pd.DataFrame] = {}
    for key, filename in required.items():
        path = results_dir / filename
        if not path.exists():
            raise FileNotFoundError(f'Missing required file: {path}')
        tables[key] = pd.read_csv(path)
    return tables


def fig1_schematic(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    boxes = [
        (0.02, 0.20, 0.18, 0.60, 'Native component\nattention / MLP'),
        (
            0.26,
            0.14,
            0.20,
            0.72,
            'Weight-derived\nprogram\n\nAttention:\n'
            '$M_{qk}^{aug}[d], C_{vo}^{aug}$\n\n'
            'MLP:\natoms $B_j$, coeffs $a_j(x)$',
        ),
        (
            0.53,
            0.18,
            0.17,
            0.64,
            'Executable\nreplacement\n\nsubtract native\ninsert program',
        ),
        (
            0.76,
            0.10,
            0.20,
            0.80,
            'Factorized interventions\n\nQK off\nVO off\nMLP off\n'
            'held-out atom test\npropagation atlas',
        ),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle='round,pad=0.02',
            linewidth=1.5,
            facecolor='white',
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=12)

    for start, end in [
        ((0.20, 0.50), (0.26, 0.50)),
        ((0.46, 0.50), (0.53, 0.50)),
        ((0.70, 0.50), (0.76, 0.50)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle='-|>',
                mutation_scale=18,
                linewidth=1.5,
            )
        )

    save_both(fig, out_dir / 'fig1_method_overview')


def fig2_heatmaps(heads: pd.DataFrame, out_dir: Path) -> None:
    n_layers = int(heads['layer'].max()) + 1
    n_heads = int(heads['head'].max()) + 1
    replacement = np.full((n_layers, n_heads), np.nan)
    qk = np.full((n_layers, n_heads), np.nan)

    for _, row in heads.iterrows():
        layer = int(row['layer'])
        head = int(row['head'])
        replacement[layer, head] = row['program_logit_rel_percent']
        qk[layer, head] = row['qk_off_logit_rel_percent']

    fig, axes = plt.subplots(1, 2, figsize=(12, 7), constrained_layout=True)

    im0 = axes[0].imshow(replacement, aspect='auto')
    axes[0].set_title('(a) Replacement discrepancy, %')
    axes[0].set_xlabel('Head')
    axes[0].set_ylabel('Layer')
    axes[0].set_xticks(range(n_heads))
    axes[0].set_yticks(range(n_layers))
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    c0.set_label('%')

    im1 = axes[1].imshow(qk, aspect='auto')
    axes[1].set_title('(b) QK-off effect, %')
    axes[1].set_xlabel('Head')
    axes[1].set_ylabel('Layer')
    axes[1].set_xticks(range(n_heads))
    axes[1].set_yticks(range(n_layers))
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    c1.set_label('%')

    save_both(fig, out_dir / 'fig2_head_heatmaps')


def _extract_relative_propagation(prop: pd.DataFrame) -> pd.DataFrame:
    pattern = re.compile(r'^head_L(\d+)H(\d+)_(qk|vo)_off$')
    records: list[dict[str, float | int | str]] = []

    residual = prop[prop['target_type'].eq('residual')]
    for _, row in residual.iterrows():
        match = pattern.match(str(row['experiment']))
        if match is None:
            continue
        source_layer = int(match.group(1))
        source_head = int(match.group(2))
        kind = match.group(3)
        # target_index = source_layer + 1 is the first post-source residual.
        relative_offset = int(row['target_index']) - (source_layer + 1)
        if relative_offset < 0:
            continue
        records.append(
            {
                'experiment': row['experiment'],
                'source_layer': source_layer,
                'source_head': source_head,
                'kind': kind,
                'relative_offset': relative_offset,
                'valid_tensor_rel_percent': float(row['valid_tensor_rel_percent']),
            }
        )
    return pd.DataFrame.from_records(records)


def fig3_propagation(prop: pd.DataFrame, heads: pd.DataFrame, out_dir: Path) -> None:
    relative = _extract_relative_propagation(prop)
    if relative.empty:
        raise ValueError('No per-head residual propagation rows found')

    examples = {
        'QK L2H5': 'head_L02H05_qk_off',
        'QK L17H2': 'head_L17H02_qk_off',
        'VO L16H10': 'head_L16H10_vo_off',
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)

    # Panel A uses exactly the same metric as the textual examples:
    # valid_tensor_rel_percent = relative L2 perturbation over all valid
    # residual tokens in the 64-prompt batch.
    for label, experiment in examples.items():
        sub = relative[relative['experiment'].eq(experiment)].sort_values('relative_offset')
        if sub.empty:
            raise ValueError(f'Missing propagation experiment: {experiment}')
        axes[0].plot(
            sub['relative_offset'],
            sub['valid_tensor_rel_percent'],
            marker='o',
            markersize=3.5,
            linewidth=2,
            label=label,
        )

    axes[0].set_title('(a) Selected propagation trajectories')
    axes[0].set_xlabel('Downstream offset from first post-source residual, layers')
    axes[0].set_ylabel('Valid-tensor relative perturbation, %')
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    # Panel B directly visualizes the statistic quoted in the paper:
    # per-head delay to the maximum valid-tensor residual perturbation.
    qk_delay = heads['qk_delayed_peak_layers'].dropna().to_numpy(dtype=float)
    vo_delay = heads['vo_delayed_peak_layers'].dropna().to_numpy(dtype=float)
    bins = np.arange(-0.5, max(qk_delay.max(), vo_delay.max()) + 1.5, 1.0)
    axes[1].hist(qk_delay, bins=bins, alpha=0.62, label=f'QK (median={np.median(qk_delay):.0f})')
    axes[1].hist(vo_delay, bins=bins, alpha=0.62, label=f'VO (median={np.median(vo_delay):.0f})')
    axes[1].axvline(np.median(qk_delay), linestyle='--', linewidth=1.5)
    axes[1].axvline(np.median(vo_delay), linestyle=':', linewidth=1.8)
    axes[1].set_title('(b) Distribution of per-head peak delay')
    axes[1].set_xlabel('Layers from first post-source residual to peak')
    axes[1].set_ylabel('Number of heads')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend()

    save_both(fig, out_dir / 'fig3_propagation_profiles')


def fig4_prompt_suites(prompts: pd.DataFrame, out_dir: Path) -> None:
    sub = prompts[prompts['experiment'].eq('global_backbone_program')].copy()
    if sub.empty:
        raise ValueError('No global_backbone_program rows found')

    agg = (
        sub.groupby('suite')
        .agg(
            median=('logit_rel_percent', 'median'),
            max=('logit_rel_percent', 'max'),
            n=('prompt_id', 'count'),
        )
        .reset_index()
        .sort_values('median', ascending=False)
    )

    x = np.arange(len(agg))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, agg['median'], width=width, label='Median')
    ax.bar(x + width / 2, agg['max'], width=width, label='Maximum')
    ax.set_xticks(x)
    ax.set_xticklabels(agg['suite'], rotation=30, ha='right')
    ax.set_ylabel('Relative logit error, %')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    save_both(fig, out_dir / 'fig4_global_replacement_by_suite')


def write_metric_audit(data: dict[str, pd.DataFrame], out_dir: Path) -> None:
    prop = data['prop']
    heads = data['heads']
    checks: dict[str, dict[str, float]] = {}

    for layer, head, kind in [(2, 5, 'qk'), (17, 2, 'qk'), (16, 10, 'vo')]:
        experiment = f'head_L{layer:02d}H{head:02d}_{kind}_off'
        sub = prop[
            prop['experiment'].eq(experiment) & prop['target_type'].eq('residual')
        ].sort_values('target_index')
        atlas = heads[heads['layer'].eq(layer) & heads['head'].eq(head)]
        if sub.empty or atlas.empty:
            raise ValueError(f'Missing data for {experiment}')
        peak = sub.loc[sub['valid_tensor_rel_percent'].idxmax()]
        row = atlas.iloc[0]
        checks[experiment] = {
            'csv_peak_percent': float(peak['valid_tensor_rel_percent']),
            'csv_peak_index': int(peak['target_index']),
            'atlas_peak_percent': float(row[f'{kind}_residual_peak_percent']),
            'atlas_peak_index': int(row[f'{kind}_residual_peak_index']),
            'atlas_first_percent': float(row[f'{kind}_residual_first_percent']),
            'atlas_final_logit_percent': float(row[f'{kind}_off_logit_rel_percent']),
            'atlas_internal_to_final_ratio': float(
                row[f'{kind}_internal_peak_to_final_logit_ratio']
            ),
            'atlas_delay_layers': int(row[f'{kind}_delayed_peak_layers']),
        }
        if not np.isclose(
            checks[experiment]['csv_peak_percent'],
            checks[experiment]['atlas_peak_percent'],
            rtol=1e-10,
            atol=1e-10,
        ):
            raise AssertionError(f'Peak percent mismatch for {experiment}')
        if checks[experiment]['csv_peak_index'] != checks[experiment]['atlas_peak_index']:
            raise AssertionError(f'Peak index mismatch for {experiment}')

    audit = {
        'figure_3_metric': (
            'valid_tensor_rel_percent: relative L2 perturbation over all valid '
            'residual tokens in the 64-prompt batch'
        ),
        'relative_offset_definition': 'target_index - (source_layer + 1)',
        'qk_median_peak_delay_layers': float(heads['qk_delayed_peak_layers'].median()),
        'vo_median_peak_delay_layers': float(heads['vo_delayed_peak_layers'].median()),
        'examples': checks,
    }
    (out_dir / 'figure_metric_audit.json').write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results-dir',
        required=True,
        help='Directory containing final_article CSV/JSON files',
    )
    parser.add_argument('--out-dir', required=True, help='Where to save PDF/PNG figures')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_tables(results_dir)
    fig1_schematic(out_dir)
    fig2_heatmaps(data['heads'], out_dir)
    fig3_propagation(data['prop'], data['heads'], out_dir)
    fig4_prompt_suites(data['prompts'], out_dir)
    write_metric_audit(data, out_dir)

    manifest = {
        'results_dir': str(results_dir),
        'figure_3_metric': 'valid_tensor_rel_percent',
        'figure_3_alignment': 'target_index - (source_layer + 1)',
        'figures': [
            'fig1_method_overview.pdf/png',
            'fig2_head_heatmaps.pdf/png',
            'fig3_propagation_profiles.pdf/png',
            'fig4_global_replacement_by_suite.pdf/png',
        ],
    }
    (out_dir / 'figure_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'Saved figures and metric audit to {out_dir}')


if __name__ == '__main__':
    main()
