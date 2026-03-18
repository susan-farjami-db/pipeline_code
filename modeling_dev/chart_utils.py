"""Chart utilities for evaluation and monitoring notebooks."""

import numpy as np


def create_horizon_ape_chart(q_mapes: list, partition: str, model_type: str, horizon_range: tuple):
    """Line chart of APE per forecast horizon for a single model. Returns matplotlib figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    horizons = list(range(horizon_range[0], horizon_range[1] + 1))
    fig, ax = plt.subplots(figsize=(8, 5))

    valid_horizons, valid_mapes = _filter_plottable(horizons, q_mapes)

    if valid_horizons:
        ax.plot(valid_horizons, valid_mapes, marker='o', linewidth=2, markersize=8, color='#1f77b4')
        ax.fill_between(valid_horizons, valid_mapes, alpha=0.2, color='#1f77b4')

    ax.set_xlabel('Forecast Horizon (Q+)', fontsize=11)
    ax.set_ylabel('MAPE (%)', fontsize=11)
    partition_display = partition[:35] + '...' if len(partition) > 35 else partition
    ax.set_title(f'{partition_display}\n{model_type}', fontsize=12)
    ax.set_xticks(horizons)
    ax.set_xticklabels([f'Q+{h}' for h in horizons])
    ax.grid(True, alpha=0.3)

    for h, m in zip(valid_horizons, valid_mapes):
        ax.annotate(f'{m:.1f}%', (h, m), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)

    plt.tight_layout()
    return fig


def create_champion_summary_chart(champions_df, horizon_range: tuple):
    """Line chart with one line per partition's champion APE over horizons. Returns matplotlib figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    horizons = list(range(horizon_range[0], horizon_range[1] + 1))
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab20.colors

    for idx, (_, row) in enumerate(champions_df.iterrows()):
        q_mapes = [row.get(f'Q{h}_MAPE') for h in horizons]
        valid_horizons, valid_mapes = _filter_plottable(horizons, q_mapes)

        if not valid_horizons:
            continue

        label = row['PARTITION_COLUMN']
        label = label[:30] + '...' if len(label) > 30 else label
        color = colors[idx % len(colors)]
        ax.plot(valid_horizons, valid_mapes, marker='o', label=label, alpha=0.7, color=color, linewidth=1.5)

    ax.set_xlabel('Forecast Horizon (Q+)', fontsize=12)
    ax.set_ylabel('MAPE (%)', fontsize=12)
    ax.set_title(f'Champion Models: MAPE by Forecast Horizon (Q+{horizon_range[0]} to Q+{horizon_range[1]})', fontsize=14)
    ax.set_xticks(horizons)
    ax.set_xticklabels([f'Q+{h}' for h in horizons])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8, ncol=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def _filter_plottable(horizons, mapes):
    """Filter out None/NaN values from horizon-MAPE pairs for plotting."""
    valid_h, valid_m = [], []
    for h, m in zip(horizons, mapes):
        if m is not None and not (isinstance(m, float) and np.isnan(m)):
            valid_h.append(h)
            valid_m.append(m)
    return valid_h, valid_m
