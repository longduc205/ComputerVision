"""Result analysis: per-class breakdown, t-SNE, PD heatmaps."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False


def analyze_per_class_metrics(
    metrics_per_class: Dict[int, float],
    class_names: Optional[Dict[int, str]],
    output_dir: str | Path,
    prefix: str = "per_class",
) -> pd.DataFrame:
    """Generate per-class performance table and bar chart.

    Args:
        metrics_per_class: Dict mapping class_id -> mAP.
        class_names: Dict mapping class_id -> class name string.
        output_dir: Directory to save outputs.
        prefix: Filename prefix.

    Returns:
        DataFrame with class metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if class_names is None:
        class_names = {k: f"class_{k}" for k in metrics_per_class}

    rows = []
    for class_id, ap in metrics_per_class.items():
        rows.append({
            "class_id": class_id,
            "class_name": class_names.get(class_id, f"class_{class_id}"),
            "AP": ap,
        })

    df = pd.DataFrame(rows).sort_values("AP", ascending=False)

    if HAS_PLOTTING:
        fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.4)))
        sns.barplot(data=df, y="class_name", x="AP", ax=ax, palette="viridis")
        ax.set_xlabel("Average Precision")
        ax.set_title("Per-Class AP Performance")
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}_ap_bar.png", dpi=150)
        plt.close()

    df.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    return df


def analyze_domain_performance(
    results_by_domain: Dict[str, Dict[str, float]],
    output_dir: str | Path,
) -> pd.DataFrame:
    """Compare performance across domains. Generate grouped bar chart."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for domain, metrics in results_by_domain.items():
        for metric_name, value in metrics.items():
            rows.append({"domain": domain, "metric": metric_name, "value": value})

    df = pd.DataFrame(rows)

    if HAS_PLOTTING and not df.empty:
        metric_count = len(df["metric"].unique())
        fig, axes = plt.subplots(1, metric_count, figsize=(6 * metric_count, 5))
        if metric_count == 1:
            axes = [axes]
        for ax, metric in zip(axes, df["metric"].unique()):
            subset = df[df["metric"] == metric]
            sns.barplot(data=subset, x="domain", y="value", ax=ax, palette="Set2")
            ax.set_title(f"{metric} by Domain")
            ax.set_ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / "domain_comparison.png", dpi=150)
        plt.close()

    return df


def plot_tsne_features(
    features: np.ndarray,
    labels: np.ndarray,
    domains: np.ndarray,
    output_dir: str | Path,
    filename: str = "tsne_features.png",
) -> None:
    """Generate t-SNE visualization of detector features colored by class and domain."""
    if not HAS_TSNE or not HAS_PLOTTING:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    perplexity = min(30, len(features) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    features_2d = tsne.fit_transform(features)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab20", alpha=0.6, s=10)
    axes[0].set_title("t-SNE: Features Colored by Class")
    plt.colorbar(scatter, ax=axes[0], label="Class ID")

    scatter2 = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], c=domains, cmap="Set2", alpha=0.6, s=10)
    axes[1].set_title("t-SNE: Features Colored by Domain")
    plt.colorbar(scatter2, ax=axes[1], label="Domain ID")

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()


def save_results_summary(
    all_results: Dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Save evaluation results to JSON with summary table."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    if "metrics" in all_results:
        df = pd.DataFrame([all_results["metrics"]], index=["value"]).T
        df.to_csv(output_dir / "metrics_summary.csv")
