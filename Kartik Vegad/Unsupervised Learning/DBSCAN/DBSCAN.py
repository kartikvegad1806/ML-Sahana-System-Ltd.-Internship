import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Optional, Tuple, List

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

warnings.filterwarnings('ignore')


# --- Global Configuration -----------------------------------------------------
RANDOM_STATE       = 42
DATASET_PATH       = None        # Not required for synthetic Two Moons dataset
FEATURES           = ["Feature1", "Feature2"]

# DBSCAN Hyperparameter Search Space
EPS_VALUES         = [0.3, 0.5, 0.8, 1.0, 1.5]
MIN_SAMPLES_VALUES = [2, 3, 4, 5]

# Visualization
DPI                = 100
STYLE              = 'seaborn-v0_8-darkgrid'
FIGURE_SIZE        = (12, 8)
CLUSTER_CMAP       = 'tab10'

# Output
OUTPUT_CSV         = "DBSCAN\\dbscan_clustered.csv"

CLUSTER_COLORS     = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]


# --- Data Class ---------------------------------------------------------------
@dataclass
class DBSCANMetrics:
    """Stores evaluation metrics for a DBSCAN run."""
    eps:                float
    min_samples:        int
    n_clusters:         int
    n_noise:            int
    noise_ratio:        float
    silhouette:         float
    davies_bouldin:     float
    calinski_harabasz:  float

    def __str__(self) -> str:
        return (
            f"\nDBSCAN Evaluation Metrics:\n"
            f"{'=' * 55}\n"
            f"  eps             : {self.eps}\n"
            f"  min_samples     : {self.min_samples}\n"
            f"  Clusters found  : {self.n_clusters}\n"
            f"  Noise points    : {self.n_noise}  ({self.noise_ratio * 100:.1f}%)\n"
            f"  Silhouette      : {self.silhouette:.4f}  (higher is better)\n"
            f"  Davies-Bouldin  : {self.davies_bouldin:.4f}  (lower is better)\n"
            f"  Calinski-Harabasz: {self.calinski_harabasz:.4f}  (higher is better)\n"
            f"{'=' * 55}"
        )


@dataclass
class TuningResult:
    """Stores the outcome of a single hyperparameter trial."""
    eps:           float
    min_samples:   int
    n_clusters:    int
    n_noise:       int
    noise_ratio:   float
    silhouette:    float


# --- DatasetLoader ------------------------------------------------------------
class DatasetLoader:
    """
    Generates the Two Moons dataset for DBSCAN clustering.

    This dataset is ideal for DBSCAN because clusters are
    non linear and density based.
    """

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path
        self.data: Optional[pd.DataFrame] = None


    def load(self) -> pd.DataFrame:

        print(f"\n{'=' * 70}")
        print("GENERATING TWO MOONS DATASET")
        print(f"{'=' * 70}")

        X, y = make_moons(
            n_samples=500,
            noise=0.05,
            random_state=RANDOM_STATE
        )

        df = pd.DataFrame(
            X,
            columns=["Feature1", "Feature2"]
        )

        df["TrueCluster"] = y

        self.data = df

        print("[OK] Two moons dataset generated")
        print("[OK] Samples :", len(df))
        print("[OK] Columns :", list(df.columns))

        return df


# --- DatasetValidator ---------------------------------------------------------
class DatasetValidator:
    """Validates structure, types, and quality of the loaded dataset."""

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        if self.data.empty:
            print("[ERROR] Dataset is empty!")
            return False
        print("[OK] Dataset is not empty")

        print(f"\n--- Shape ---")
        print(f"Rows: {self.data.shape[0]}  |  Columns: {self.data.shape[1]}")

        print(f"\n--- Missing Values ---")
        miss = self.data.isnull().sum()
        total_miss = miss.sum()
        print(miss[miss > 0] if total_miss > 0 else "No missing values detected")

        print(f"\n--- Data Types ---")
        print(self.data.dtypes)

        print(f"\n--- First 5 Rows ---")
        print(self.data.head())

        print(f"\n--- Statistical Summary ---")
        print(self.data[FEATURES].describe())

        missing_feats = [f for f in FEATURES if f not in self.data.columns]
        if missing_feats:
            print(f"[ERROR] Required feature columns missing: {missing_feats}")
            return False

        print(f"\n[OK] All required features present: {FEATURES}")
        return True


# --- DatasetProcessor ---------------------------------------------------------
class DatasetProcessor:
    """
    Scales features for DBSCAN.

    NOTE: DBSCAN, like KMeans, is a distance-based algorithm.
    StandardScaler is MANDATORY so that Weight and Height contribute
    equally to the epsilon neighbourhood distance calculation.
    Without scaling, a feature with a wider numeric range would
    dominate the density estimates and produce incorrect clusters.
    """

    def __init__(self, data: pd.DataFrame):
        self.data    = data.copy()
        self.scaler  = StandardScaler()
        self.X_raw:    Optional[np.ndarray] = None
        self.X_scaled: Optional[np.ndarray] = None

    def process(self) -> Tuple[np.ndarray, np.ndarray]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        print("\n--- Handling Missing Values ---")
        before = self.data[FEATURES].isnull().sum().sum()
        for col in FEATURES:
            self.data[col].fillna(self.data[col].median(), inplace=True)
        after = self.data[FEATURES].isnull().sum().sum()
        print(f"  Missing before: {before}  ->  after: {after}")

        self.X_raw = self.data[FEATURES].values

        print("\n--- StandardScaler (Required for DBSCAN) ---")
        print("  NOTE: DBSCAN uses epsilon-neighbourhood (distance-based).")
        print("        Feature scaling ensures each dimension is weighted equally.")
        self.X_scaled = self.scaler.fit_transform(self.X_raw)

        print(f"\n  Feature means  : {self.scaler.mean_}")
        print(f"  Feature stdevs : {np.sqrt(self.scaler.var_)}")
        print(f"\n[OK] Scaled feature shape: {self.X_scaled.shape}")
        return self.X_raw, self.X_scaled


# --- ClusteringVisualizer -----------------------------------------------------
class ClusteringVisualizer:
    """Produces all exploratory and post-clustering visualisations."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
        plt.style.use(STYLE)

    def visualize_raw(self):
        print(f"\n{'=' * 70}")
        print("EXPLORATORY VISUALIZATION (RAW DATA)")
        print(f"{'=' * 70}")

        self._plot_missing_values()
        self._plot_raw_scatter()
        self._plot_feature_distributions()
        self._plot_correlation_heatmap()
        self._plot_boxplots()
        print("[OK] All raw data visualisations saved")

    def _plot_missing_values(self):
        if self.data.isnull().sum().sum() == 0:
            fig, ax = plt.subplots(figsize=(6, 3), dpi=DPI)
            ax.text(0.5, 0.5, 'No missing values in dataset',
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
        else:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=DPI)
            sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis', ax=ax)
            ax.set_title('Missing Values Heatmap', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_missing_values.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Missing values plot saved")

    def _plot_raw_scatter(self):
        plt.figure(figsize=(9, 6), dpi=DPI)
        plt.scatter(self.data[FEATURES[0]], self.data[FEATURES[1]],
                    alpha=0.6, edgecolors='black', linewidths=0.3,
                    color='#1f77b4', s=60)
        plt.title("Raw Data Before Clustering", fontsize=12, fontweight='bold')
        plt.xlabel(FEATURES[0])
        plt.ylabel(FEATURES[1])
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_raw_scatter.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Raw scatter plot saved")

    def _plot_feature_distributions(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)
        for ax, col in zip(axes, FEATURES):
            ax.hist(self.data[col], bins=30, color='#1f77b4',
                    edgecolor='black', alpha=0.8)
            ax.axvline(self.data[col].mean(), color='red', linestyle='--',
                       label=f'Mean: {self.data[col].mean():.1f}')
            ax.axvline(self.data[col].median(), color='green', linestyle='-.',
                       label=f'Median: {self.data[col].median():.1f}')
            ax.set_title(f'{col} Distribution', fontsize=11, fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_feature_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature distributions saved")

    def _plot_boxplots(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)
        for ax, col, color in zip(axes, FEATURES, ['#1f77b4', '#ff7f0e']):
            ax.boxplot(self.data[col], patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.7))
            ax.set_title(f'{col} Boxplot', fontsize=11, fontweight='bold')
            ax.set_ylabel(col)
            ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_boxplots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Boxplots saved")

    def _plot_correlation_heatmap(self):
        num_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return
        plt.figure(figsize=(7, 5), dpi=DPI)
        sns.heatmap(self.data[num_cols].corr(), annot=True, cmap='coolwarm',
                    fmt='.2f', center=0, square=True,
                    linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title("Feature Correlation Heatmap", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Correlation heatmap saved")


# --- DBSCANHyperparameterTuner ------------------------------------------------
class DBSCANHyperparameterTuner:
    """
    Grid-searches eps x min_samples space.

    Selection criteria
    ------------------
    1. Maximise number of valid clusters (exclude noise label -1).
    2. Among runs with the same cluster count, prefer highest silhouette score.
    3. Skips degenerate runs (all noise or single cluster) automatically.
    """

    def __init__(self, eps_values: List[float], min_samples_values: List[int]):
        self.eps_values         = eps_values
        self.min_samples_values = min_samples_values
        self.results: List[TuningResult] = []

    def tune(self, X_scaled: np.ndarray) -> Tuple[DBSCAN, np.ndarray]:
        print(f"\n{'=' * 70}")
        print("DBSCAN HYPERPARAMETER TUNING")
        print(f"{'=' * 70}")
        print(f"  eps range         : {self.eps_values}")
        print(f"  min_samples range : {self.min_samples_values}")
        print(f"  Total trials      : {len(self.eps_values) * len(self.min_samples_values)}\n")

        best_model:   Optional[DBSCAN]    = None
        best_labels:  Optional[np.ndarray] = None
        best_score:   float = -np.inf
        best_clusters: int  = -1

        header = f"{'eps':>6}  {'min_smp':>7}  {'clusters':>8}  {'noise':>6}  {'silhouette':>10}"
        print(header)
        print('-' * len(header))

        for eps in self.eps_values:
            for min_s in self.min_samples_values:
                model  = DBSCAN(eps=eps, min_samples=min_s)
                labels = model.fit_predict(X_scaled)

                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise    = int(np.sum(labels == -1))
                noise_ratio = n_noise / len(labels)

                # Silhouette requires >= 2 clusters and >= 2 non-noise points
                non_noise_mask = labels != -1
                if n_clusters >= 2 and non_noise_mask.sum() >= 2:
                    sil = silhouette_score(X_scaled[non_noise_mask],
                                           labels[non_noise_mask])
                else:
                    sil = -1.0

                print(f"{eps:>6.2f}  {min_s:>7}  {n_clusters:>8}  "
                      f"{n_noise:>6}  {sil:>10.4f}")

                self.results.append(TuningResult(
                    eps=eps, min_samples=min_s, n_clusters=n_clusters,
                    n_noise=n_noise, noise_ratio=noise_ratio, silhouette=sil
                ))

                # Choose best: more clusters first, then higher silhouette
                is_better = (
                    n_clusters > best_clusters or
                    (n_clusters == best_clusters and sil > best_score)
                )
                if n_clusters >= 1 and is_better:
                    best_clusters = n_clusters
                    best_score    = sil
                    best_model    = model
                    best_labels   = labels

        print(f"\n[OK] Best params -> eps={best_model.eps}, "
              f"min_samples={best_model.min_samples}")
        print(f"     Clusters    : {best_clusters}")
        print(f"     Silhouette  : {best_score:.4f}")

        self._plot_tuning_heatmaps()
        return best_model, best_labels

    def _plot_tuning_heatmaps(self):
        eps_vals = sorted(set(r.eps for r in self.results))
        min_vals = sorted(set(r.min_samples for r in self.results))

        cluster_grid = np.zeros((len(min_vals), len(eps_vals)))
        sil_grid     = np.zeros((len(min_vals), len(eps_vals)))

        for r in self.results:
            i = min_vals.index(r.min_samples)
            j = eps_vals.index(r.eps)
            cluster_grid[i, j] = r.n_clusters
            sil_grid[i, j]     = max(r.silhouette, 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        sns.heatmap(cluster_grid, annot=True, fmt='.0f', cmap='YlOrRd',
                    xticklabels=[str(e) for e in eps_vals],
                    yticklabels=[str(m) for m in min_vals],
                    ax=axes[0], cbar_kws={"shrink": 0.8})
        axes[0].set_title('Number of Clusters (eps x min_samples)',
                           fontsize=11, fontweight='bold')
        axes[0].set_xlabel('eps')
        axes[0].set_ylabel('min_samples')

        sns.heatmap(sil_grid, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=[str(e) for e in eps_vals],
                    yticklabels=[str(m) for m in min_vals],
                    ax=axes[1], cbar_kws={"shrink": 0.8})
        axes[1].set_title('Silhouette Score (eps x min_samples)',
                           fontsize=11, fontweight='bold')
        axes[1].set_xlabel('eps')
        axes[1].set_ylabel('min_samples')

        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_tuning_heatmaps.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Tuning heatmaps saved")


# --- DBSCANModel --------------------------------------------------------------
class DBSCANModel:
    """
    DBSCAN model wrapper with evaluation and k-distance plot support.

    Key differences from KMeans / Decision Tree
    --------------------------------------------
    - KMeans: partitions ALL points into K clusters (no noise concept).
    - Decision Tree: supervised, builds rule-based decision boundaries.
    - DBSCAN: density-based; discovers arbitrarily-shaped clusters AND
      labels low-density points as noise (-1).  Does NOT require K.
    - Like KMeans, DBSCAN is distance-based -> feature scaling required.
    """

    def __init__(self, model: DBSCAN, labels: np.ndarray):
        self.model  = model
        self.labels = labels

    def compute_metrics(self, X_scaled: np.ndarray) -> DBSCANMetrics:
        n_clusters  = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise     = int(np.sum(self.labels == -1))
        noise_ratio = n_noise / len(self.labels)

        mask = self.labels != -1
        if n_clusters >= 2 and mask.sum() >= 2:
            sil = silhouette_score(X_scaled[mask], self.labels[mask])
            dbi = davies_bouldin_score(X_scaled[mask], self.labels[mask])
            chi = calinski_harabasz_score(X_scaled[mask], self.labels[mask])
        else:
            sil = dbi = chi = float('nan')

        return DBSCANMetrics(
            eps=self.model.eps,
            min_samples=self.model.min_samples,
            n_clusters=n_clusters,
            n_noise=n_noise,
            noise_ratio=noise_ratio,
            silhouette=sil,
            davies_bouldin=dbi,
            calinski_harabasz=chi,
        )

    @staticmethod
    def plot_k_distance(X_scaled: np.ndarray, k: int = 4):
        """
        K-distance plot for epsilon selection guidance.
        The 'elbow' in this curve is a good starting point for eps.
        """
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        distances, _ = nbrs.kneighbors(X_scaled)
        k_distances  = np.sort(distances[:, k - 1])[::-1]

        plt.figure(figsize=(9, 5), dpi=DPI)
        plt.plot(k_distances, linewidth=2, color='#1f77b4')
        plt.title(f"K-Distance Plot (k={k}) — Use Elbow to Choose eps",
                  fontsize=12, fontweight='bold')
        plt.xlabel("Points (sorted by distance)")
        plt.ylabel(f"{k}-NN Distance")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_k_distance.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] K-distance plot saved")


# --- ModelEvaluator -----------------------------------------------------------
class ModelEvaluator:
    """Generates full post-clustering evaluation plots."""

    def __init__(self, dbscan_model: DBSCANModel, scaler: StandardScaler):
        self.dbscan_model = dbscan_model
        self.scaler       = scaler

    def evaluate(self, X_raw: np.ndarray, X_scaled: np.ndarray, df: pd.DataFrame):
        print(f"\n{'=' * 70}")
        print("MODEL EVALUATION")
        print(f"{'=' * 70}")

        metrics = self.dbscan_model.compute_metrics(X_scaled)
        print(metrics)

        self._plot_cluster_result(X_raw)
        self._plot_noise_vs_clusters(X_raw)
        self._plot_cluster_distributions(df)
        self._plot_pca_projection(X_scaled)
        self._plot_metrics_summary(metrics)

    def _unique_labels(self):
        return sorted(set(self.dbscan_model.labels))

    def _plot_cluster_result(self, X_raw: np.ndarray):
        labels     = self.dbscan_model.labels
        unique_lbl = self._unique_labels()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=DPI)

        # Before clustering
        axes[0].scatter(X_raw[:, 0], X_raw[:, 1],
                        alpha=0.6, edgecolors='black', linewidths=0.3,
                        color='#1f77b4', s=60)
        axes[0].set_title("Before Clustering (Raw Data)",
                           fontsize=12, fontweight='bold')
        axes[0].set_xlabel(FEATURES[0])
        axes[0].set_ylabel(FEATURES[1])
        axes[0].grid(True, alpha=0.3)

        # After clustering
        for lbl in unique_lbl:
            mask  = labels == lbl
            color = 'black' if lbl == -1 else CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
            label = 'Noise' if lbl == -1 else f'Cluster {lbl}'
            marker = 'x' if lbl == -1 else 'o'
            axes[1].scatter(X_raw[mask, 0], X_raw[mask, 1],
                            color=color, label=label,
                            marker=marker, alpha=0.7,
                            edgecolors='black' if lbl != -1 else None,
                            linewidths=0.3, s=60)

        axes[1].set_title(
            f"DBSCAN Result (eps={self.dbscan_model.model.eps}, "
            f"min_samples={self.dbscan_model.model.min_samples})",
            fontsize=12, fontweight='bold')
        axes[1].set_xlabel(FEATURES[0])
        axes[1].set_ylabel(FEATURES[1])
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_cluster_result.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Cluster result plot saved")

    def _plot_noise_vs_clusters(self, X_raw: np.ndarray):
        labels  = self.dbscan_model.labels
        n_total = len(labels)
        n_noise = int(np.sum(labels == -1))
        n_core  = n_total - n_noise

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        # Pie: noise vs clustered
        axes[0].pie([n_core, n_noise],
                    labels=['Clustered', 'Noise'],
                    autopct='%1.1f%%',
                    colors=['#1f77b4', '#d62728'],
                    startangle=90, explode=[0, 0.05])
        axes[0].set_title("Noise vs Clustered Points",
                           fontsize=11, fontweight='bold')

        # Bar: cluster sizes
        unique_lbl = [l for l in self._unique_labels() if l != -1]
        sizes = [int(np.sum(labels == l)) for l in unique_lbl]
        colors_bar = [CLUSTER_COLORS[l % len(CLUSTER_COLORS)] for l in unique_lbl]
        bars = axes[1].bar([f'Cluster {l}' for l in unique_lbl],
                           sizes, color=colors_bar,
                           edgecolor='black', alpha=0.8)
        for bar, sz in zip(bars, sizes):
            axes[1].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                         str(sz), ha='center', va='bottom', fontsize=11)
        axes[1].set_title("Cluster Sizes", fontsize=11, fontweight='bold')
        axes[1].set_ylabel("Number of Points")
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_noise_vs_clusters.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Noise vs clusters plot saved")

    def _plot_cluster_distributions(self, df: pd.DataFrame):
        labels     = self.dbscan_model.labels
        unique_lbl = self._unique_labels()

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)

        for feat, ax_row in zip(FEATURES, axes):
            # Histogram
            for lbl in unique_lbl:
                mask   = labels == lbl
                color  = 'black' if lbl == -1 else CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
                label  = 'Noise' if lbl == -1 else f'Cluster {lbl}'
                ax_row[0].hist(df.loc[mask, feat].values, bins=15,
                               alpha=0.6, label=label,
                               color=color, edgecolor='black')
            ax_row[0].set_title(f'{feat} Distribution by Cluster',
                                fontsize=11, fontweight='bold')
            ax_row[0].set_xlabel(feat)
            ax_row[0].set_ylabel('Frequency')
            ax_row[0].legend()
            ax_row[0].grid(True, alpha=0.3)

            # Boxplot
            cluster_data = [df.loc[labels == lbl, feat].values
                            for lbl in unique_lbl]
            xlabels = ['Noise' if l == -1 else f'Cluster {l}' for l in unique_lbl]
            bp = ax_row[1].boxplot(cluster_data, patch_artist=True, labels=xlabels)
            for patch, lbl in zip(bp['boxes'], unique_lbl):
                patch.set_facecolor('gray' if lbl == -1
                                    else CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)])
                patch.set_alpha(0.7)
            ax_row[1].set_title(f'{feat} Spread per Cluster',
                                fontsize=11, fontweight='bold')
            ax_row[1].set_ylabel(feat)
            ax_row[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_cluster_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Cluster distributions saved")

    def _plot_pca_projection(self, X_scaled: np.ndarray):
        pca   = PCA(n_components=2, random_state=RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        labels = self.dbscan_model.labels

        plt.figure(figsize=(9, 6), dpi=DPI)
        for lbl in self._unique_labels():
            mask   = labels == lbl
            color  = 'black' if lbl == -1 else CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)]
            label  = 'Noise' if lbl == -1 else f'Cluster {lbl}'
            marker = 'x' if lbl == -1 else 'o'
            plt.scatter(X_pca[mask, 0], X_pca[mask, 1],
                        color=color, label=label, marker=marker,
                        alpha=0.7, edgecolors='black' if lbl != -1 else None,
                        linewidths=0.3, s=60)

        plt.title("PCA Projection of DBSCAN Clusters", fontsize=12, fontweight='bold')
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_pca_projection.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] PCA projection saved")

    def _plot_metrics_summary(self, metrics: DBSCANMetrics):
        labels = ['Silhouette', 'DB Index\n(inverted)', 'CH Score\n(scaled)']

        sil = metrics.silhouette if not np.isnan(metrics.silhouette) else 0
        dbi = (1.0 / (1.0 + metrics.davies_bouldin)
               if not np.isnan(metrics.davies_bouldin) else 0)
        chi = min(metrics.calinski_harabasz / 1000.0, 1.0) \
              if not np.isnan(metrics.calinski_harabasz) else 0

        values = [sil, dbi, chi]
        colors_bar = ['#1f77b4', '#ff7f0e', '#2ca02c']

        plt.figure(figsize=(8, 5), dpi=DPI)
        bars = plt.bar(labels, values, color=colors_bar,
                       edgecolor='black', alpha=0.8)
        plt.ylim([0, 1.1])
        plt.ylabel('Score (normalised for display)')
        plt.title('Cluster Quality Metrics Summary', fontsize=12, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                     f'{val:.3f}', ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        plt.savefig("DBSCAN\\graphs\\dbscan_metrics_summary.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Metrics summary plot saved")


# --- MLPipeline ---------------------------------------------------------------
class MLPipeline:
    """End-to-end DBSCAN clustering pipeline for Weight/Height data."""

    def __init__(self):
        self.loader    = DatasetLoader(DATASET_PATH)
        self.processor = None
        self.model     = None
        self.evaluator = None

    def run(self):
        print(f"\n{'=' * 70}")
        print("DBSCAN PIPELINE --- TWO MOONS CLUSTERING")
        print(f"{'=' * 70}")

        # 1. Load
        df = self.loader.load()

        # 2. Visualise raw data
        visualizer = ClusteringVisualizer(df)
        visualizer.visualize_raw()

        # 3. Validate
        validator = DatasetValidator(df)
        if not validator.verify_dataset():
            raise RuntimeError("Dataset validation failed. Aborting pipeline.")

        # 4. Process (scale features)
        self.processor = DatasetProcessor(df)
        X_raw, X_scaled = self.processor.process()

        # 5. K-distance plot (eps guidance)
        DBSCANModel.plot_k_distance(X_scaled, k=4)

        # 6. Hyperparameter tuning
        tuner = DBSCANHyperparameterTuner(EPS_VALUES, MIN_SAMPLES_VALUES)
        best_dbscan, best_labels = tuner.tune(X_scaled)

        # 7. Build model wrapper
        self.model = DBSCANModel(model=best_dbscan, labels=best_labels)

        # 8. Attach labels to dataframe & print summary
        df['Cluster'] = best_labels
        self._print_cluster_summary(df)

        # 9. Evaluate & plot
        self.evaluator = ModelEvaluator(self.model, self.processor.scaler)
        self.evaluator.evaluate(X_raw, X_scaled, df)

        # 10. Save output
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[OK] Labelled dataset saved -> {OUTPUT_CSV}")

        # 11. Predict a new point
        self._predict_new_point()

    def _print_cluster_summary(self, df: pd.DataFrame):
        print(f"\n{'=' * 70}")
        print("CLUSTER SUMMARY")
        print(f"{'=' * 70}")

        for cl in sorted(df['Cluster'].unique()):
            subset = df[df['Cluster'] == cl]
            tag    = "NOISE" if cl == -1 else f"Cluster {cl}"
            print(f"\n  {tag}  ({len(subset)} points):")
            for feat in FEATURES:
                print(f"    {feat:<10}: min={subset[feat].min():.1f}  "
                      f"max={subset[feat].max():.1f}  "
                      f"mean={subset[feat].mean():.1f}")

    def _predict_new_point(self):
        print(f"\n{'=' * 70}")
        print("NEW POINT PREDICTION")
        print(f"{'=' * 70}")

        new_point = {FEATURES[0]: 1.2, FEATURES[1]: 0.3}
        print("New point profile:")
        for k, v in new_point.items():
            print(f"  {k}: {v}")

        X_new        = np.array([[new_point[f] for f in FEATURES]])
        X_new_scaled = self.processor.scaler.transform(X_new)

        # DBSCAN has no built-in predict; assign to nearest cluster centroid
        labels = self.model.labels
        unique_clusters = [l for l in set(labels) if l != -1]

        if not unique_clusters:
            print("[WARN] No valid clusters found for prediction.")
            return

        X_scaled = self.processor.X_scaled
        centroids = np.array([
            X_scaled[labels == cl].mean(axis=0) for cl in unique_clusters
        ])

        distances  = np.linalg.norm(X_new_scaled - centroids, axis=1)
        nearest_cl = unique_clusters[int(np.argmin(distances))]

        print(f"\n  Predicted Cluster : {nearest_cl}")
        print(f"\n  Distances to cluster centroids (scaled space):")
        for cl, dist in zip(unique_clusters, distances):
            bar = '#' * int(max(0, (3.0 - dist) / 3.0 * 20))
            print(f"    Cluster {cl} : {dist:.4f}  {bar}")


# --- Entry Point --------------------------------------------------------------
def main():
    """Run the DBSCAN clustering pipeline."""
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed:")
        print(f"{type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":

    main()