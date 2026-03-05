import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class Config:
    """Global configuration for the clustering pipeline."""

    FILE_PATH = "Clustering_gmm (2).csv"
    FEATURES = ["Weight", "Height"]

    # Hyperparameter tuning ranges
    EPS_VALUES = [0.3, 0.5, 0.8, 1.0]
    MIN_SAMPLES_VALUES = [2, 3, 4]


class DataLoader:
    """Responsible for loading dataset."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> pd.DataFrame:
        """Load CSV dataset."""
        data = pd.read_csv(self.file_path)
        return data


class DataPreprocessor:
    """Handles feature selection and scaling."""

    def __init__(self, features):
        self.features = features
        self.scaler = StandardScaler()

    def preprocess(self, data: pd.DataFrame):
        """Select and scale features."""
        x = data[self.features]
        x_scaled = self.scaler.fit_transform(x)
        return x, x_scaled


class DBSCANHyperparameterTuner:
    """Tune DBSCAN hyperparameters."""

    def __init__(self, eps_values, min_samples_values):
        self.eps_values = eps_values
        self.min_samples_values = min_samples_values

    def tune(self, x_scaled):
        """Find best DBSCAN parameters."""
        best_model = None
        best_clusters = None
        best_cluster_count = -1

        for eps in self.eps_values:
            for min_samples in self.min_samples_values:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(x_scaled)

                cluster_count = len(set(clusters)) - (1 if -1 in clusters else 0)

                if cluster_count > best_cluster_count:
                    best_cluster_count = cluster_count
                    best_model = model
                    best_clusters = clusters

        return best_model, best_clusters


class ClusterVisualizer:
    """Handles visualization of clustering results."""

    @staticmethod
    def plot_before(data, x_col, y_col):
        """Plot raw dataset before clustering."""
        plt.scatter(data[x_col], data[y_col])
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("Data Before Clustering")
        plt.show()

    @staticmethod
    def plot_after(data, x_col, y_col, clusters):
        """Plot clustered dataset."""
        plt.scatter(data[x_col], data[y_col], c=clusters, cmap="viridis")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title("DBSCAN Clustering Result")
        plt.colorbar(label="Cluster")
        plt.show()


class UnsupervisedLearningPipeline:
    """Complete clustering pipeline."""

    def __init__(self):
        self.loader = DataLoader(Config.FILE_PATH)
        self.preprocessor = DataPreprocessor(Config.FEATURES)
        self.tuner = DBSCANHyperparameterTuner(
            Config.EPS_VALUES,
            Config.MIN_SAMPLES_VALUES
        )

    def run(self):
        """Execute the clustering pipeline."""

        # Load dataset
        data = self.loader.load()

        # Plot raw data
        ClusterVisualizer.plot_before(
            data,
            Config.FEATURES[0],
            Config.FEATURES[1]
        )

        # Preprocess data
        x, x_scaled = self.preprocessor.preprocess(data)

        # Hyperparameter tuning
        model, clusters = self.tuner.tune(x_scaled)

        # Store cluster labels
        data["Cluster"] = clusters

        # Plot clustered data
        ClusterVisualizer.plot_after(
            data,
            Config.FEATURES[0],
            Config.FEATURES[1],
            clusters
        )

        return data


def main():
    """Main execution function."""
    pipeline = UnsupervisedLearningPipeline()
    result = pipeline.run()

    print("\nClustered Data Preview:\n")
    print(result.head())


if __name__ == "__main__":
    main()