import warnings
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ===========================================================
# GLOBAL CONFIGURATION
# ===========================================================
RANDOM_STATE    = 42
DATASET_PATH    = "SVMtrain.csv"
TEST_SIZE       = 0.2

# AdaBoost Hyperparameters
N_ESTIMATORS    = 100     # Number of weak learners
LEARNING_RATE   = 1.0     # Shrinks contribution of each classifier
MAX_DEPTH       = 1       # Depth of base decision stumps (1 = stumps)

# Model Persistence
MODEL_SAVE_PATH = "adaboost_model.pkl"

# Visualization
FIGURE_SIZE = (12, 8)
DPI         = 100
STYLE       = 'seaborn-v0_8-darkgrid'


# ===========================================================
# DATA CLASS
# ===========================================================
@dataclass
class ModelMetrics:
    """Stores evaluation metrics for easy reporting."""
    accuracy:  float
    precision: float
    recall:    float
    f1:        float
    roc_auc:   float

    def __str__(self) -> str:
        return (
            f"Model Performance Metrics:\n"
            f"{'=' * 50}\n"
            f"Accuracy:   {self.accuracy:.4f}\n"
            f"Precision:  {self.precision:.4f}\n"
            f"Recall:     {self.recall:.4f}\n"
            f"F1-Score:   {self.f1:.4f}\n"
            f"ROC-AUC:    {self.roc_auc:.4f}\n"
            f"{'=' * 50}"
        )


# ===========================================================
# DATASET LOADER
# ===========================================================
class DatasetLoader:
    """Loads the Titanic dataset from CSV or generates a synthetic fallback."""

    def __init__(self, dataset_path: str = None):
        self.dataset_path  = dataset_path
        self.data: Optional[pd.DataFrame] = None
        self.target: Optional[pd.Series]  = None
        self.feature_names: Optional[list] = None

    def _generate_synthetic_dataset(self, n_samples: int = 891) -> Tuple[pd.DataFrame, pd.Series]:
        """Generate a synthetic Titanic-like dataset as fallback."""
        np.random.seed(RANDOM_STATE)

        pclass   = np.random.choice([1, 2, 3], n_samples, p=[0.24, 0.21, 0.55])
        sex      = np.random.choice([0, 1], n_samples)   # 0=male, 1=female
        age      = np.clip(np.random.normal(29, 14, n_samples), 1, 80)
        sibsp    = np.random.choice([0, 1, 2, 3], n_samples, p=[0.68, 0.23, 0.07, 0.02])
        parch    = np.random.choice([0, 1, 2], n_samples, p=[0.76, 0.13, 0.11])
        fare     = np.clip(np.random.exponential(32, n_samples), 0, 512)
        embarked = np.random.choice([1, 2, 3], n_samples, p=[0.19, 0.09, 0.72])

        survival_prob = (
            0.35
            + 0.35 * sex
            + 0.10 * (pclass == 1).astype(float)
            - 0.10 * (pclass == 3).astype(float)
            + np.random.normal(0, 0.1, n_samples)
        )
        survived = (survival_prob > 0.5).astype(int)

        df = pd.DataFrame({
            'Pclass': pclass, 'Sex': sex,   'Age': age,
            'SibSp':  sibsp,  'Parch': parch, 'Fare': fare,
            'Embarked': embarked,
        })

        self.data          = df
        self.target        = pd.Series(survived, name='Survived')
        self.feature_names = list(df.columns)
        return self.data, self.target

    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("LOADING DATASET")
        print(f"{'=' * 70}")

        if self.dataset_path and os.path.exists(self.dataset_path):
            try:
                df = pd.read_csv(self.dataset_path)
                print(f"[OK] Loaded dataset from : {self.dataset_path}")
                print(f"[OK] Raw columns         : {list(df.columns)}")

                drop_cols = [c for c in ['PassengerId', 'Name', 'Ticket', 'Cabin']
                             if c in df.columns]
                if drop_cols:
                    df = df.drop(columns=drop_cols)
                    print(f"[OK] Dropped columns     : {drop_cols}")

                self.data          = df.drop(columns=['Survived'])
                self.target        = df['Survived']
                self.feature_names = list(self.data.columns)

                print(f"[OK] Samples             : {len(self.data)}")
                print(f"[OK] Features            : {len(self.feature_names)}")
                print(f"[OK] Feature names       : {', '.join(self.feature_names)}")
                return self.data, self.target

            except Exception as ex:
                print(f"[WARNING] Failed to load {self.dataset_path}: {ex}")
                print("[WARNING] Falling back to synthetic dataset generation")

        self.data, self.target = self._generate_synthetic_dataset()
        print(f"[OK] Synthetic Titanic dataset generated")
        print(f"[OK] Samples  : {len(self.data)}")
        print(f"[OK] Features : {', '.join(self.feature_names)}")
        return self.data, self.target


# ===========================================================
# DATASET VALIDATOR
# ===========================================================
class DatasetValidator:
    """Validates the dataset before processing."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target

    def verify_dataset(self) -> bool:
        print(f"\n{'=' * 70}")
        print("DATASET VERIFICATION")
        print(f"{'=' * 70}")

        if self.data.empty or self.target.empty:
            print("[FAIL] Dataset is empty!")
            return False
        print("[OK] Dataset is not empty")

        print("\n--- Shape ---")
        print(f"  Features : {self.data.shape}")
        print(f"  Target   : {self.target.shape}")

        if self.data.shape[0] != self.target.shape[0]:
            print("[FAIL] Row count mismatch between features and target!")
            return False
        print("[OK] Row counts match")

        print("\n--- Missing Values ---")
        missing = self.data.isnull().sum()
        print(missing[missing > 0].to_string() if missing.sum() > 0 else "  None")

        print("\n--- First 5 Rows ---")
        print(self.data.head())

        print("\n--- Statistical Summary ---")
        print(self.data.describe())

        print("\n--- Class Distribution ---")
        counts = self.target.value_counts()
        print(counts)
        print(f"  Class balance: {counts.min() / counts.max():.2f}")

        return True


# ===========================================================
# DATASET PROCESSOR
# ===========================================================
class DatasetProcessor:
    """Handles preprocessing: encoding and imputation."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data.copy()
        self.target = target.copy()
        self.processed_data: Optional[pd.DataFrame]  = None
        self.processed_target: Optional[pd.Series]   = None
        self.label_encoders = {}

    def process_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        print(f"\n{'=' * 70}")
        print("DATASET PROCESSING")
        print(f"{'=' * 70}")

        # Encode categorical columns
        print("\n--- Encoding Categorical Features ---")
        cat_cols = self.data.select_dtypes(include=['object']).columns.tolist()
        for col in cat_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col].astype(str))
            self.label_encoders[col] = le
            print(f"  [OK] Encoded: {col}")
        if not cat_cols:
            print("  No categorical columns found (already numeric)")

        # Impute missing values
        print("\n--- Imputing Missing Values ---")
        missing_before = self.data.isnull().sum().sum()
        for col in self.data.columns:
            if self.data[col].isnull().any():
                if self.data[col].dtype in ['float64', 'int64']:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                else:
                    self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        missing_after = self.data.isnull().sum().sum()
        print(f"  Missing before : {missing_before}")
        print(f"  Missing after  : {missing_after}")

        # Target conversion
        print("\n--- Target Conversion ---")
        self.processed_target = self.target.astype(int)
        print(f"  Classes: {sorted(self.processed_target.unique())}")

        self.processed_data = self.data.copy()

        print("\n--- Processed Shape ---")
        print(f"  Features : {self.processed_data.shape}")
        print(f"  Target   : {self.processed_target.shape}")

        return self.processed_data, self.processed_target


# ===========================================================
# VISUALIZER
# ===========================================================
class TitanicVisualizer:
    """Rich visualizations for the Titanic dataset."""

    def __init__(self, data: pd.DataFrame, target: pd.Series):
        self.data   = data
        self.target = target
        plt.style.use(STYLE)

    def visualize(self):
        print(f"\n{'=' * 70}")
        print("TITANIC DATASET VISUALIZATION")
        print(f"{'=' * 70}")
        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_feature_distributions()
        self.plot_feature_boxplots()
        self.plot_survival_by_feature()
        self.plot_feature_importance_proxy()
        print("[OK] All visualizations saved")

    # 1. Class Distribution
    def plot_target_distribution(self):
        counts = self.target.value_counts().sort_index()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=DPI)

        bars = axes[0].bar(
            ['Not Survived (0)', 'Survived (1)'], counts.values,
            color=['#d62728', '#2ca02c'], edgecolor='black', alpha=0.8
        )
        axes[0].set_title("Survival Class Distribution", fontsize=13, fontweight='bold')
        axes[0].set_ylabel("Count")
        axes[0].grid(True, alpha=0.3, axis='y')
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2., bar.get_height(),
                f'{int(bar.get_height())}', ha='center', va='bottom', fontweight='bold'
            )

        axes[1].pie(
            counts.values, labels=['Not Survived', 'Survived'],
            autopct='%1.1f%%', colors=['#d62728', '#2ca02c'],
            startangle=90, explode=(0.05, 0.05)
        )
        axes[1].set_title("Survival Ratio", fontsize=13, fontweight='bold')

        plt.tight_layout()
        plt.savefig("titanic_target_distribution.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Target distribution saved")

    # 2. Correlation Heatmap
    def plot_correlation_heatmap(self):
        df = self.data.copy()
        for col in df.select_dtypes(include='object').columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        df['Survived'] = self.target

        plt.figure(figsize=(12, 9), dpi=DPI)
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f',
                    center=0, square=True, linewidths=1)
        plt.title("Feature Correlation Heatmap", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig("titanic_correlation_heatmap.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Correlation heatmap saved")

    # 3. Feature Distributions
    def plot_feature_distributions(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        n_cols = 3
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), dpi=DPI)
        # Ensure axes is always a flat array regardless of shape
        axes = np.array(axes).flatten()

        for i, col in enumerate(numeric_cols):
            axes[i].hist(self.data[self.target == 0][col], bins=30,
                         alpha=0.6, label='Not Survived', color='#d62728', edgecolor='black')
            axes[i].hist(self.data[self.target == 1][col], bins=30,
                         alpha=0.6, label='Survived',     color='#2ca02c', edgecolor='black')
            axes[i].set_title(col, fontsize=11, fontweight='bold')
            axes[i].set_xlabel('Value')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("titanic_feature_distributions.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature distributions saved")

    # 4. Feature Boxplots
    def plot_feature_boxplots(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        df_plot = self.data[numeric_cols].copy()
        df_plot['Survived'] = self.target

        top = min(4, len(numeric_cols))
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=DPI)
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols[:top]):
            sns.boxplot(x='Survived', y=col, data=df_plot,
                        palette=['#d62728', '#2ca02c'], ax=axes[i])
            axes[i].set_title(f"{col} vs Survived", fontsize=11, fontweight='bold')
            axes[i].set_xticklabels(['Not Survived', 'Survived'])
            axes[i].grid(True, alpha=0.3, axis='y')

        for idx in range(top, 4):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig("titanic_feature_boxplots.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature boxplots saved")

    # 5. Survival Rate by Feature
    def plot_survival_by_feature(self):
        df = self.data.copy()
        df['Survived'] = self.target

        cat_candidates = [c for c in ['Pclass', 'Sex', 'Embarked', 'SibSp', 'Parch']
                          if c in df.columns]
        if not cat_candidates:
            cat_candidates = list(df.select_dtypes(include=[np.number]).columns[:4])

        n = min(4, len(cat_candidates))
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=DPI)
        if n == 1:
            axes = [axes]

        for i, col in enumerate(cat_candidates[:n]):
            survival_rate = df.groupby(col)['Survived'].mean()
            bars = axes[i].bar(
                survival_rate.index.astype(str), survival_rate.values,
                color='#1f77b4', edgecolor='black', alpha=0.8
            )
            axes[i].set_title(f"Survival Rate by {col}", fontsize=11, fontweight='bold')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Survival Rate")
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3, axis='y')
            for bar in bars:
                h = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width() / 2., h,
                             f'{h:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig("titanic_survival_by_feature.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Survival-by-feature plots saved")

    # 6. Feature Importance (Correlation Proxy)
    def plot_feature_importance_proxy(self):
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            print("[WARNING] No numeric columns available for feature importance plot")
            return

        corr = (self.data[numeric_cols]
                .corrwith(self.target)
                .abs()
                .sort_values(ascending=True))

        plt.figure(figsize=(10, 5), dpi=DPI)
        colors = ['#d62728' if v == corr.max() else '#1f77b4' for v in corr.values]
        bars = plt.barh(corr.index, corr.values, color=colors, edgecolor='black', alpha=0.8)
        plt.xlabel('Absolute Correlation with Survived', fontsize=11)
        plt.title('Feature Importance (Correlation Proxy)', fontsize=13, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        for bar, val in zip(bars, corr.values):
            plt.text(val + 0.005, bar.get_y() + bar.get_height() / 2.,
                     f'{val:.3f}', va='center', fontsize=9)
        plt.tight_layout()
        plt.savefig("titanic_feature_importance.png", dpi=DPI, bbox_inches='tight')
        plt.close()
        print("[OK] Feature importance plot saved")


# ===========================================================
# ADABOOST MODEL
# ===========================================================
class AdaBoostModel:
    """AdaBoost Classifier wrapper with train / predict / evaluate."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 1,
                 learning_rate: float = 1.0):
        self.n_estimators  = n_estimators
        self.max_depth     = max_depth
        self.learning_rate = learning_rate

        base_estimator = DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=RANDOM_STATE
        )
        # Note: algorithm parameter removed in scikit-learn 1.6+ (SAMME is now the only algorithm)
        self.model = AdaBoostClassifier(
            estimator=base_estimator,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=RANDOM_STATE
        )

    def fit(self, X_train, y_train):
        print(f"\n{'=' * 60}")
        print("TRAINING ADABOOST MODEL (TITANIC)")
        print(f"{'=' * 60}")
        print(f"  Estimators    : {self.n_estimators}")
        print(f"  Base depth    : {self.max_depth}")
        print(f"  Learning rate : {self.learning_rate}")
        print(f"  Algorithm     : SAMME")

        self.model.fit(X_train, y_train)
        print("[OK] AdaBoost training complete")

        try:
            joblib.dump(self.model, MODEL_SAVE_PATH)
            print(f"[OK] Model saved -> {MODEL_SAVE_PATH}")
        except Exception as ex:
            print(f"[WARNING] Could not save model: {ex}")

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y_true, name="Dataset"):
        print(f"\n{'=' * 60}")
        print(f"EVALUATION - {name}")
        print(f"{'=' * 60}")

        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC-AUC   : {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=['Not Survived', 'Survived']))
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}


# ===========================================================
# MODEL EVALUATOR (WITH PLOTS)
# ===========================================================
class ModelEvaluator:
    """Generates full evaluation plots for the AdaBoost model."""

    def __init__(self, model: AdaBoostModel):
        self.model = model
        plt.style.use(STYLE)

    def evaluate(self, X, y_true, dataset_name="Dataset"):
        print(f"\n{'=' * 70}")
        print(f"FULL MODEL EVALUATION - {dataset_name}")
        print(f"{'=' * 70}")

        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)

        acc  = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1   = f1_score(y_true, y_pred, zero_division=0)
        auc  = roc_auc_score(y_true, y_prob)

        print(f"Accuracy  : {acc:.4f}")
        print(f"Precision : {prec:.4f}")
        print(f"Recall    : {rec:.4f}")
        print(f"F1 Score  : {f1:.4f}")
        print(f"ROC AUC   : {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=['Not Survived', 'Survived']))

        self._plot_evaluation(y_true, y_pred, y_prob, dataset_name)
        self._plot_prediction_analysis(y_true, y_pred, y_prob, dataset_name)
        self._plot_adaboost_learning_curve(y_true, X, dataset_name)

    def _plot_evaluation(self, y_true, y_pred, y_prob, dataset_name):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=DPI)

        # Confusion Matrix
        cm      = confusion_matrix(y_true, y_pred)
        cm_pct  = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis] * 100
        annot   = np.array([[f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)'
                             for j in range(2)] for i in range(2)])
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes[0, 0],
                    xticklabels=['Not Survived', 'Survived'],
                    yticklabels=['Not Survived', 'Survived'])
        axes[0, 0].set_title(f'Confusion Matrix - {dataset_name}', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val      = roc_auc_score(y_true, y_prob)
        axes[0, 1].plot(fpr, tpr, linewidth=2.5,
                        label=f'ROC (AUC={auc_val:.3f})', color='#1f77b4')
        axes[0, 1].plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random')
        axes[0, 1].fill_between(fpr, tpr, alpha=0.15, color='#1f77b4')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title(f'ROC Curve - {dataset_name}', fontsize=12, fontweight='bold')
        axes[0, 1].legend(loc='lower right')
        axes[0, 1].grid(True, alpha=0.3)

        # Probability Distribution
        axes[1, 0].hist(y_prob[y_true == 0], bins=25, alpha=0.6,
                        label='Not Survived', color='#d62728', edgecolor='black')
        axes[1, 0].hist(y_prob[y_true == 1], bins=25, alpha=0.6,
                        label='Survived',     color='#2ca02c', edgecolor='black')
        axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=1.5,
                           label='Decision boundary')
        axes[1, 0].set_xlabel('Predicted Probability (Survived)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Prediction Probability Distribution - {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')

        # Metrics Bar Chart
        metric_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metric_values = [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0),
            auc_val,
        ]
        bar_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = axes[1, 1].bar(metric_names, metric_values,
                              color=bar_colors, edgecolor='black', alpha=0.8)
        axes[1, 1].set_ylim(0, 1.1)
        axes[1, 1].set_title(f'Performance Metrics - {dataset_name}',
                             fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, metric_values):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                            f'{val:.3f}', ha='center', va='bottom',
                            fontsize=10, fontweight='bold')

        plt.tight_layout()
        fname = f"evaluation_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Evaluation plot saved -> {fname}")

    def _plot_prediction_analysis(self, y_true, y_pred, y_prob, dataset_name):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=DPI)

        is_correct = (y_true == y_pred)
        axes[0].hist(y_prob[is_correct],  bins=20, alpha=0.6,
                     label='Correct',   color='#2ca02c', edgecolor='black')
        axes[0].hist(y_prob[~is_correct], bins=20, alpha=0.6,
                     label='Incorrect', color='#d62728', edgecolor='black')
        axes[0].set_xlabel('Predicted Probability')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Prediction Confidence - {dataset_name}',
                          fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        # Calibration plot
        n_bins    = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        mean_probs, mean_true = [], []
        for i in range(n_bins):
            mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
            if mask.sum() > 0:
                mean_probs.append(y_prob[mask].mean())
                mean_true.append(np.array(y_true)[mask].mean())

        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        axes[1].plot(mean_probs, mean_true, 'o-', linewidth=2, markersize=8,
                     color='#1f77b4', label='Model Calibration')
        axes[1].set_xlabel('Mean Predicted Probability')
        axes[1].set_ylabel('Actual Positive Rate')
        axes[1].set_title(f'Calibration Plot - {dataset_name}', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f"predictions_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] Prediction analysis saved -> {fname}")

    def _plot_adaboost_learning_curve(self, y_true, X, dataset_name):
        """Plot staged training accuracy (AdaBoost-specific learning curve)."""
        staged_scores = [
            accuracy_score(y_true, y_staged)
            for y_staged in self.model.model.staged_predict(X)
        ]

        plt.figure(figsize=(10, 5), dpi=DPI)
        plt.plot(range(1, len(staged_scores) + 1), staged_scores,
                 linewidth=2, color='#1f77b4', label='Staged Accuracy')
        plt.axhline(staged_scores[-1], color='red', linestyle='--', linewidth=1.5,
                    label=f'Final Accuracy: {staged_scores[-1]:.4f}')
        plt.xlabel('Number of Estimators', fontsize=11)
        plt.ylabel('Accuracy', fontsize=11)
        plt.title(f'AdaBoost Staged Accuracy - {dataset_name}',
                  fontsize=13, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"adaboost_learning_curve_{dataset_name.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=DPI, bbox_inches='tight')
        plt.close()
        print(f"[OK] AdaBoost learning curve saved -> {fname}")


# ===========================================================
# ML PIPELINE
# ===========================================================
class MLPipeline:
    """Full end-to-end pipeline: Load -> Validate -> Process -> Train -> Evaluate."""

    def __init__(self):
        self.loader    = DatasetLoader(DATASET_PATH)
        self.processor = None
        self.model     = None
        self.evaluator = None

    def run(self):
        print("\n" + "=" * 70)
        print("ADABOOST PIPELINE - TITANIC SURVIVAL PREDICTION")
        print("=" * 70)

        # 1. Load
        data, target = self.loader.load_data()

        # 2. Validate
        validator = DatasetValidator(data, target)
        validator.verify_dataset()

        # 3. Visualize raw data
        visualizer = TitanicVisualizer(data, target)
        visualizer.visualize()

        # 4. Process
        self.processor = DatasetProcessor(data, target)
        processed_data, processed_target = self.processor.process_dataset()

        # 5. Split
        X_train, X_test, y_train, y_test = train_test_split(
            processed_data.values, processed_target.values,
            test_size=TEST_SIZE, random_state=RANDOM_STATE,
            stratify=processed_target
        )
        print(f"\nTrain samples : {len(X_train)} | Test samples : {len(X_test)}")

        # 6. Train
        self.model = AdaBoostModel(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE
        )
        self.model.fit(X_train, y_train)

        # 7. Evaluate with plots
        self.evaluator = ModelEvaluator(self.model)
        self.evaluator.evaluate(X_train, y_train, "Training Set")
        self.evaluator.evaluate(X_test,  y_test,  "Test Set")

        # 8. Cross-Validation
        self._cross_validate(processed_data.values, processed_target.values)

        # 9. New Passenger Prediction
        self._predict_new_passenger(processed_data.shape[1],
                                    processed_data.columns.tolist())

    def _cross_validate(self, X, y):
        print(f"\n{'=' * 70}")
        print("K-FOLD CROSS-VALIDATION (5-Fold Stratified)")
        print(f"{'=' * 70}")

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        clf = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=MAX_DEPTH,
                random_state=RANDOM_STATE
            ),
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE
        )

        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            scores = cross_val_score(clf, X, y, cv=cv, scoring=metric)
            print(f"  {metric.capitalize():12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")

        acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        note = '[OK] GOOD generalization' if acc_scores.std() < 0.05 else '[WARNING] HIGH variance'
        print(f"\n  {note}  (std={acc_scores.std():.4f})")

    def _predict_new_passenger(self, n_features, feature_names):
        print(f"\n{'=' * 70}")
        print("NEW PASSENGER PREDICTION")
        print(f"{'=' * 70}")

        # Sample: 3rd-class male, age 28, no family, low fare, Southampton
        sample_values = {
            'Pclass': 3, 'Sex': 0, 'Age': 28,
            'SibSp': 0,  'Parch': 0, 'Fare': 7.9,
            'Embarked': 3,
        }
        row = [sample_values.get(f, 0) for f in feature_names]

        passenger = np.array(row, dtype=float).reshape(1, -1)
        prob = self.model.predict_proba(passenger)[0]
        pred = self.model.predict(passenger)[0]

        print(f"  Passenger features              : {dict(zip(feature_names, row))}")
        print(f"  Predicted Probability (Survived): {prob:.4f}")
        print(f"  Prediction                      : {'[OK] Survived' if pred == 1 else '[FAIL] Not Survived'}")


# ===========================================================
# ENTRY POINT
# ===========================================================
def main():
    """Run the complete AdaBoost Titanic prediction pipeline."""
    try:
        pipeline = MLPipeline()
        pipeline.run()
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    main()
