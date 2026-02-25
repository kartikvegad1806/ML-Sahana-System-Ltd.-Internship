# =============================================================================
# CREDIT DEFAULT PREDICTION - RANDOM FOREST (PRODUCTION READY)
# =============================================================================

import os
import logging
from dataclasses import dataclass
from typing import Tuple

import matplotlib
matplotlib.use("Agg")  # Use non-GUI backend

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "credit_default.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "credit_default_rf.pkl")

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# =============================================================================
# METRICS STRUCTURE
# =============================================================================

@dataclass
class Metrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(path: str) -> pd.DataFrame:
    logger.info("Loading credit default dataset...")

    if not os.path.exists(path):
        logger.error(f"Dataset not found at: {path}")
        raise FileNotFoundError(f"Dataset not found at: {path}")

    # Skip the first row (X1, X2, ..., Y)
    df = pd.read_csv(path, skiprows=1)

    return df


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    logger.info("Preprocessing dataset...")

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    X = df.drop(columns=["default payment next month"])
    y = df["default payment next month"]

    return X, y


# =============================================================================
# TRAINING
# =============================================================================

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
    logger.info("Training Random Forest model...")
    logger.info(f"Training samples: {len(X_train)}")

    param_dist = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"]
    }

    rf = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True
    )

    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1
    )

    search.fit(X_train, y_train)

    logger.info(f"Best Parameters: {search.best_params_}")
    logger.info(f"OOB Score: {search.best_estimator_.oob_score_:.4f}")

    return search.best_estimator_


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(model: RandomForestClassifier,
             X_test: pd.DataFrame,
             y_test: pd.Series) -> Metrics:

    logger.info("Evaluating model...")
    logger.info(f"Testing samples: {len(X_test)}")

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = Metrics(
        accuracy=accuracy_score(y_test, preds),
        precision=precision_score(y_test, preds, zero_division=0),
        recall=recall_score(y_test, preds, zero_division=0),
        f1=f1_score(y_test, preds, zero_division=0),
        roc_auc=roc_auc_score(y_test, probs)
    )

    logger.info(f"Accuracy  : {metrics.accuracy:.4f}")
    logger.info(f"Precision : {metrics.precision:.4f}")
    logger.info(f"Recall    : {metrics.recall:.4f}")
    logger.info(f"F1 Score  : {metrics.f1:.4f}")
    logger.info(f"ROC AUC   : {metrics.roc_auc:.4f}")

    generate_graphs(model, X_test, y_test, probs)

    return metrics


# =============================================================================
# VISUALISATION
# =============================================================================

def generate_graphs(model: RandomForestClassifier,
                    X_test: pd.DataFrame,
                    y_test: pd.Series,
                    probs: np.ndarray) -> None:

    # Confusion Matrix
    cm = confusion_matrix(y_test, model.predict(X_test))
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "roc_curve.png"))
    plt.close()

    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), importances[indices])
    plt.xticks(range(len(indices)), X_test.columns[indices], rotation=45)
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, "feature_importance.png"))
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    df = load_data(DATA_PATH)
    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    logger.info(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    main()