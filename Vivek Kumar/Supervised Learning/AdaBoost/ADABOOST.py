import warnings 
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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
import seaborn as sns
from typing import Tuple, Optional
from dataclasses import dataclass

warnings.filterwarnings('ignore')

#Global Variales
RANDOM_STATE = 42
DATASET_PATH = "SVMtarin.csv" 
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Model Persistence
MODEL_SAVE_PATH = "knn_model.pkl"

# Visualization Configuration
FIGURE_SIZE = (12, 8)
DPI = 100
STYLE = 'seaborn-v0_8-darkgrid'

@dataclass
class ModelMetrics:
    """Data class to store model evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    def __str__(self) -> str:
        """Return formatted string representation of metrics."""
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
    