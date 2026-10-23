"""
cryptonite.engine.ml_bot_detector
------------------------------------
AIML CONCEPT: Artificial Neural Network (Multi-Layer Perceptron) classifier
that predicts P(account is a bot) from the account's numeric feature vector.

A Random Forest is trained alongside as a classical-ML baseline purely for
comparison/evaluation (printed in training report) -- the ANN is the model
actually used at inference time by the alert engine, per the "AIML" and
"soft computing (neural network)" requirement of this project.
"""

import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from cryptonite.data.simulator import generate_dataset, FEATURE_COLUMNS, LABEL_COLUMN

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ANN_PATH = os.path.join(MODEL_DIR, "ann_bot_detector.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")


class BotDetector:
    def __init__(self):
        self.model = None
        self.scaler = None

    def train(self, df=None, verbose=True):
        df = df if df is not None else generate_dataset()
        X = df[FEATURE_COLUMNS].values
        y = df[LABEL_COLUMN].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.scaler = StandardScaler().fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        # --- Artificial Neural Network (the model used in production) ---
        self.model = MLPClassifier(
            hidden_layer_sizes=(16, 8),
            activation="relu",
            solver="adam",
            alpha=1e-2,
            max_iter=2000,
            random_state=42,
        )
        self.model.fit(X_train_s, y_train)
        ann_pred = self.model.predict(X_test_s)
        ann_proba = self.model.predict_proba(X_test_s)[:, 1]
        print("\nSample ANN probabilities from held-out test set:")
        for i in range(20):
            print(
                f"Actual: {y_test[i]} | "
                f"Bot probability: {ann_proba[i] * 100:.2f}%"
            )

        report = {
            "ann": self._metrics(y_test, ann_pred, ann_proba),
        }

        if verbose:
            # --- Random Forest baseline, for comparison only ---
            rf = RandomForestClassifier(n_estimators=200, random_state=42)
            rf.fit(X_train_s, y_train)
            rf_pred = rf.predict(X_test_s)
            rf_proba = rf.predict_proba(X_test_s)[:, 1]
            report["random_forest_baseline"] = self._metrics(y_test, rf_pred, rf_proba)

            print("=== Training report (25% held-out test set) ===")
            for name, m in report.items():
                print(f"\n[{name}]")
                for k, v in m.items():
                    print(f"  {k:10s}: {v:.4f}")

        return report

    @staticmethod
    def _metrics(y_true, y_pred, y_proba):
        return dict(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred),
            recall=recall_score(y_true, y_pred),
            f1=f1_score(y_true, y_pred),
            roc_auc=roc_auc_score(y_true, y_proba),
        )

    def predict_proba(self, feature_vector: np.ndarray) -> float:
        """Return P(bot) in [0,1] for a single account's raw feature vector."""
        x = self.scaler.transform(feature_vector.reshape(1, -1))
        return float(self.model.predict_proba(x)[0, 1])

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(self.model, ANN_PATH)
        joblib.dump(self.scaler, SCALER_PATH)

    def load(self):
        self.model = joblib.load(ANN_PATH)
        self.scaler = joblib.load(SCALER_PATH)
        return self

    @staticmethod
    def exists():
        return os.path.exists(ANN_PATH) and os.path.exists(SCALER_PATH)


if __name__ == "__main__":
    det = BotDetector()
    det.train()
    det.save()
    print("\nSaved ANN model to", ANN_PATH)
