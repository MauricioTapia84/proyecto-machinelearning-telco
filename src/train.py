"""Train and persist the Telco churn pipeline."""

from pathlib import Path
import sys

import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.pipeline import TARGET, build_pipeline, prepare_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'teleco_clean.csv'
MODEL_PATH = PROJECT_ROOT / 'models' / 'telco_churn_pipeline.joblib'


def main() -> None:
    data = prepare_dataset(pd.read_csv(DATA_PATH))
    X = data.drop(columns=[TARGET])
    y = data[TARGET].map({'No': 0, 'Yes': 1}).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model_pipeline = build_pipeline(X_train)
    model_pipeline.fit(X_train, y_train)

    predictions = model_pipeline.predict(X_test)
    probabilities = model_pipeline.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, predictions, target_names=['No Churn', 'Churn']))
    print(f'ROC AUC: {roc_auc_score(y_test, probabilities):.4f}')

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, MODEL_PATH)
    print(f'Modelo guardado en: {MODEL_PATH}')


if __name__ == '__main__':
    main()
