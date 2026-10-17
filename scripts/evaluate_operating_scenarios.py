"""Evaluate illustrative capacity and cost scenarios without using test labels to select."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.decision import calculate_policy_economics, select_operating_threshold
from src.inference import load_model_artifact
from src.modeling import RANDOM_STATE, generate_oof_scores
from src.paths import (
    DECISION_SCENARIOS_PATH,
    MODEL_PATH,
    RAW_DATA_PATH,
    RESULTS_DIR,
)
from src.pipeline import TARGET, prepare_dataset

ELIGIBLE_COHORT_SIZE = 7043
CONTACT_CAPACITY = 250
RETENTION_EFFECTIVENESS = 0.20
CHURN_HORIZON_MONTHS = 24
CONTACT_COST_RATIOS = (0.01, 0.05, 0.10)


def main() -> None:
    raw_data = pd.read_csv(RAW_DATA_PATH)
    data = prepare_dataset(raw_data)
    X = data.drop(columns=[TARGET])
    y = data[TARGET].map({"No": 0, "Yes": 1}).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    artifact = load_model_artifact(MODEL_PATH)
    metadata = artifact["metadata"]
    pipeline = artifact["pipeline"]
    classifier = pipeline.named_steps["classifier"]
    if metadata.get("score_method") != "decision_function":
        raise ValueError("El reporte espera el score de decisión del SVM.")

    oof_scores = generate_oof_scores(
        X_train,
        y_train,
        classifier,
        representation=metadata["representation"],
        use_smote=metadata["uses_smote"],
        random_state=RANDOM_STATE,
    )
    test_scores = np.asarray(pipeline.decision_function(X_test), dtype=float)

    median_monthly_charge = float(
        pd.to_numeric(data["MonthlyCharges"], errors="raise").median()
    )
    capacity_rate = CONTACT_CAPACITY / ELIGIBLE_COHORT_SIZE
    oof_capacity = max(1, round(len(X_train) * capacity_rate))
    test_capacity = max(1, round(len(X_test) * capacity_rate))
    churn_loss = median_monthly_charge * CHURN_HORIZON_MONTHS
    y_test_values = y_test.to_numpy()
    records: list[dict[str, float | int | bool]] = []

    for contact_cost_ratio in CONTACT_COST_RATIOS:
        contact_cost = median_monthly_charge * contact_cost_ratio
        decision = select_operating_threshold(
            y_train,
            oof_scores,
            capacity=oof_capacity,
            contact_cost=contact_cost,
            churn_loss=churn_loss,
            retention_effectiveness=RETENTION_EFFECTIVENESS,
            retained_value=0,
        )
        oof_predictions = (oof_scores >= float(decision["threshold"])).astype(int)
        oof_economics = calculate_policy_economics(
            y_train,
            oof_predictions,
            contact_cost=contact_cost,
            churn_loss=churn_loss,
            retention_effectiveness=RETENTION_EFFECTIVENESS,
            retained_value=0,
        )
        eligible = np.flatnonzero(test_scores >= float(decision["threshold"]))
        if len(eligible) > test_capacity:
            top_capacity = np.argsort(test_scores[eligible])[-test_capacity:]
            selected = eligible[top_capacity]
        else:
            selected = eligible

        test_predictions = np.zeros(len(X_test), dtype=int)
        test_predictions[selected] = 1
        test_economics = calculate_policy_economics(
            y_test_values,
            test_predictions,
            contact_cost=contact_cost,
            churn_loss=churn_loss,
            retention_effectiveness=RETENTION_EFFECTIVENESS,
            retained_value=0,
        )
        tn = int(test_economics["true_negative"])
        fp = int(test_economics["false_positive"])
        fn = int(test_economics["false_negative"])
        tp = int(test_economics["true_positive"])
        records.append({
            "cohort_size_assumed": ELIGIBLE_COHORT_SIZE,
            "capacity_per_cohort": CONTACT_CAPACITY,
            "capacity_rate": capacity_rate,
            "oof_capacity": oof_capacity,
            "test_capacity": test_capacity,
            "horizon_months": CHURN_HORIZON_MONTHS,
            "median_monthly_charge": median_monthly_charge,
            "churn_loss_proxy_gross_revenue": churn_loss,
            "retained_value": 0.0,
            "retention_effectiveness": RETENTION_EFFECTIVENESS,
            "contact_cost_ratio_of_median_charge": contact_cost_ratio,
            "contact_cost_proxy": contact_cost,
            "oof_threshold": float(decision["threshold"]),
            "oof_contacts": int(decision["contacts"]),
            "oof_recall": float(decision["recall"]),
            "oof_precision": float(decision["precision"]),
            "oof_expected_cost_proxy": float(decision["expected_cost"]),
            "oof_expected_loss_without_campaign_proxy": oof_economics["baseline_loss"],
            "oof_contact_cost_proxy": oof_economics["contact_cost_total"],
            "oof_gross_loss_avoided_proxy": oof_economics["gross_loss_avoided"],
            "oof_residual_churn_loss_proxy": oof_economics["residual_churn_loss"],
            "oof_net_savings_vs_no_contact_proxy": oof_economics[
                "net_savings_vs_no_contact"
            ],
            "test_contacts": int(test_predictions.sum()),
            "test_true_positive": int(tp),
            "test_false_positive": int(fp),
            "test_false_negative": int(fn),
            "test_recall": float(tp / (tp + fn)) if tp + fn else 0.0,
            "test_precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "test_expected_cost_proxy": test_economics["expected_net_cost"],
            "test_expected_loss_without_campaign_proxy": test_economics["baseline_loss"],
            "test_contact_cost_proxy": test_economics["contact_cost_total"],
            "test_gross_loss_avoided_proxy": test_economics["gross_loss_avoided"],
            "test_residual_churn_loss_proxy": test_economics["residual_churn_loss"],
            "test_net_savings_vs_no_contact_proxy": test_economics[
                "net_savings_vs_no_contact"
            ],
            "test_capacity_respected": int(test_predictions.sum()) <= test_capacity,
        })

    results = pd.DataFrame.from_records(records)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(DECISION_SCENARIOS_PATH, index=False)
    display_columns = [
        "contact_cost_ratio_of_median_charge",
        "oof_threshold",
        "oof_contacts",
        "oof_recall",
        "test_contacts",
        "test_capacity",
        "test_recall",
        "test_precision",
        "test_expected_loss_without_campaign_proxy",
        "test_contact_cost_proxy",
        "test_gross_loss_avoided_proxy",
        "test_residual_churn_loss_proxy",
        "test_expected_cost_proxy",
        "test_net_savings_vs_no_contact_proxy",
    ]
    print("Escenarios hipotéticos; no constituyen una política aprobada.")
    print(
        "MonthlyCharges se usa como proxy de ingreso bruto durante 24 meses; "
        "no es margen ni utilidad. retained_value=0 evita doble conteo."
    )
    print(results[display_columns].round(4).to_string(index=False))
    print(f"Resultados guardados en: {DECISION_SCENARIOS_PATH}")


if __name__ == "__main__":
    main()
