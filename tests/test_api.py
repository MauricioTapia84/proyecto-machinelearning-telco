from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get('/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'


def test_root_dashboard_has_no_customer_id_field() -> None:
    response = client.get('/')
    assert response.status_code == 200
    html = response.text
    assert 'name="customer_id"' not in html
    assert 'id="gauge"' in html
    assert 'id="form"' in html
    assert 'Desempeño histórico del modelo' in html
    assert 'data-example="stable"' in html
    assert 'data-example="at-risk"' in html
    assert 'data-example="new"' in html
    assert 'data-example="premium"' in html


def test_predict_endpoint() -> None:
    payload = {
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'Yes',
        'Dependents': 'No',
        'tenure': 12,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'No',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 75.5,
        'TotalCharges': 850.0,
    }

    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    body = response.json()
    assert 'customer_id' not in body
    assert 'predicted_churn' in body
    assert 'risk_score' in body
    assert 'probability_churn' in body
    assert 'probability_percent' in body
    assert body['probability_method'] == 'sigmoid_approximation'
    assert 'risk_label' in body
    assert 'metrics_context' in body
    assert body['metrics_context']['recall_focus'] == 'prioridad de detección de churn'
    assert 'precision' in body['metrics_context']
    assert 'f1_churn' in body['metrics_context']
    assert 'pr_auc' in body['metrics_context']
