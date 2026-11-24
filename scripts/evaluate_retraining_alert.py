"""Convert drift evidence into an auditable retraining decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import RESULTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='output/resultados/monitorizacion_drift.csv')
    parser.add_argument('--psi-alert', type=float, default=0.20)
    parser.add_argument('--consecutive-alerts', type=int, default=2)
    parser.add_argument('--history', default='output/resultados/historial_alertas_drift.csv')
    args = parser.parse_args()

    report_path = Path(args.report)
    history_path = Path(args.history)
    if not report_path.exists():
        raise FileNotFoundError(f'No existe el reporte de deriva: {report_path}')

    report = pd.read_csv(report_path)
    if 'status' not in report or 'psi' not in report:
        raise ValueError('El reporte debe incluir columnas status y psi.')

    alerts = report.loc[report['psi'] >= args.psi_alert]
    has_alert = len(alerts) > 0

    history_rows = []
    if history_path.exists():
        history = pd.read_csv(history_path)
        history_rows = history.to_dict('records')
    history_rows.append({'periodo': len(history_rows) + 1, 'alerta': int(has_alert)})

    history_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    recent = history_rows[-args.consecutive_alerts:] if args.consecutive_alerts > 0 else []
    consecutive = len(recent) >= args.consecutive_alerts and all(row['alerta'] == 1 for row in recent)

    decision = {
        'action': 'retraining_review_required' if consecutive else 'continue_monitoring',
        'reason': 'Dos o más periodos consecutivos superaron el PSI.' if consecutive else 'Sin evidencia consecutiva suficiente.',
        'psi_threshold': args.psi_alert,
        'variables_alertadas': alerts['variable'].tolist() if 'variable' in report else [],
        'automatic_retraining': False,
        'approval_required': True,
        'consecutive_alert_policy': args.consecutive_alerts,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / 'decision_reentrenamiento.json'
    output.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    print(f'Decisión guardada en: {output}')


if __name__ == '__main__':
    main()
