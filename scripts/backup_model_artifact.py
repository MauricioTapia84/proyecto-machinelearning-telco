"""Create an integrity-checked backup of the canonical model artifact."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MODEL_PATH, MODELS_DIR


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='manual')
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'No existe el artefacto canónico: {MODEL_PATH}')

    stamp = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')
    backup_dir = MODELS_DIR / 'backups' / f'{stamp}_{args.label}'
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / MODEL_PATH.name

    shutil.copy2(MODEL_PATH, backup_path)
    backup_hash = sha256(backup_path)
    (backup_dir / 'SHA256SUMS.txt').write_text(
        f'{backup_hash}  {MODEL_PATH.name}\n', encoding='utf-8'
    )
    print(f'Backup creado: {backup_path}')
    print(f'SHA-256: {backup_hash}')


if __name__ == '__main__':
    main()
