"""Restore and verify a backed-up canonical model artifact."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import MODEL_PATH


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--backup', required=True, type=Path)
    args = parser.parse_args()

    backup_path = args.backup.resolve()
    checksum_path = backup_path.parent / 'SHA256SUMS.txt'
    if not backup_path.exists():
        raise FileNotFoundError(f'No existe el backup: {backup_path}')
    if not checksum_path.exists():
        raise FileNotFoundError(f'No existe el checksum: {checksum_path}')

    expected = checksum_path.read_text(encoding='utf-8').split()[0].lower()
    actual = sha256(backup_path)
    if actual.lower() != expected:
        raise RuntimeError('Rollback rechazado: el checksum del backup no coincide.')

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(backup_path, MODEL_PATH)
    if sha256(MODEL_PATH) != actual:
        raise RuntimeError('Rollback falló al verificar el artefacto restaurado.')

    print(f'Rollback aplicado: {backup_path}')
    print(f'SHA-256 verificado: {actual}')


if __name__ == '__main__':
    main()
