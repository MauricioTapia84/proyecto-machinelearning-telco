import builtins
import importlib
import sys


def test_pipeline_imports_without_imblearn(monkeypatch):
    """Loading the project package should not require imbalanced-learn upfront."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith('imblearn'):
            raise ModuleNotFoundError('simulated missing imblearn')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', fake_import)

    for module_name in ['src.pipeline', 'src.modeling']:
        sys.modules.pop(module_name, None)

    module = importlib.import_module('src.pipeline')
    assert hasattr(module, 'build_pipeline')
    assert hasattr(module, 'build_evaluation_pipeline')
