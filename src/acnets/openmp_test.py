"""Regression test for the XGBoost/PyTorch OpenMP segfault.

Importing torch and *then* fitting an XGBoost model kills the process with
SIGSEGV on macOS -- no traceback, no output, just exit 139. It is invisible to
an ordinary test because the interpreter dies, so this runs the sequence in a
subprocess and checks the exit status.

The bug was latent for months: `runner._make_classifier` imports both lazily and
the drivers happened to iterate `svm, xgboost, mlp`, putting XGBoost first by
luck. `scripts/run_normative.py` imported a torch-backed module at the top and
hit it immediately.
"""

import subprocess
import sys

import pytest


def run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, '-c', code], capture_output=True,
                          text=True, timeout=300, check=False)


class TestOpenMPPreload:

    def test_importing_acnets_deep_then_fitting_xgboost_survives(self):
        """The real-world sequence: a torch-backed import, then a tree fit."""
        result = run(
            'import acnets.deep\n'
            'import numpy as np\n'
            'from acnets.multiverse.runner import _make_classifier\n'
            'x = np.random.rand(40, 10); y = (np.random.rand(40) > 0.5).astype(int)\n'
            '_make_classifier("xgboost", 0).fit(x, y)\n'
            'print("ok")\n')
        assert result.returncode == 0, (
            f'exit {result.returncode} (139 = SIGSEGV): {result.stderr[-400:]}')
        assert 'ok' in result.stdout

    def test_make_classifier_mlp_then_xgboost_survives(self):
        """Constructing the torch classifier first must not poison xgboost."""
        result = run(
            'import numpy as np\n'
            'from acnets.multiverse.runner import _make_classifier\n'
            'x = np.random.rand(40, 10); y = (np.random.rand(40) > 0.5).astype(int)\n'
            '_make_classifier("mlp", 0)\n'
            '_make_classifier("xgboost", 0).fit(x, y)\n'
            'print("ok")\n')
        assert result.returncode == 0, (
            f'exit {result.returncode} (139 = SIGSEGV): {result.stderr[-400:]}')

    def test_preload_reports_whether_xgboost_is_available(self):
        from acnets._openmp import preload

        pytest.importorskip('xgboost')
        assert preload() is True
        assert preload() is True          # idempotent
