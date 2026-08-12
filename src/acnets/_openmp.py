"""Force XGBoost to load its OpenMP runtime before PyTorch loads a different one.

On macOS, `import torch` followed later by an XGBoost `fit` segfaults the
process. Both libraries ship their own OpenMP runtime -- torch vendors
`libomp.dylib`, XGBoost links another copy -- and whichever loads second finds
the symbols already bound. XGBoost is the one that crashes, and it crashes
inside native code, so there is no Python traceback: the process simply dies
with SIGSEGV and whatever was buffered on stdout is lost.

Import order fixes it, and only import order. Verified both ways:

    import torch; from xgboost import XGBClassifier; XGBClassifier().fit(x, y)
    -> SIGSEGV (exit 139)

    from xgboost import XGBClassifier; import torch; XGBClassifier().fit(x, y)
    -> fine

This was latent for a long time because `runner._make_classifier` imports both
lazily, and the multiverse drivers happen to iterate classifiers in the order
`svm, xgboost, mlp` -- XGBoost first, by luck rather than design. Any script
that imports a torch-backed module at the top and then fits an XGBoost model
hits it, which is what `scripts/run_normative.py` did.

`preload()` is called from the two places torch can enter the package, so the
ordering no longer depends on which specification a worker happens to run first.
"""

from __future__ import annotations


def preload() -> bool:
    """Import xgboost if it is installed. Returns whether it was imported.

    Safe to call repeatedly and safe when xgboost is absent -- it is an optional
    dependency for the `mlp`-only paths.
    """
    try:
        import xgboost  # noqa: F401
    except ImportError:
        return False
    return True
