

def test_init_progress_bar():

    import os
    os.environ['KERAS_BACKEND'] = 'torch'
    import keras

    from .utils import ProgressBarCallback
    
    p1 = ProgressBarCallback(n_epochs=10, n_runs=10, run_index=1)
    assert p1.n_epochs == 10
    assert p1.pbar.total == 10
    assert p1.pbar.desc == 'run 01/10: '
    assert p1.pbar.n == 0

    p2 = ProgressBarCallback(n_epochs=20, n_runs=15, run_index=1, reusable_pbar=p1.pbar)
    assert p2.n_epochs == 20
    assert p2.pbar.total == 20
    assert p2.pbar.desc == 'run 01/15: '
    assert p2.pbar.n == 0
