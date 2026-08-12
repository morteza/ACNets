from .utils import ProgressBar, ProgressBarCallback


def test_init_progress_bar():
    p1 = ProgressBar(n_epochs=10, n_runs=10, run_index=1)
    assert p1.n_epochs == 10
    assert p1.pbar.total == 10
    assert p1.pbar.desc == 'run 01/10: '
    assert p1.pbar.n == 0

    p2 = ProgressBar(n_epochs=20, n_runs=15, run_index=1, reusable_pbar=p1.pbar)
    assert p2.n_epochs == 20
    assert p2.pbar.total == 20
    assert p2.pbar.desc == 'run 01/15: '
    assert p2.pbar.n == 0


def test_keras_era_alias_still_resolves():
    assert ProgressBarCallback is ProgressBar


def test_advances_one_step_per_epoch():
    with ProgressBar(n_epochs=3, n_runs=1, run_index=1) as bar:
        for epoch in range(3):
            bar.on_epoch_end(epoch, {'loss': 1.0 / (epoch + 1)})
            assert bar.pbar.n == epoch + 1
    # the bar resets on exit so it can be handed to the next run
    assert bar.pbar.n == 0
