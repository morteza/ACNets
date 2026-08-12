"""Progress reporting for training runs.

Was a `keras.callbacks.Callback`. Keras is gone, and `acnets.deep.training`
has no callback system to hook into -- deliberately, since the one thing the
callback did was drive a tqdm bar, and a bar does not need a framework to
dispatch to it.

Used either as a context manager around a training run, or by handing
`on_epoch_end` to something that calls it once per epoch:

    with ProgressBar(n_epochs=100, n_runs=10, run_index=3) as bar:
        for epoch, metrics in enumerate(train()):
            bar.on_epoch_end(epoch, metrics)
"""

from tqdm.auto import tqdm


class ProgressBar:
    """A tqdm bar over epochs, reusable across runs.

    Passing `reusable_pbar` keeps one bar for a whole sweep instead of leaving
    a trail of finished bars behind, which is why the original took it.
    """

    def __init__(self,
                 n_epochs=None,
                 n_runs=None,
                 run_index=None,
                 reusable_pbar=None):

        self.n_epochs = n_epochs
        self.pbar = reusable_pbar
        if self.pbar is None:
            self.pbar = tqdm(
                total=n_epochs,
                unit='epoch',
                dynamic_ncols=True,
                leave=False)
        self.pbar.total = n_epochs
        if run_index is not None and n_runs is not None:
            self.pbar.set_description(f'run {run_index:02}/{n_runs:02}')

    def __enter__(self):
        self.on_train_begin()
        return self

    def __exit__(self, *exc_info):
        self.on_train_end()
        return False

    def on_train_begin(self, logs=None):
        self.pbar.reset()

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            self.pbar.set_postfix({k: f'{v:.4f}' if isinstance(v, float) else v
                                   for k, v in logs.items()})
        self.pbar.update(epoch - self.pbar.n + 1)

    def on_train_end(self, logs=None):
        self.pbar.reset()


# The Keras-era name. Nothing dispatches to it any more, but notebooks import it.
ProgressBarCallback = ProgressBar
