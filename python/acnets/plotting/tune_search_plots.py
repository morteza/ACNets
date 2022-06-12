import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ray.tune.sklearn import TuneSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from ..pipeline import ConnectivityPipeline


def _plot_chance_scores(X, y, cv):
    _chance_pipe = Pipeline([
        ('connectivity', ConnectivityPipeline(mock=True)),
        ('clf', DummyClassifier(strategy='stratified'))
    ])

    _chance_scores = cross_val_score(_chance_pipe, X, y, cv=cv, n_jobs=-1)
    # TODO plot shaded chance region again


def plot_scores(opt: TuneSearchCV, X=None, y=None, cv=None, with_chance=False, show=True):
    sns.set_style('ticks')
    sns.despine()

    if cv is None:
        cv = opt.cv

    n_splits = opt.cv if isinstance(opt.cv, int) else opt.cv.get_n_splits()
    n_folds = opt.cv if isinstance(opt.cv, int) else int(1/opt.cv.test_size)

    # prepare tuning scores dataframe
    scores = pd.DataFrame(opt.cv_results_)
    split_score_cols = [c for c in scores.columns if 'split' in c]
    scores['cv_test_score'] = scores[split_score_cols].apply(lambda x: list(x), axis=1)
    scores = scores.drop(columns=split_score_cols) # + ['params'])
    scores['label'] = scores['params'].apply(lambda p: str(list(p.values())))

    scores = scores.explode('cv_test_score').reset_index(drop=True)
    scores = scores.sort_values('rank_test_score', ascending=True)

    # DEBUG print(test_results.groupby('parcellation').min())

    if with_chance and (X is not None) and (y is not None):
        _plot_chance_scores(opt.X, opt.y, opt.cv)

    _, ax = plt.subplots(figsize=(scores['rank_test_score'].nunique() * 1.1, 5))

    sns.lineplot(data=scores.iloc[::-1], x='label', y='cv_test_score',
                 # ci='sd',
                 lw=3, sort=False, ax=ax)

    sns.scatterplot(data=scores.iloc[::-1],
                    x='label', y='mean_test_score',
                    marker='o', s=100,
                    ax=ax)

    ax.legend(['average', '95% CI'],  # '_average', 'chance'],
              title_fontproperties={'weight': 'bold', 'size': 'x-large'},
              prop={'size': 'xx-large'},
              title=f'{n_splits} $\\times$ {n_folds}-fold CV')
    ax.get_legend()._legend_box.align = 'left'
    ax.set_xlabel('')
    ax.set_ylabel(opt.scoring, fontsize='xx-large')
    plt.xticks(rotation=45, ha='right', fontsize='x-large', rotation_mode='anchor')
    plt.suptitle(f'classification {opt.scoring} (validation set)',
                 fontsize='xx-large', y=.95)
    plt.grid(axis='y')

    # table = plt.table(cellText=[['s','s2','s3','s4','s5'],['s','s2','s','s','s']],
    #                   rowLabels=['atlas','connectivity'],
    #                   rowColours=['lightgreen','gray'],
    #                 #   colLabels=['1','2'],
    #                 colLoc=['center','center'],
    #                   loc='bottom')

    if show:
        plt.show()
