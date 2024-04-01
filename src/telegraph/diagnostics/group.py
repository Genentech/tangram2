from typing import Any, Dict, List, Literal

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import spmatrix
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE, Isomap

from . import _utils as ut


def _get_X_and_labels(
    X, D=None, labels=None, layer=None, obsm=None, group_col: str | None = None
):

    assert any([D is not None, labels is not None])

    if labels is not None:
        assert labels.shape[0] == X.shape[0]

    if D is not None:
        assert D.shape[0] == X.shape[0]
        assert group_col is not None
        labels = D[group_col].values
        labels = np.array([f"{group_col}_{lab}" for lab in labels])

    match (layer, obsm):
        case (None, None):
            if isinstance(X, ad.AnnData):
                if isinstance(X.X, spmatrix):
                    Xn = X.X.toarray()
                else:
                    Xn = X.X

            elif isinstance(X, pd.DataFrame):
                Xn = X.values
            elif isinstance(X, np.ndarray):
                Xn = X

        case (None, str()):
            Xn = X.obsm[obsm]

        case (str(), None):
            Xn = X.layers[layer]

    return Xn, labels


@ut.easy_input
def plot_group_separation(
    X,
    D=None,
    labels=None,
    group_col: str = None,
    group_type: Literal["discrete", "continuous"] = "discrete",
    project_method: str | List[str] = "pca",
    cmap=plt.cm.Dark2,
    plot=True,
    marker_size: int = None,
    plt_kwargs: Dict[str, Any] | None = None,
    batch_key: List[str] = None,
    normalize_X: bool = False,
):

    Xn, labels = _get_X_and_labels(X, D=D, labels=labels, group_col=group_col)

    if normalize_X:
        Xn = Xn / Xn.sum(keepdims=True, axis=1) * 1e4
        Xn = np.log1p(Xn)

    if group_type == "discrete":
        if not isinstance(labels[0], str):
            if not np.all(np.mod(labels, 1) == labels):
                raise ValueError(
                    'labels must be categorical if "group_type" is "discrete"'
                )

    elif group_type[0:4] == "cont":
        if isinstance(labels[0], str):
            raise ValueError(
                'labels must be continuous values when "group_type" is "continuous"'
            )

    else:
        raise ValueError('"group_type" must be "continuous" or "discrete"')

    _pms = dict(
        pca=PCA,
        isomap=Isomap,
        mds=MDS,
        tsne=TSNE,
        harmony=ut.harmony_helper,
    )

    if not isinstance(project_method, (tuple, list)):
        project_method = [project_method]

    project_method = [x.lower() for x in project_method if x.lower() in _pms]

    n_pms = len(project_method)

    fig, ax = plt.subplots(1, n_pms, figsize=(4 * n_pms, 4))
    if isinstance(ax, plt.Axes):
        ax = [ax]

    uni_labels = np.unique(labels)
    color_mapper = {l: k for k, l in enumerate(uni_labels)}
    plt_kwargs_default = dict(s=marker_size)

    if plt_kwargs is None:
        _plt_kwargs = {}
    else:
        _plt_kwargs = {k: v for k, v in plt_kwargs.items()}

    for k, v in plt_kwargs_default.items():
        if k not in _plt_kwargs:
            _plt_kwargs[k] = v

    for k, m in enumerate(project_method):
        if m == "harmony":
            proj = _pms[m]
            if batch_key is None:
                batch_key = D.columns.tolist()
            Xnd = proj(Xn, D, batch_key=batch_key)
        elif m != "pca":
            proj_1 = _pms["pca"](n_components=min(Xn.shape[1], 100))
            Xnd = proj_1.fit_transform(Xn)
            proj_2 = _pms[m](n_components=2)
            Xnd = proj_2.fit_transform(Xnd)
        else:
            proj = _pms[m](n_components=2)
            Xnd = proj.fit_transform(Xn)
        for lab in uni_labels:
            is_label = labels == lab
            is_label = is_label.flatten()
            ax[k].scatter(
                Xnd[is_label, 0],
                Xnd[is_label, 1],
                c=cmap(color_mapper[lab]),
                label=lab,
                **_plt_kwargs,
            )
        ax[k].set_title("Projection Method : {}".format(m))
        ax[k].legend()

    if plot:
        plt.show()
    else:
        return fig, ax


@ut.easy_input
def test_group_separation(
    X,
    D=None,
    labels=None,
    group_col: str = None,
    classifier: str = "svm",
    n_pca: int = 50,
    clf_params: None | Dict[str, Any] = None,
    print_res: bool = False,
    n_reps: int = 10,
    stratify_by_labels: bool = False,
    normalize_cmatrix: str = None,
):

    from sklearn.metrics import confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    Xn, labels = _get_X_and_labels(X, D=D, labels=labels, group_col=group_col)

    n_obs, n_var = X.shape
    n_pca = min(n_var, n_pca)

    proj = PCA(n_components=n_pca)
    Xd = proj.fit_transform(Xn)

    _clfs = dict(svm=SVC)

    _classifier = _clfs[classifier.lower()]

    if clf_params is None:
        clf_params = {}

    f1_score_train = list()
    f1_score_test = list()

    stratify = None
    if stratify_by_labels:
        stratify = labels

    for ii in range(n_reps):
        X_train, X_test, y_train, y_test = train_test_split(
            Xd, labels, test_size=0.2, stratify=stratify
        )

        clf = _classifier(**clf_params)
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        f1_score_train.append(f1_score(y_train, y_train_pred, average="macro"))
        f1_score_test.append(f1_score(y_test, y_test_pred, average="macro"))

    f1_mean_train = np.mean(f1_score_train)
    f1_mean_test = np.mean(f1_score_test)

    f1_std_train = np.std(f1_score_train)
    f1_std_test = np.std(f1_score_test)

    cfm_train = confusion_matrix(y_train, y_train_pred, normalize=normalize_cmatrix)
    cfm_test = confusion_matrix(y_test, y_test_pred, normalize=normalize_cmatrix)

    if print_res:
        print(
            "Mean Macro F1 score [TRAIN]: {} +/- {}".format(
                f1_mean_train, 2 * f1_std_train
            )
        )
        print(
            "Mean Macro F1 score [TEST]: {} +/- {}".format(
                f1_mean_test, 2 * f1_std_test
            )
        )

    return dict(
        f1_train=f1_score_train,
        f1_test=f1_score_test,
        confusion_matrix_train=cfm_train,
        confusion_matrix_test=cfm_test,
    )


def plot_separation_confusion_matrix(
    confusion_matrix: Dict[str, Any] | np.ndarray,
    plot: bool = True,
):
    from sklearn.metrics import ConfusionMatrixDisplay

    if isinstance(confusion_matrix, dict):
        cfm_train = confusion_matrix.get("confusion_matrix_train")
        cfm_test = confusion_matrix.get("confusion_matrix_test")
        cfms = dict(train=cfm_train, test=cfm_test)
    elif isinstance(confusion_matrix, np.ndarray):
        cfms = {"": confusion_matrix}

    n_cfs = len(cfms)
    fig, ax = plt.subplots(1, n_cfs, figsize=(4 * n_cfs, 4))
    if isinstance(ax, plt.Axes):
        ax = [ax]

    for k, (name, cf) in enumerate(cfms.items()):

        cm = ConfusionMatrixDisplay(cf)
        cm.plot(ax=ax[k])
        ax[k].set_title("Confusion Matrix : {}".format(name))

    fig.tight_layout()
    if plot:
        plt.show()
    else:
        return fig, ax
