import anndata as ad
import numpy as np
from scipy.sparse import spmatrix
from scipy.stats import chi2


def _SMAI_test(
    X1: np.ndarray,
    X2: np.ndarray,
    K: int | None = None,
    cutoff_t: float = 1.01,
    EPS: float = 1e-7,
):

    # based on: https://github.com/rongstat/SMAI/blob/master/R/SMAI-fun2.R

    # leaving this in-case we want to make
    # manupulations to X1,X2
    _X1, _X2 = X1, X2

    X1_aligned = _X1 - np.mean(_X1, axis=1, keepdims=True)
    X2_aligned = _X2 - np.mean(_X2, axis=1, keepdims=True)
    p, n = X1_aligned.shape

    # get eigenvalues using svd
    u1, s1, vh1 = np.linalg.svd(X1_aligned, full_matrices=False)
    lam1 = (s1**2) / n

    u2, s2, vh2 = np.linalg.svd(X2_aligned, full_matrices=False)
    lam2 = (s2**2) / n

    # Adjust lam1
    lam1 = lam1 * np.sum(lam2[:K]) / np.sum(lam1[:K])

    # get K if not provided (number of relevant eigenvalues)
    if K is None:

        if min(lam1) < 0.001:
            r0 = min(min(np.where(lam1 < 0.001)[0]) - 1, 30)
        else:
            r0 = min(n, p)
        if sum(lam1[0 : (r0 - 1)] / lam1[1:r0] > cutoff_t) > 0:
            r1 = (
                max(np.where(lam1[0 : (r0 - 1)] / lam1[1:r0] > cutoff_t)[0]) + 1
            )  # +1 due to zero indexing
        else:
            r1 = r0
            print("Warning: smaller cutoff_t recommended!")

        if min(lam2) < 0.001:
            r0 = min(min(np.where(lam2 < 0.001)[0]) - 1, 30)
        else:
            r0 = min(n, p)

        if sum(lam2[0 : (r0 - 1)] / lam2[1:r0] > cutoff_t) > 0:
            r2 = (
                max(np.where(lam2[0 : (r0 - 1)] / lam2[1:r0] > cutoff_t)[0]) + 1
            )  # +1 due to zero indexing
        else:
            r2 = r0
            print("Warning: smaller cutoff_t recommended!")

        K = min(r1, r2)

    K = 2 if K < 2 else K

    # Define m_lb and m_p helper functions
    def m_lb(z, lam, k, gamma):
        return -(1 - gamma) / z + np.sum(1 / (lam[k + 1 :] - z + EPS)) / len(lam)

    def m_p(z, lam, k, gamma):
        return -(1 - gamma) / z**2 + np.sum(1 / (lam[k + 1 :] - z + EPS) ** 2) / len(
            lam
        )

    # Initialize alpha1, alpha2, psi1, psi2
    alpha1, alpha2, psi1, psi2 = [], [], [], []

    # Calculate alpha1, alpha2, psi1, psi2
    for i in range(K):
        alpha1.append(-1 / m_lb(lam1[i], lam1, i, p / n))
        alpha2.append(-1 / m_lb(lam2[i], lam2, i, p / n))
        psi1.append(1 / alpha1[i] ** 2 / m_p(lam1[i], lam1, i, p / n))
        psi2.append(1 / alpha2[i] ** 2 / m_p(lam2[i], lam2, i, p / n))

    # convert to numpy array
    alpha1 = np.array(alpha1)
    alpha2 = np.array(alpha2)
    psi1 = np.array(psi1)
    psi2 = np.array(psi2)

    # Diagonal of asymptotic covariance matrices
    as_1 = 2 * alpha1**2 * psi1
    as_2 = 2 * alpha2**2 * psi2

    # t-statistics and p-value
    t_stat = np.sqrt(n) * (lam1[0:K] - lam2[0:K]) / np.sqrt(as_1 + as_2)
    p_value = chi2.sf(np.sum(t_stat[0:K] ** 2), K)

    return p_value


def SMAI_test(
    ad_1: ad.AnnData,
    ad_2: ad.AnnData,
    use_vars: str | None = None,
    layer: str | None = None,
    obsm_key: str | None = None,
    c: float = 0.01,
):

    # find intersecting genes
    inter = ad_1.var_names.intersection(ad_2.var_names)

    if use_vars is not None:
        inter = list(set(inter).intersection(set(use_vars)))

    # extract features from anndata
    match (layer, obsm_key):
        case (str(), str()):
            raise ValueError('Only one of "layer" and "obsm_key" can be specified')
        case (None, str()):
            X1 = ad_1.obsm[obsm_key].copy()
            X2 = ad_2.obsm[obsm_key].copy()
        case (str(), None):
            X1 = ad_1.layer[layer][:, inter].copy()
            X2 = ad_2.layer[layer][:, inter].copy()
        case (None, None):
            X1 = ad_1[:, inter].X.copy()
            X2 = ad_2[:, inter].X.copy()
        case (_, _):
            raise ValueError('Incpmpatible specfication of "obsm_key" and "layer"')

    if isinstance(X1, spmatrix):
        X1 = X1.toarray()
    if isinstance(X2, spmatrix):
        X2 = X2.toarray()

    # get number of obs in each object
    n_1 = X1.shape[0]
    n_2 = X2.shape[0]

    # get minimal number of obs in each object
    n_obs = np.min((n_1, n_2))

    # subset indices, we need same number of obs
    idx_1 = np.random.choice(n_1, replace=False, size=n_obs)
    idx_2 = np.random.choice(n_2, replace=False, size=n_obs)

    # subset counts
    X1 = X1[idx_1]
    X2 = X2[idx_2]

    # conduct SMAI test
    p_val = _SMAI_test(X1, X2, cutoff_t=1 + c)

    return p_val
