from functools import reduce

import numpy as np
import pandas as pd
from scipy.stats import beta, cauchy

from telegraph.methods.dea_methods import DEA


def _collate_results(*dea_res):

    if len(dea_res) == 1:
        dea_res = dea_res[0]

    if isinstance(dea_res[0], dict):
        assert "DEA" in all(("DEA" in d for d in dea_res))
        dea_res = [d["DEA"] for d in dea_res]

    assert all((isinstance(d, pd.DataFrame) for d in dea_res))

    n_tests = len(dea_res)

    uni_names = list(
        set([feature_name for p in dea_res for feature_name in p.index.tolist()])
    )

    all_df = pd.DataFrame(
        -1 * np.ones((len(uni_names), n_tests), dtype=np.float32),
        index=uni_names,
        columns=[f"test_{k}" for k in range(n_tests)],
    )

    for k, d_i in enumerate(dea_res):
        names = d_i.index
        col_name = all_df.columns[k]
        all_df.loc[names, col_name] = (
            d_i[DEA.adj_p_value.value].values.flatten().astype(np.float32)
        )

    return all_df


def _CCT(pvals_raw, weights=None):

    # pvals_raw : [n_tests] x [1]
    #    pvalues that are not present in this test are indicated with -1
    # weights: [n_tests] x [1]
    #   will be set to 1/n_tests if no values are given (uniform weight)

    # - independence: assumes that the p-values being combined are independent across tests; but is also robust to some degree of violation to this
    # - uniform distribution under null hypothesis: assumes that each p-value is uniformly distributed between 0 and 1 under the null hypothesis, standard for p-values.
    # - robustness to heterogeneity: assumes robustness to heterogeneity among studies, meaning it can effectively combine p-values from studies

    # based on : https://rdrr.io/github/xihaoli/STAAR/src/R/CCT.R

    is_not_nan = pvals_raw != -1
    pvals = pvals_raw[is_not_nan]

    if weights is None:
        weights = np.ones(len(pvals)) / len(pvals)
    else:
        weights = weights[is_not_nan]
        weights = weights / np.sum(weights)

    np.testing.assert_almost_equal(np.sum(weights), 1)

    is_small = pvals < 1e-16
    # Transform pvals into weighted Cauchy-distributed statistic and aggregate result
    if np.sum(is_small) == 0:
        cct_stat = np.sum(weights * np.tan((0.5 - pvals) * np.pi))
    else:
        cct_stat = np.sum(weights[is_small] / pvals[is_small] / np.pi)
        cct_stat += np.sum(
            weights[~is_small] * np.tan((0.5 - pvals[~is_small]) * np.pi)
        )
    # Compute aggregated statistic
    if cct_stat > 1e15:
        pval = 1 / cct_stat / np.pi
    else:
        pval = cauchy.sf(cct_stat)

    return pval


def CCT(*dea_res, weights=None):

    all_df = _collate_results(*dea_res)
    n_feat, n_tests = all_df.shape

    if weights is not None:
        assert n_tests == len(weights), "One weight per test has to be given"

    agg_df = pd.DataFrame(
        np.apply_along_axis(_CCT, 1, all_df.values),
        index=all_df.index,
        columns=[DEA.agg_p_value.value],
    )

    return agg_df


def _RRA(pvals_raw, **kwargs):

    # pvals_raw : [n_features] : [1]
    #    pvalues that are not present in this test are indicated with -1

    # get rank matrix [n_features] x [n_tests]
    # low p-values are good, high bad
    r = 1 - pvals_raw
    # normalize ranks within test
    r = r / r.max()
    # sort ranks within test
    r = np.sort(r)

    # container for p-values
    p = np.ones(len(r))
    # get indicator for tests where feature was present
    is_not_nan = r != -1
    # get ranks
    r_nz = r[is_not_nan]
    # get length of observations
    n = len(r)
    # return if only one obs
    if n == 1:
        return r
    # get parameters for beta-dsitribution
    prm_a = np.arange(1, n + 1, dtype=np.float32)
    prm_b = n - prm_a + 1
    # compute pvals
    p_nz = beta.sf(r_nz, prm_a, prm_b)
    # fill in pval container
    # multiply with n for MHT correction
    p[is_not_nan] = p_nz * n
    # clip values
    p = np.clip(p, a_min=0, a_max=1)

    return p


def RRA(*dea_res, weights=None, **kwargs):

    # Assumptions on the data:

    # - list Independence: Assumes that all input ranked lists are independent
    # - non-uniform significant item distribution: assumes significant items are not uniformly distributed across ranks but  consistently ranks higher than would occur by chance.
    # - rank-based significance: ranks are more important than values
    # - robust to list variability: capable of handling lists of varying lengths and compositions, ensuring robustness in results.
    # - sensitivity to consistency: designed to identify items that consistently rank highly across multiple lists
    # - tolerance to noise: assumes the presence of noise and outliers in the data

    # Robust Rank Aggregation: https://cran.r-project.org/web/packages/RobustRankAggreg/index.html

    # collate results from tests
    # non present is indicated with -1
    all_df = _collate_results(*dea_res)
    # return aggregated values

    agg_df = pd.DataFrame(
        np.min(np.apply_along_axis(_RRA, axis=0, arr=all_df.values), axis=1),
        index=all_df.index,
        columns=[DEA.agg_p_value.value],
    )

    return agg_df
