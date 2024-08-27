import os
import os.path as os
import shutil

import anndata as ad
import numpy as np
import scanpy as sc

import telegraph as tg


def read_h5ad_uniqify(path, tag=None):
    adata = ad.read_h5ad(path)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    if tag is not None:
        adata.obs_names = [f"{tag}_{x}" for x in adata.obs_names]
    return adata


def pp_adata(ad_sc, ad_sp=None):
    is_mt = ad_sc.var_names.str.startswith("MT-")
    is_rp = ad_sc.var_names.str.startswith(("RPS", "RPL", "RP-", "RP"))
    keep_genes = (~is_mt) & (~is_rp)
    ad_sc = ad_sc[:, keep_genes].copy()

    sc.pp.filter_cells(ad_sc, min_counts=300)

    sc.pp.filter_genes(ad_sc, min_counts=10)

    if ad_sp is not None:
        sc.pp.filter_cells(ad_sp, min_counts=100)
        sc.pp.filter_genes(ad_sp, min_counts=10)

    ad_sc.layers["raw"] = ad_sc.X.copy()
    sc.pp.normalize_total(ad_sc, 1e4)
    sc.pp.log1p(ad_sc)
    sc.pp.highly_variable_genes(ad_sc, n_top_genes=5000)
    ad_sc.layers["norm"] = ad_sc.X.copy()
    ad_sc.X = ad_sc.layers["raw"].copy()

    if ad_sp is not None:
        return ad_sc, ad_sp
    return ad_sc


SC_PTH = "/gstore/data/resbioai/andera29/cci-explore/validation/pipeline/data/common/SCC/sc/P9_normal+tumor.h5ad"


adata = ad.read_h5ad(SC_PTH)

label_col = "level1_celltype"


receiver_name, signaler_name = adata.obs[label_col].value_counts().index[0:2]

ad_sp, ad_sc = tg.datagen.cellmix.cellmix.cellmix(
    adata,
    n_spots=500,
    n_cells_per_spot=10,
    n_types_per_spot=5,
    label_col=label_col,
    signaler_names=signaler_name,
    receiver_names=receiver_name,
    n_interactions=1,
    effect_direction="pos",
    signal_effect_base=0.99,
    signal_effect_scaling=3,
    p_inter=0.8,
    p_signal_spots=0.9,
)


ad_sc, ad_sp = pp_adata(ad_sc, ad_sp)


hvg_genes = ad_sc.var_names[ad_sc.var.highly_variable.values].tolist()

input_dict_1 = tg.met.utils.adatas_to_input(
    {"from": ad_sc.copy(), "to": ad_sp.copy()},  # provide the data to be used
    categorical_labels={
        "from": ["level1_celltype"]
    },  # include cluster labels in the design matrix
)

tg.met.pp.StandardTangramV1.run(input_dict_1)

map_res_1 = tg.met.map_methods.TangramV2Map.run(
    input_dict_1,
    num_epochs=1000,
    genes=hvg_genes,
    mode="hejin_workflow",
)

input_dict_1.update(map_res_1)

sc.pp.normalize_total(input_dict_1["X_from"], target_sum=1e4)
sc.pp.log1p(input_dict_1["X_from"])

inter_res = tg.dev.imod.methods.InteractionModel.run(input_dict_1, n_epochs=1000)


beta = inter_res["beta"].to_dataframe()["beta"]
