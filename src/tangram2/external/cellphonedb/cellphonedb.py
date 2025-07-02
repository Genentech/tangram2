import os
import os.path as osp
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from .cpdb.src.core.methods import (
    cpdb_analysis_method,
    cpdb_statistical_analysis_method,
)


def run(
    adata: ad.AnnData,
    label_col: str,
    cpdb_file_path: str,
    counts_data: str = "hgnc_symbol",
    output_dir: str | None = None,
    layer: str | None = None,
    require_receptor: bool = False,
    require_ligand: bool = True,
    only_return_signficant: bool = True,
    method: str = "statistical",
) -> pd.DataFrame:
    """Run CellPhoneDB

    This function runs CellPhoneDB and returns a pandas DataFrame
    with the following columns ['ligand','receptor','target','source','value']

    where source and target indicate the cell type that acts as a source and target of the signal
    (ligand). The value is the CellPhoneDB mean results.

    """

    if layer is not None:
        adata.layers["_old"] = adata.X.copy()
        adata.X = adata.layers[layer].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        df = adata.obs[[label_col]].copy()
        df.columns = ["cell_type"]
        meta_path = osp.join(tmpdir, "meta.csv")
        df.to_csv(meta_path)
        os.listdir(tmpdir)

        if method == "statistical":
            cpdb_results = cpdb_statistical_analysis_method.call(
                cpdb_file_path=cpdb_file_path,
                meta_file_path=meta_path,
                counts_file_path=adata,
                counts_data=counts_data,
                output_path=output_dir if output_dir is not None else tmpdir,
            )
            cpdb = cpdb_results["means"]
            pvals = cpdb_results["pvalues"]

        else:
            cpdb_results = cpdb_analysis_method.call(
                cpdb_file_path=cpdb_file_path,
                meta_file_path=meta_path,
                counts_file_path=adata,
                counts_data=counts_data,
                output_path=output_dir if output_dir is not None else tmpdir,
            )
            cpdb = cpdb_results["significant_means_result"]
            pvals = None

        sub_col = ["gene_a", "gene_b"] + [x for x in cpdb.columns if "|" in x]
        sub_cpdb = cpdb[sub_col]

    if layer is not None:
        adata.X = adata.layers["_old"].copy()

    signal_and_label = []
    for k, row in sub_cpdb.iterrows():
        gene_a = str(row.gene_a)
        gene_b = str(row.gene_b)
        inters = row[2::]
        for inter, val in inters.items():
            source, target = inter.split("|")
            if pvals is not None:
                pval = float(pvals.loc[k, inter])
            else:
                pval = ~np.isnan(val)
            signal_and_label.append((gene_a, gene_b, target, source, val, pval))

    out = pd.DataFrame(
        signal_and_label,
        columns=["ligand", "receptor", "target", "source", "value", "pval"],
    )

    out["_uni_col"] = out["ligand"] + "_" + out["target"] + "_" + out["source"]
    out = out.drop_duplicates(subset=["_uni_col"])
    out.drop(columns=["_uni_col"], inplace=True)

    if require_ligand:
        out = out.iloc[out.ligand.values != "nan", :].copy()
    if require_receptor:
        out = out.iloc[out.receptor.values != "nan", :].copy()

    out = out.sort_values(by="value", ascending=False)

    return out
