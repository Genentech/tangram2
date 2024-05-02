import os
import os.path as osp
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from cellphonedb.src.core.methods import cpdb_analysis_method


def run(
    adata: ad.AnnData,
    label_col: str,
    cpdb_file_path: str,
    counts_data: str = "hgnc_symbol",
    output_dir: str | None = None,
    layer: str | None = None,
    require_receptor: bool = False,
    require_ligand: bool = True,
) -> pd.DataFrame:
    """Run CellPhoneDB

    This function runs CellPhoneDB and returns a pandas DataFrame
    with the following columns ['ligand','receptor','target','source','value']

    where source and target indicate the cell type that acts as a source and target of the signal
    (ligand). The value is the CellPhoneDB mean results.

    """

    if layer is not None:
        adata.layers["_old"] = adata.X.copy()
        adata.X = adata[layer].copy()

    with tempfile.TemporaryDirectory() as tmpdir:
        df = adata.obs[[label_col]].copy()
        df.columns = ["cell_type"]
        meta_path = osp.join(tmpdir, "meta.csv")
        df.to_csv(meta_path)
        os.listdir(tmpdir)

        cpdb_results = cpdb_analysis_method.call(
            cpdb_file_path=cpdb_file_path,
            meta_file_path=meta_path,
            counts_file_path=adata,
            counts_data=counts_data,
            output_path=output_dir if output_dir is not None else tmpdir,
        )

    if layer is not None:
        adata.X = adata.layers["_old"].copy()

    cpdb = cpdb_results["significant_means_result"]
    sub_cpdb = cpdb[["gene_a", "gene_b"] + [x for x in cpdb.columns if "|" in x]]

    signal_and_label = []
    for k, row in sub_cpdb.iterrows():
        gene_a = str(row.gene_a)
        gene_b = str(row.gene_b)
        inters = row[2::]
        for inter, val in inters.items():
            if not np.isnan(val):
                source, target = inter.split("|")
                signal_and_label.append((gene_a, gene_b, target, source, val))

    out = pd.DataFrame(
        signal_and_label,
        columns=["ligand", "receptor", "target", "source", "value"],
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
