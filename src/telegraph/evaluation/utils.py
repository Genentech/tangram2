import gzip
import os.path as osp
from functools import reduce
from typing import Any, Dict, List, Tuple, TypeVar

import anndata as ad
import numpy as np
import pandas as pd
from numba import njit
from scipy.sparse import spmatrix

W = TypeVar("W")


def ifnonereturn(obj, return_object: None):
    if obj is None:
        return return_object
    return obj


def read_input_object(
    path: str,
    return_array: bool = False,
    layer=None,
    return_df: bool = True,
    adata_key: str = None,
    **kwargs,
):
    # function to read a single file of multiple different types
    if path is None:
        obj = None
    else:
        # if anndata object
        if path.endswith(".h5ad"):
            # read
            obj = ad.read_h5ad(path)
            # adjust for layer
            if adata_key is not None:
                obj = get_ad_value(obj, key=adata_key)

            elif layer is not None:
                obj.X = obj.layers[layer]

            # return np.ndarry if specified
            if return_array:
                if hasattr(obj, "X"):
                    obj = obj.X
                # check if sparse
                if isinstance(obj, spmatrix):
                    obj = obj.todense()
            elif return_df:
                if hasattr(obj, "to_df"):
                    obj = obj.to_df()
            else:
                pass

        # if path to csv or tsv file
        elif path.endswith((".csv", ".tsv")):
            obj = pd.read_csv(path, header=0, index_col=0)
            # return np.ndarray if specified
            if return_array:
                obj = obj.values

        # if path to numpy object
        elif path.endswith(".npy"):
            obj = np.load(path)

        else:
            raise NotImplementedError

    return obj


def read_data(data_dict: Dict[str, str]) -> Dict[str, Any]:
    # read files specified in a dictionary (data_dict)

    # instantiate dictionary to hold read objects
    input_dict = dict()
    # map to rename objects when necessary
    rename_map = dict(sp="X_to", sc="X_from")
    rename_map["map"] = "T"

    # iterate over objects in dictionary
    for name in data_dict.keys():
        # get path
        pth = data_dict[name]["path"]
        # get layer
        layer = data_dict[name].get("layer", None)
        # check if should be returned as np.ndarray
        return_array = data_dict[name].get("asarray", False)
        # if return as data frame
        return_df = data_dict[name].get("asdf", False)
        # get object
        obj = read_input_object(
            pth, return_array=return_array, layer=layer, return_df=return_df
        )

        # add option to transpose
        transpose = data_dict[name].get("transpose", False)
        # transpose if specified
        if transpose:
            obj = obj.T

        # remove all nan spots
        if name in (["sp", "X_to"]):
            if (hasattr(obj, "obsm")) and ("spatial" in obj.obsm):
                keep = ~(np.any(np.isnan(obj.obsm["spatial"]), axis=1))
                obj = obj[keep, :].copy()
            elif hasattr(obj, "obs"):
                x = data_dict[name].get("x_coords", None)
                y = data_dict[name].get("y_coords", None)
                if x is None or y is None:
                    raise RuntimeError(
                        "Could not find x_coords/y_coords in obs. Please provide coordinate column names in config file."
                    )
                else:
                    keep = obj.obs[[x, y]].dropna(axis=0).index
                    obj = obj[keep, :].copy()
            else:
                pass

        # store object
        input_dict[name] = obj

    # correct names
    for old_name, new_name in rename_map.items():
        if old_name in input_dict:
            input_dict[new_name] = input_dict.pop(old_name)

    return input_dict
