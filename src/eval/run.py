import os
import os.path as osp
from typing import Any, Dict

import anndata as ad

from . import constants as C
from . import utils as ut
from . import workflows as wf


def run(
    config: Dict[str, Any],
    root_dir: str = ".",
    save_mode: bool = False,
    use_fuzzy_match: bool = False,
    verbose: bool = False,
):
    # run whole validation workflow

    argmap = dict(
        sc="X_from",
        sp="X_to",
    )

    # get data info
    data = config.pop("data")
    # get list of experiment names
    exps = list(data.keys())
    # get specified methods, expand if "all" is used
    methods = ut.expand_key(config.get("methods", {}), exps)
    # get specified metrics, expand if "all" is used
    metrics = ut.expand_key(config.get("metrics", {}), exps)

    # adjust to no metrics being specified
    if metrics is not None:
        # expand on "all" if metrics are given
        metrics = ut.expand_key(metrics, exps)
    else:
        # if not metrics are given
        metrics = {}

    # get preprocessing specs
    pp = config.get("preprocess", None)
    # adjust to no preprocessing being specified
    if pp is not None:
        # expand on "all" if used
        pp = ut.expand_key(pp, exps)
    else:
        # if no preprocessing is specified
        pp = {}

    # get implemented methods
    methods_dict = C.METHODS["OPTIONS"].value
    # get implemented metrics
    metrics_dict = C.METRICS["OPTIONS"].value
    # get implemented workflows
    wfs_dict = C.WORKFLOWS["OPTIONS"].value

    # iterate over experiments
    for exp in exps:
        # get methods to use for experiment
        met_names = methods.get(exp, {})

        # get preprocessing for experiment and given method, expand if "all" is used
        pp[exp] = ut.expand_key(pp.get(exp, {}), met_names)

        # read data for experiment
        input_dict = ut.read_data(data[exp])

        # iterate over methods for experiment
        exp_metrics = ut.expand_key(metrics.get(exp, {}), met_names)

        for _met_name in met_names:
            # check for fuzzy match if enabled
            if use_fuzzy_match:
                met_name = ut.get_fuzzy_key(_met_name, methods_dict)
                met_name = ut.get_fuzzy_key(met_name, wfs_dict)
            else:
                met_name = _met_name

            if met_name not in wfs_dict:
                # extract composition formula if specified, else None
                recipe_dict = methods[exp][_met_name].get("recipe")
                if recipe_dict is None:
                    recipe_dict = methods[exp][_met_name].get("compose")

                if recipe_dict is not None:
                    # compose workflow from recipe if specified
                    # Note: keyword `recipe` should be used in config file
                    #       to signify a recipe
                    method = wf.Composite(recipe_dict, use_fuzzy_match=use_fuzzy_match)
                else:
                    NotImplementedError
            else:
                method = wfs_dict[met_name]
            # define output directory
            out_dir = osp.join(root_dir, exp, met_name)
            # create output directory (and parent folders if necessary)
            os.makedirs(out_dir, exist_ok=True)

            # preprocess data
            for ad_type in ["sc", "sp"]:
                # argmap is used to map sp->X_to and sc->X_from
                ad_i = input_dict.get(argmap[ad_type])
                if ad_i is None:
                    continue
                # to avoid sequential preprocessing
                if "_old" in ad_i.layers:
                    ad_i.X = ad_i.layers["_old"].copy()
                else:
                    ad_i.layers["_old"] = ad_i.X.copy()

                # get preprocessing procedure for (experiment,method, data type)
                pp_met_dict = ut.recursive_get(pp, exp, _met_name, ad_type)
                # iterate over preprocessing steps if multiple specified
                for pp_met_name, pp_met_kwargs in pp_met_dict.items():
                    if pp_met_name in C.PREPROCESS["OPTIONS"].value:
                        pp_met = C.PREPROCESS["OPTIONS"].value[pp_met_name]
                        ad_i = pp_met.pp(ad_i, ad_type, **pp_met_kwargs)
                input_dict[argmap[ad_type]] = ad_i

            # get method parameters for experiment
            method_params = methods[exp][_met_name].get("params", {})
            # define input to method for experiment
            method_params["out_dir"] = out_dir
            # run method
            input_dict = method.run(input_dict, experiment_name=exp, **method_params)

            # save output if specified
            if save_mode:
                method.save(input_dict, out_dir)

            # get method for (experiment,method)
            # (key,val) = (object, Dict[str,str])
            method_metrics = ut.recursive_get(exp_metrics, _met_name)

            # calculate all metrics
            # object is type of object to evaluate e.g., T or X_to_pred
            for object_name, object_cf in method_metrics.items():
                # object_name is name of object
                # object_cf is config for that object
                out_dir_object = osp.join(out_dir, object_name)
                os.makedirs(out_dir_object, exist_ok=True)
                # get metrics listed for object
                object_metrics = object_cf.get("metrics", {})
                # get fuzzy matches if enabled
                object_metrics = {
                    key: ut.get_from_dict_with_fuzzy(key, metrics_dict, use_fuzzy_match)
                    for key in object_metrics
                }
                object_metrics = {
                    key: val for key, val in object_metrics.items() if val is not None
                }
                # get reference (GT) for object
                # we expect same GT for all metrics pertaining to the same object
                ref_data = object_cf.get("data", {})
                metric_params = object_cf.get("params", {})

                if ref_data is None:
                    for metric_name, metric_fun in object_metrics.items():
                        # compute score
                        score = metric_fun.score(
                            input_dict, **metric_params
                        )
                        # save metric
                        metric_fun.save(score, out_dir_object)
                else:
                    # iterate over ground truth data sets
                    for ref_datum_name, ref_datum_cf in ref_data.items():
                        # read reference datum
                        ref_datum_value = ut.read_input_object(**ref_datum_cf)
                        # apply all metrics to reference
                        for metric_name, metric_fun in object_metrics.items():
                            # compute score
                            score = metric_fun.score(
                                input_dict, {object_name: ref_datum_value}, **metric_params
                            )
                            # save metric
                            metric_fun.save(score, out_dir_object)
