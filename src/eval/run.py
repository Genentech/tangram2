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
    methods = ut.expand_key(config["methods"], exps)
    # get specified metrics, expand if "all" is used
    metrics = config.get("metrics", None)

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
        met_names = methods[exp]

        # get preprocessing for experiment and given method, expand if "all" is used
        pp[exp] = ut.expand_key(pp.get(exp, {}), met_names)

        # read data for experiment
        input_dict = ut.read_data(data[exp])

        # iterate over methods for experiment
        for _met_name in met_names:

            # check for fuzzy match if enabled
            if use_fuzzy_match:
                met_name = ut.get_fuzzy_key(_met_name, methods_dict)
                met_name = ut.get_fuzzy_key(met_name, wfs_dict)
            else:
                met_name = _met_name

            # extract composition formula if specified, else None
            compose_dict = methods[exp][_met_name].get("compose")

            if compose_dict is not None:
                # compose workflow from recipe if specified
                # Note: keyword compose should be used in config file
                #       to signify a recipe
                method = wf.compose_workflow_from_input(compose_dict)
            elif met_name in methods_dict:
                # method is a single step get implementation
                method = methods_dict[met_name]
            elif met_name in wfs_dict:
                # method is a workflow, get implementation
                method = wfs_dict[met_name]
            else:
                NotImplementedError

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

                # get preprocssing procedure for (experiment,method, data type)
                pp_met_dict = ut.recursive_get(pp, exp, _met_name, ad_type)

                # iterate over preprocessing steps if multiple specified
                for pp_met_name, pp_met_kwargs in pp_met_dict.items():
                    if pp_met_name in C.PREPROCESS["OPTIONS"].value:
                        pp_met = C.PREPROCESS["OPTIONS"].value[pp_met_name]
                        pp_met.pp(ad_i, **pp_met_kwargs)

                input_dict[ad_type] = ad_i

            # TODO: remove this
            met_kwargs = method.get_kwargs()

            # get spatial coordinates from data
            inp_kwargs = dict(
                to_spatial_key=(
                    data[exp]["sp"].get("spatial_key", "spatial")
                    if "sp" in data[exp]
                    else None
                ),
                from_spatial_key=(
                    data[exp]["sc"].get("spatial_key", None)
                    if "sc" in data[exp]
                    else None
                ),
            )

            # get method parameters for experiment
            method_params = methods[exp][_met_name].get("params", {})
            # define input to method for experiment
            met_input = met_kwargs | inp_kwargs | method_params
            met_input["out_dir"] = out_dir

            # run method
            met_val = method.run(input_dict, **met_input)

            # save output if specified
            if save_mode:
                method.save(met_val, out_dir)

            # get method for (experiment,method)
            method_metrics = ut.recursive_get(metrics, exp, _met_name)

            # calculate all metrics
            for _metric_name, metric_props in method_metrics.items():

                # use fuzzy match if specified
                if use_fuzzy_match:
                    metric_name = ut.get_fuzzy_key(
                        _metric_name, metrics_dict, verbose=verbose
                    )
                else:
                    metric_name = _metric_name

                # check if metric is implemented
                if metric_name in metrics_dict:
                    metric = metrics_dict[metric_name]
                else:
                    NotImplementedError

                # get ground truth for metric
                if metric_props is not None and "ground_truth" in metric_props:
                    name = metric_props["ground_truth"].pop("name")
                    name = argmap.get(name, name)

                    # read if path
                    if osp.isfile(name):
                        ground_truth = ut.read_input_object(
                            name, **metric_props["ground_truth"]
                        )
                        ground_truth = dict(true=ground_truth)

                    # grab from data (according to method) if not file
                    else:
                        ground_truth = metric.get_gt(
                            input_dict, **metric_props["ground_truth"]
                        )
                else:
                    ground_truth = {}

                # compute score of metric
                score_vals = ground_truth | met_val
                score = metric.score(score_vals)

                # save metric score
                metric.save(score, out_dir)
