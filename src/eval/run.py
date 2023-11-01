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

    argmap = dict(
        sc="X_from",
        sp="X_to",
    )

    data = config.pop("data")
    exps = list(data.keys())

    methods = ut.expand_key(config["methods"], exps)
    metrics = ut.expand_key(config["metrics"], exps)
    pp = ut.expand_key(config.get("preprocess", {}), exps)

    methods_dict = C.METHODS["OPTIONS"].value
    metrics_dict = C.METRICS["OPTIONS"].value
    wfs_dict = C.WORKFLOWS["OPTIONS"].value

    for exp in exps:
        met_names = methods[exp]
        pp[exp] = ut.expand_key(pp[exp], met_names)

        input_dict = ut.read_data(data[exp])

        for _met_name in methods[exp]:

            if use_fuzzy_match:
                met_name = ut.get_fuzzy_key(_met_name, methods_dict)
                met_name = ut.get_fuzzy_key(met_name, wfs_dict)
            else:
                met_name = _met_name

            compose_dict = methods[exp][_met_name].get("compose")

            if compose_dict is not None:
                method = wf.compose_workflow_from_input(compose_dict, methods_dict)
            elif met_name in methods_dict:
                method = methods_dict[met_name]
            elif met_name in wfs_dict:
                method = wfs_dict[met_name]
            else:
                NotImplementedError

            out_dir = osp.join(root_dir, exp, met_name)
            os.makedirs(out_dir, exist_ok=True)

            for ad_type in ["sc", "sp"]:
                ad_i = input_dict[argmap[ad_type]]
                if "_old" in ad_i.layers:
                    ad_i.X = ad_i.layers["_old"].copy()
                else:
                    ad_i.layers["_old"] = ad_i.X.copy()

                pp_met_dict = ut.recursive_get(pp, exp, _met_name, ad_type)

                for pp_met_name, pp_met_kwargs in pp_met_dict.items():
                    if pp_met_name in C.PREPROCESS["OPTIONS"].value:
                        pp_met = C.PREPROCESS["OPTIONS"].value[pp_met_name]
                        pp_met.pp(ad_i, **pp_met_kwargs)

                input_dict[ad_type] = ad_i

            # TODO: remove this
            met_kwargs = method.get_kwargs()

            inp_kwargs = dict(
                to_spatial_key=data[exp]["sp"].get("spatial_key", "spatial"),
                from_spatial_key=data[exp]["sc"].get("spatial_key", None),
            )
            method_params = methods[exp][_met_name].get("params", {})
            met_input = met_kwargs | inp_kwargs | method_params
            met_input["out_dir"] = out_dir

            met_val = method.run(input_dict, **met_input)

            if save_mode:
                method.save(met_val, out_dir)

            method_metrics = ut.recursive_get(metrics, exp, _met_name)

            for _metric_name, metric_props in method_metrics.items():

                if use_fuzzy_match:
                    metric_name = ut.get_fuzzy_key(
                        _metric_name, metrics_dict, verbose=verbose
                    )
                else:
                    metric_name = _metric_name

                if metric_name in metrics_dict:
                    metric = metrics_dict[metric_name]
                else:
                    NotImplementedError

                if metric_props is not None and "ground_truth" in metric_props:
                    name = metric_props["ground_truth"].pop("name")
                    name = argmap.get(name, name)
                    if osp.isfile(name):
                        ground_truth = ut.read_input_object(
                            name, **metric_props["ground_truth"]
                        )
                        ground_truth = dict(true=ground_truth)
                    else:
                        ground_truth = metric.get_gt(
                            input_dict, **metric_props["ground_truth"]
                        )
                else:
                    ground_truth = {}

                score_vals = ground_truth | met_val
                score = metric.score(score_vals)

                metric.save(score, out_dir)
