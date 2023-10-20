import os
import os.path as osp
from typing import Any, Dict

import anndata as ad

from . import constants as C
from . import utils as ut


def run(config: Dict[str, Any], root_dir: str = ".", save_mode: bool = False):

    argmap = dict(
        sc="X_from",
        sp="X_to",
    )

    data = config.pop("data")
    exps = list(data.keys())

    methods = ut.expand_key(config["methods"], exps)
    method_params = ut.expand_key(config["method_params"], exps)
    metrics = ut.expand_key(config["metrics"], exps)
    pp = ut.expand_key(config.get("preprocess", {}), exps)

    for exp in exps:
        met_names = methods[exp]
        pp[exp] = ut.expand_key(pp[exp], met_names)

        input_dict = ut.read_data(data[exp])

        for met_name in methods[exp]:
            if met_name in C.METHODS["OPTIONS"].value:
                method = C.METHODS["OPTIONS"].value[met_name]
            elif met_name in C.WORKFLOWS["OPTIONS"].value:
                method = C.WORKFLOWS["OPTIONS"].value[met_name]
            else:
                continue

            out_dir = osp.join(root_dir, exp, met_name)
            os.makedirs(out_dir, exist_ok=True)

            for ad_type in ["sc", "sp"]:
                ad_i = input_dict[argmap[ad_type]]
                if "_old" in ad_i.layers:
                    ad_i.X = ad_i.layers["_old"].copy()
                else:
                    ad_i.layers["_old"] = ad_i.X.copy()

                pp_met_dict = ut.recursive_get(pp, exp, met_name, ad_type)

                for pp_met_name, pp_met_kwargs in pp_met_dict.items():
                    if pp_met_name in C.PREPROCESS["OPTIONS"].value:
                        pp_met = C.PREPROCESS["OPTIONS"].value[pp_met_name]
                        pp_met.pp(ad_i, **pp_met_kwargs)

                # TODO: maybe this is unnecessary
                input_dict[ad_type] = ad_i

            met_kwargs = method.get_kwargs()
            inp_kwargs = dict(
                to_spatial_key=data[exp]["sp"].get("spatial_key", "spatial"),
                from_spatial_key=data[exp]["sc"].get("spatial_key", None),
            )
            met_input = met_kwargs | inp_kwargs | method_params[exp].get(met_name, {})
            met_input["out_dir"] = out_dir

            met_val = method.run(input_dict, **met_input)

            if save_mode:
                method.save(met_val, out_dir)

            method_metrics = ut.recursive_get(metrics, exp, met_name)

            for metric_name, metric_props in method_metrics.items():
                if metric_name in C.METRICS["OPTIONS"].value:
                    metric = C.METRICS["OPTIONS"].value[metric_name]
                else:
                    continue

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
