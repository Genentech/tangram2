from . import constants as C
from . import utils as ut
import os.path as osp
from typing import Any, Dict
import anndata as ad
import os


def run(config: Dict[str, Any], root_dir: str = "."):
    data = config.pop("data")
    exps = list(data.keys())

    methods = ut.expand_key(config["methods"], exps)
    method_params = ut.expand_key(config["method_params"], exps)
    metrics = ut.expand_key(config["metrics"], exps)
    pp = ut.expand_key(config.get("preprocess", {}), exps)

    for exp in exps:
        met_names = methods[exp]
        pp[exp] = ut.expand_key(pp[exp], met_names)

        input_data = ut.read_data(data[exp])


        for met_name in methods[exp]:
            if met_name in C.METHODS["OPTIONS"].value:
                method = C.METHODS["OPTIONS"].value[met_name]
            else:
                continue

            for ad_type ["sc", "sp"]:
                ad_i = input_dict[ad_type]
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
            met_input = input_data | met_kwargs | inp_kwargs | method_params[exp].get(met_name, {})

            met_val = method.run(**met_input)

            for metric_name, gt_name in metrics[exp][met_name].items():
                if metric_name in C.METRICS["OPTIONS"].value:
                    metric = C.METRICS["OPTIONS"].value[metric_name]
                else:
                    continue

                gt_val = metric.get_gt(**input_kwargs, gt_key = gt_name)
                vals = gt_val | met_val
                score = metric.score(vals)
                out_dir = osp.join(root_dir, exp, met_name)

                os.makedirs(out_dir, exist_ok=True)

                metric.save(score, out_dir)
