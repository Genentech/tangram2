from . import constants as C
import os.path as osp
from typing import Any, Dict
import anndata as ad
import os


def run(config: Dict[str, Any], out_dir: str = "."):
    data = config.pop("data")
    exps = list(data.keys())

    for key, exp_conf in config.items():
        if "all" in exp_conf:
            new_exp_conf = {x: exp_conf["all"] for x in exps}
            config[key] = new_exp_conf

    methods = config["methods"]
    method_params = config["method_params"]
    metrics = config["metrics"]
    pp = config["preprocess"]

    for exp in exps:
        ad_sc_pth = data[exp]["sc"]["path"]
        ad_sp_pth = data[exp]["sp"]["path"]
        ad_sc = ad.read_h5ad(ad_sc_pth)
        ad_sp = ad.read_h5ad(ad_sp_pth)

        sc_layer = data[exp]["sc"].get("layer", None)
        if sc_layer is not None:
            ad_sc.X = ad_sc.layers[sc_layer]
        sp_layer = data[exp]["sp"].get("layer", None)
        if sp_layer is not None:
            ad_sp.X = ad_sp.layers[sc_layer]

        for met_name in methods[exp]:
            if met_name in C.METHODS["METHODS"].value:
                method = C.METHODS["METHODS"].value[met_name]
            else:
                continue

            for ad_type, ad_i in zip(["sc", "sp"], [ad_sc, ad_sp]):
                if "_old" in ad_i.layers:
                    ad_i.X = ad_i.layers["_old"]
                else:
                    ad_i.layers["_old"] = ad_i.X.copy()

                # TODO: add preprocessing here
                # pp_met = pp[exp][met_name][ad_type]

            met_kwargs = method.get_kwargs()
            inp_kwargs = dict(
                to_spatial_key=data[exp]["sp"].get("spatial_key", "spatial"),
                from_spatial_key=data[exp]["sc"].get("spatial_key", None),
            )
            kwargs = met_kwargs | inp_kwargs | method_params[exp].get(met_name, {})
            met_val = method.run(ad_sp, ad_sc, **kwargs)

            for metric_name, gt_name in metrics[exp]:
                if metric_name in C.METRICS["METRICS"].value:
                    metric = C.METRICS["METRICS"].value[metric_name]
                else:
                    continue

                gt_val = metric.get_gt(ad_sp, ad_sc, gt_name)
                vals = gt_val | met_val
                score = metric.score(vals)
                out_pth = osp.join(out_dir, exp, met_name, metric_name + ".txt")

                os.makedirs(osp.dirname(out_pth), exist_ok=True)

                metric.save(score, out_pth)
