import os
import os.path as osp
from typing import Any, Dict

import anndata as ad

from . import constants as C
from . import utils as ut
from . import workflows as wf

OBJMAP = dict(
    sc="X_from",
    sp="X_to",
    map="T",
)


def preprocess_data(data_dict, pp_config, inplace: bool = True):
    for obj_name in data_dict.keys():
        obj = data_dict[obj_name]
        if isinstance(obj, ad.AnnData):
            if "_old" in obj.layers:
                obj.X = obj.layers["_old"].copy()
            else:
                obj.layers["_old"] = obj.X.copy()
        data_dict[obj_name] = obj

    for obj_name in pp_config.keys():
        obj = data_dict.get(obj_name)
        if obj is not None:
            pp_steps = pp_config[obj_name]
            for pp_step_name in pp_steps.keys():
                pp_step = C.PREPROCESS["OPTIONS"].value.get(pp_step_name)
                if pp_step is not None:
                    pp_step_kwargs = ut.ifnonereturn(pp_steps.get(pp_step_name), {})
                    pp_step.pp(obj, input_type=obj_name, **pp_step_kwargs)
        data_dict[obj_name] = obj

    if not inplace:
        return data_dict


def build_workflow(recipe, use_fuzzy_match=True):
    method_names = {key: key for key in recipe.keys()}

    workflow = wf.Composite(method_names, use_fuzzy_match=use_fuzzy_match)

    return workflow


def evaluate_workflow(input_dict, eval_config, out_dir):
    match_objs = [obj for obj in eval_config.keys() if obj in input_dict]

    for obj_name in match_objs:
        out_dir_object = osp.join(out_dir, "metrics", obj_name)
        os.makedirs(out_dir_object, exist_ok=True)

        obj_config = eval_config[obj_name]
        obj_data_config = obj_config.get(C.CONF.data.value, {})

        ref_data_pth = obj_data_config.get("path")
        ref_data_adata_key = obj_data_config.get("adata_key")
        ref_data = ut.read_input_object(ref_data_pth, adata_key=ref_data_adata_key)
        ref_data = {obj_name: ref_data}

        metrics = obj_config.get(C.CONF.metrics.value, {})
        metrics = {
            mn: mf for mn, mf in metrics.items() if mn in C.METRICS["OPTIONS"].value
        }

        for metric_name in metrics.keys():
            metric_fun = C.METRICS["OPTIONS"].value[metric_name]
            metrics[metric_name] = ut.ifnonereturn(metrics[metric_name], {})
            score = metric_fun.score(input_dict, ref_data, **metrics[metric_name])
            metric_fun.save(score, out_dir_object)


def process_experiment(
    exp_config: Dict[str, Any],
    use_fuzzy_match: bool = False,
    experiment_name: str = "",
    root_dir: str = ".",
    save_mode: bool = False,
    verbose: bool = False,
):
    data_paths = exp_config.get(C.CONF.data.value, {})
    data_paths = {OBJMAP.get(key, key): val for key, val in data_paths.items()}

    input_dict = ut.read_data(data_paths)

    wfs = exp_config[C.CONF.wfs.value]

    for wf_name in wfs.keys():
        out_dir = osp.join(root_dir, experiment_name, wf_name)
        os.makedirs(out_dir, exist_ok=True)

        pp_config = wfs[wf_name].get(C.CONF.pp.value, {})
        preprocess_data(input_dict, pp_config)

        recipe = wfs[wf_name].get(C.CONF.recipe.value)
        assert recipe is not None, "must give a recipe for the workflow"
        workflow = build_workflow(recipe, use_fuzzy_match)
        workflow.run(input_dict, experiment_name=experiment_name, **recipe)

        if save_mode:
            workflow.save(input_dict, out_dir)

        eval_conf = exp_config.get(C.CONF.eval.value, {})

        evaluate_workflow(
            input_dict,
            eval_conf,
            out_dir,
        )


def run(
    config: Dict[str, Any],
    root_dir: str = ".",
    save_mode: bool = False,
    use_fuzzy_match: bool = False,
    verbose: bool = False,
):
    for exp_name, exp_config in config.items():
        process_experiment(
            exp_config,
            experiment_name=exp_name,
            use_fuzzy_match=use_fuzzy_match,
            verbose=verbose,
            root_dir=root_dir,
            save_mode=save_mode,
        )
