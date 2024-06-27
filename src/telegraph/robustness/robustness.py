from typing import Any, Dict

import anndata as ad

from ..methods.workflows import Workflow


def run_k_times(
    input_dict: Dict[str, Any],
    workflow: Workflow,
    n_repeats: int,
    params: Dict[str, Dict[str, Any]] | None = None,
    inplace: bool = False,
):

    og_keys = list(input_dict.keys())

    def _dict_copy_helper(value):
        if hasattr(value, "copy"):
            return value.copy()
        return value

    new_input_dict = dict()

    for repeat in range(n_repeats):
        input_dict_copy = {
            key: _dict_copy_helper(value) for key, value in input_dict.items()
        }

        if params is not None:

            for step_name, step_params_all in params.items():
                try:
                    step_params_repeat = {
                        key: val[repeat] for key, val in step_params_all.items()
                    }
                    workflow.update_step(step_name, step_params=step_params_repeat)
                except IndexError:
                    print(
                        "The number of parameter options must be the same as the number of repeats."
                    )

        workflow.run(input_dict_copy)

        for key, value in input_dict_copy.items():
            if key not in og_keys:
                if key not in new_input_dict:
                    new_input_dict[key] = []
                new_input_dict[key].append(value)

    workflow.reset_steps()

    if inplace:
        for key, value in new_input_dict.items():
            input_dict_copy[key] = value
        return input_dict_copy

    return new_input_dict
