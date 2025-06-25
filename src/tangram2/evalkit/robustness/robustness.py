import copy
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
    """Run a workflow K times

    Args:
        input_dict: state dicts of objects needed to run workflow
        workflow: Workflow object
        n_repeats: number of times to run the workflow
        params: dictionary on the form `{step_name : {parameter_name : parameter_value}, ..}`. For example, random seed.
        inplace: bool: whether the output from the workflow should be added to the input dict or returned as a new dict.

    Returns:
      Either updates input dict with the output of the workflow or returns a new dict, each object will have n_repeat versions included.


    """

    # get original input_dict keys
    og_keys = list(input_dict.keys())

    # helper function to copy dict objects
    def _dict_copy_helper(value):
        if hasattr(value, "copy"):
            return value.copy()
        else:
            return copy.deepcopy(value)

    # output dictionary of workflow
    new_input_dict = dict()

    # iterate over each repeat
    for repeat in range(n_repeats):
        # make copy of dict items
        input_dict_copy = {
            key: _dict_copy_helper(value) for key, value in input_dict.items()
        }

        # if user has provided parameters (e.g., random seed)
        if params is not None:

            # iterate over each step/parameter set
            for step_name, step_params_all in params.items():
                try:
                    # get the parameters value for repeat
                    step_params_repeat = {
                        key: val[repeat] for key, val in step_params_all.items()
                    }
                    # update step
                    workflow.update_step(step_name, step_params=step_params_repeat)
                except IndexError:
                    print(
                        "The number of parameter options must be the same as the number of repeats."
                    )

        # run workflow
        workflow.run(input_dict_copy)

        # get the output from the workflow run
        for key, value in input_dict_copy.items():
            # only add values not in og input_dict
            if key not in og_keys:
                if key not in new_input_dict:
                    new_input_dict[key] = []
                new_input_dict[key].append(value)

    # reset any changes to the workflow
    workflow.reset_steps()

    # if update of the og input dict should be done
    if inplace:
        for key, value in new_input_dict.items():
            input_dict_copy[key] = value
        return input_dict_copy

    return new_input_dict
