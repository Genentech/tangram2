import copy
from abc import ABC
from collections import OrderedDict
from typing import Any, Callable, Dict, List

import tangram2.evalkit.constants as C
import tangram2.evalkit.methods.utils as ut
from tangram2.evalkit.methods._methods import MethodClass


def flatten_het_list(input_list: list):
    """Flattens a nested list, handling heterogeneous data types.

    Args:
        input_list: list The list to flatten. Can contain nested lists and other
          data types.

    Returns:
        A new list with all nested lists flattened into a single level.

    Raises:
        RecursionError: If the list nesting is too deep.
    """
    flattened_list = []
    for element in input_list:
        if isinstance(element, list):
            flattened_list.extend(flatten_list(element))
        else:
            flattened_list.append(element)
    return flattened_list


class IdentityFun:
    """Identify Function

    Has all the method of a 'MethodClass'
    but applies to transformation to the data,
    i.e., it returns the input data unchanged
    upon calling 'run'

    """

    @classmethod
    def run(cls, *args, **kwargs):
        """Provides compatibility with a specific interface.

        Args:
            cls: The class this method is called on.
            *args: Variable positional arguments.
            **kwargs: Variable keyword arguments.

        Returns:
            dict: An empty dictionary.
        """
        return {}

    @classmethod
    def save(cls, *args, **kwargs):
        """Does nothing and returns None.

        Args:
            cls: The class this method is called on.
            *args: Variable length positional arguments (ignored).
            **kwargs: Variable length keyword arguments (ignored).

        Returns:
            None
        """
        return None


class Composite:
    """Composite class"""

    def __init__(
        self,
        methods: Dict[str, str],
        use_fuzzy_match: bool = False,
        check_compatibility: bool = True,
    ):
        self.cc = check_compatibility
        self.methods = []
        self._methods = OrderedDict()
        available_vars = []
        for k, (method_key, method_name) in enumerate(methods.items()):
            self.methods.append(method_name)
            method_fun = C.METHODS["OPTIONS"].value.get(method_name)
            if method_fun is not None:
                if not self._methods:
                    self._methods[method_key] = method_fun
                    if self.cc:
                        available_vars += flatten_het_list(method_fun.ins)
                        available_vars += method_fun.outs
                else:
                    if self.cc:
                        is_comp = all([x in available_vars for x in method_fun.ins])
                        assert is_comp, "{} is not compatible with {}".format(
                            method_name, self.methods[k - 1]
                        )
                        available_vars += method_fun.outs
                    self._methods[method_key] = method_fun
            else:
                raise NotImplementedError

    def run(self, input_dict: Dict[str, Any], **kwargs):
        """Executes a series of methods defined in a workflow.

        Args:
            input_dict:  A dictionary containing input data for the workflow.
            **kwargs: Keyword arguments containing method-specific parameters.  Each
                key should correspond to a method key, and the value should be a
                dictionary of parameters for that method.

        Returns:
            A dictionary containing the updated input data after all methods
            in the workflow have been executed.

        Raises:
            Exception: Exceptions raised by individual methods in the workflow will be
                propagated upwards.
        """
        for method_key in self._methods.keys():
            method_kwargs = ut.ifnonereturn(kwargs.get(method_key), {})
            out = self._methods[method_key].run(input_dict, **method_kwargs)
            input_dict.update(out)
        return input_dict

    def save(self, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        """Saves the results of the workflow.

        Args:
            res_dict: A dictionary containing the results of the workflow.
            out_dir: The directory to save the results to.
            **kwargs: Additional keyword arguments to pass to the save methods
                of individual workflow elements.

        Returns:
            None

        Raises:
            Exception: If any error occurs during the saving process.
        """
        for method_key in self._methods.keys():
            out = self._methods[method_key].save(res_dict, out_dir, **kwargs)


class WorkFlowClass(MethodClass):
    """Workflow baseclass

    to be used when we want to
    hardcode a composable (recipe) workflow

    """

    flow = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(cls, input_dict: Dict[str, Any], **kwargs):
        """Executes a flow with the given input.

        Args:
            cls: The class instance the flow belongs to.
            input_dict (Dict[str, Any]): The input data for the flow.
            **kwargs: Additional keyword arguments passed to the flow's run method.

        Returns:
            Any: The output of the flow execution.  The exact type and value
                depend on the specific flow being executed.

        Raises:
            Any: Exceptions raised by the underlying flow's run method can be
                propagated.
        """
        out = cls.flow.run(input_dict=input_dict, **kwargs)
        return out

    @classmethod
    def save(cls, res_dict: Dict[str, Any], out_dir: str, **kwargs) -> None:
        """Saves the results.

        Args:
            cls: The class instance.
            res_dict: A dictionary containing the results to be saved.
            out_dir: The output directory where the results will be saved.
            **kwargs: Additional keyword arguments passed to the underlying save method.

        Returns:
            None

        Raises:
            Any exception raised by the underlying save method.
        """
        cls.flow.save(res_dict, out_dir, **kwargs)


class Workflow:

    def __init__(
        self,
        methods: Dict[str, Dict[str, Any]],
        check_compatibility: bool = True,
        methods_key: str = "method",
        params_key: str = "params",
    ):
        """
        helper to construct workflow of methods

        Args:
          methods: A dict of dicts with the name of each method/step. The dict is expected to be
          formatted accordingly dict(step_1 = dict( methods_key = function/class, params_key =
          {prm_1:prm_1_val,..., prm_p:prm_p_val}), ..., step_k = .. )

          check_compatibility: if True we check if methods can be chained together

          methods_key: name of key to indicate method/function name in the methods dict

          params_key: name of key to indicate parameters in the methods dict


        Returns:
          a worfklow object that can be run via the .run method,
          the run method is essentially a composition of the different steps listed in the methods dict.
          the steps will be run in the order that they are listed in the dict.

        Example:
        ```
        wf_setup = {'pp' : dict(method = tg.met.preprocess.StandardMoscot),
                    'map': dict(method = tg.met.map_methods.TangramV1Map),
                    'pred': dict(method =  tg.met.pred_methods.TangramV1Pred),
                     'group': dict(method = tg.met.grp_methods.QuantileGroup,
                                   params = {'feature_name':'cd274'}),
        }

        wf = Workflow(wf_setup)
        wf.run(input_dict)

        ```

        """
        self.params_key = params_key
        self.methods_key = methods_key
        for key, val in methods.items():
            has_met = self.methods_key in val
            if not has_met:
                raise ValueError(
                    "please provide methods as dict(step_1 = dict({} = Callable, {} = {{prm_1:prm_1_val,...,prm_p:prm_p_val}}), .., step_k = ...)".format(
                        self.methods_key, self.params_key
                    )
                )
            has_prm = "params" in val
            if not has_prm:
                val["params"] = {}
                methods[key] = val
        self.methods = dict()
        self.cc = check_compatibility
        available_vars = []
        for method_name, method_obj in methods.items():
            _method_fun = method_obj[self.methods_key]
            if hasattr(_method_fun, "run"):
                method_mod = _method_fun
                method_fun = _method_fun.run
            elif hasattr(_method_fun, "pp"):
                method_mod = _method_fun
                method_fun = _method_fun.pp
            else:
                method_fun = _method_fun
                method_mod = None
            if len(method_obj) == 1:
                method_prms = {}
            else:
                method_prms = method_obj[self.params_key]
            self.methods[method_name] = [method_fun, method_prms]
            if self.cc and self._is_checkable(method_mod):
                if len(available_vars) == 0:
                    available_vars += method_mod.ins
                    available_vars += method_mod.outs
                else:
                    for obj in method_mod.ins:
                        if isinstance(obj, (list, tuple)):
                            any_in_input = any([x in available_vars for x in obj])
                            if not any_in_input:
                                msg = "Method {} is not compatible with prior methods in workflow".format(
                                    method_mod
                                )
                                raise ValueError(msg)
                    available_vars += method_mod.outs
            self._methods = copy.deepcopy(self.methods)

    def _is_checkable(self, x):
        if x is None:
            return False
        return hasattr(x, "ins") & hasattr(x, "outs")

    def __add__(self, other):
        combined_methods = (
            self._construct_method_dict() | other._construct_method_dict()
        )
        return Workflow(
            combined_methods,
            check_compatibility=False,
            methods_key=self.methods_key,
            params_key=self.params_key,
        )

    def reset_steps(self):
        """Resets the methods to their initial state.

        Args:
            self: The object instance.

        Returns:
            None.
        """
        self.methods = copy.deepcopy(self._methods)

    def _construct_method_dict(
        self, methods_key: str | None = None, params_key: str | None = None
    ):
        methods_key = self.methods_key if methods_key is None else methods_key
        params_key = self.params_key if params_key is None else params_key
        method_dict = {
            name: {methods_key: method_fun, params_key: method_params}
            for name, (method_fun, method_params) in self.methods.items()
        }
        return method_dict

    def list_methods(self, include_callable: bool = True):
        """Prints the methods and their parameters.

        Args:
            include_callable: Whether to include the callable object in the output.
                Defaults to True.

        Returns:
            None.

        """
        for step_name, (step_fun, step_prms) in self.methods.items():
            print(f"step_name : {step_name}")
            if include_callable:
                print(f"callable : {step_fun}")
            print("params :")
            if step_prms:
                print_prms = "\n".join(
                    ["  - {} : {}".format(k, str(v)) for k, v in step_prms.items()]
                )
            else:
                print_prms = "  - default"
            print(print_prms)

    def update_step(
        self,
        step_name,
        step_fun: Callable | None = None,
        step_params: Dict[str, Any] | None = None,
    ):
        """Updates a processing step.

        Args:
            step_name: The name of the step to update.
            step_fun: The new function for the step. If None, the existing function is kept.
            step_params:  New parameters for the step. If None, existing parameters are kept.
                          If provided, new parameters will update/override the existing ones.

        Returns:
            None. Prints a message if the step name is not found.
        """
        if step_name not in self.methods:
            print("step not present in methods")
            return None
        if step_fun is None:
            step_fun, _ = self.methods[step_name]
        _, step_params_old = self.methods[step_name]
        if step_params is None:
            step_params = step_params_old
        else:
            step_params = step_params_old | step_params
        self.methods[step_name] = [step_fun, step_params]

    def run(
        self,
        input_dict: Dict[str, Any],
        verbose: bool = False,
        return_dict: bool = False,
        **kwargs,
    ):
        """run worfklow

        input_dict: standard input dictionary
        verbose: use verbose mode
        return_dict: return the updated input_dict


        """
        for method_name, (method_fun, method_prms) in self.methods.items():
            if verbose:
                print(
                    "\t>>running step {} with the following parameters".format(
                        method_name
                    )
                )
                if method_prms:
                    print(method_prms)
            out = method_fun(input_dict, **method_prms)
            if out is not None:
                input_dict.update(out)
        if return_dict:
            return input_dict
