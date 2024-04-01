from abc import ABC
from collections import OrderedDict
from typing import Any, Callable, Dict, List

import telegraph.constants as C
import telegraph.methods.utils as ut
from telegraph.methods._methods import MethodClass


class IdentityFun:
    """Identify Function

    Has all the method of a 'MethodClass'
    but applies to transformation to the data,
    i.e., it returns the input data unchanged
    upon calling 'run'

    """

    @classmethod
    def run(cls, *args, **kwargs):
        # return empty dictionary for compatibility
        return {}

    @classmethod
    def save(cls, *args, **kwargs):
        # save nothing
        return None


class Composite:

    def __init__(
        self,
        methods: Dict[str, str],
        use_fuzzy_match: bool = False,
        check_compatibility: bool = True,
    ):

        # if compatibility should be checked
        self.cc = check_compatibility

        # list method names, for 'info' support
        self.methods = []

        # Ordered Dictionary for method objects
        self._methods = OrderedDict()

        # variables that will be available as input to a method in the chain
        available_vars = []

        # iterate over all methods in the specified workflow

        for k, (method_key, method_name) in enumerate(methods.items()):

            # add method to list of methods in workflow
            self.methods.append(method_name)

            # get method object, return None if not implemented
            method_fun = C.METHODS["OPTIONS"].value.get(method_name)

            # if method is implemented
            if method_fun is not None:
                # if first method to be added to workflow
                if not self._methods:
                    # add method to dictionary of methods
                    self._methods[method_key] = method_fun
                    # update available variables
                    if self.cc:
                        available_vars += method_fun.ins
                        available_vars += method_fun.outs

                # if not first method to be added
                else:
                    # check if the method to add is compatible with chain
                    if self.cc:
                        is_comp = all([x in available_vars for x in method_fun.ins])
                        # if not compatible raise error
                        assert is_comp, "{} is not compatible with {}".format(
                            method_name, self.methods[k - 1]
                        )
                        # update set of available variables
                        available_vars += method_fun.outs

                    # if compatible, add to list of methods
                    self._methods[method_key] = method_fun
            else:
                raise NotImplementedError

    def run(
        self,
        input_dict: Dict[str, Any],
        **kwargs,
    ):

        # execute chain of methods in workflow
        for method_key in self._methods.keys():
            method_kwargs = ut.ifnonereturn(kwargs.get(method_key), {})

            out = self._methods[method_key].run(
                input_dict,
                **method_kwargs,
            )
            # update input dict
            input_dict.update(out)

        return input_dict

    def save(self, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        # save each of the outputs from
        # every element

        # iterate over methods in workflow
        for method_key in self._methods.keys():
            out = self._methods[method_key].save(res_dict, out_dir, **kwargs)


class WorkFlowClass(MethodClass):
    """Workflow baseclass

    to be used when we want to
    hardcode a composable (recipe) workflow

    """

    # flow should hold the composition of methods
    flow = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def run(
        cls,
        input_dict: Dict[str, Any],
        **kwargs,
    ):
        # run function equates to running the composition in "flow"
        out = cls.flow.run(
            input_dict=input_dict,
            **kwargs,
        )

        return out

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        **kwargs,
    ) -> None:
        # save using "flow's" save method
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

        # list method names, for 'info' support
        self.methods = []
        self.methods_prms = []
        self.methods_names = []

        self.cc = check_compatibility

        # variables that will be available as input to a method in the chain
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

            # get method parameters
            if len(method_obj) == 1:
                method_prms = {}
            else:
                method_prms = method_obj[self.params_key]

            self.methods.append(method_fun)
            self.methods_prms.append(method_prms)
            self.methods_names.append(method_name)

            if self.cc and self._is_checkable(method_mod):
                if len(available_vars) == 0:
                    available_vars += method_mod.ins
                    available_vars += method_mod.outs
                else:
                    is_comp = all([x in available_vars for x in method_mod.ins])
                    # if not compatible raise error
                    if not is_comp:
                        msg = "Method {} is not compatible with prior methods in workflow".format(
                            method_mod
                        )
                        raise ValueError(msg)

                    available_vars += method_mod.outs

    def _is_checkable(self, x):
        if x is None:
            return False
        return hasattr(x, "ins") & hasattr(x, "outs")

    def __add__(self, other):
        combined_methods = {
            **self._construct_method_dict(
                methods_key=self.methods_key, params_key=self.params_key
            ),
            **other._construct_method_dict(
                methods_key=self.methods_key, params_key=self.params_key
            ),
        }
        return Workflow(
            combined_methods,
            check_compatibility=False,
            methods_key=self.methods_key,
            params_key=self.params_key,
        )

    def _construct_method_dict(
        self, methods_key: str | None = None, params_key: str | None = None
    ):

        methods_key = self.methods_key if methods_key is None else methods_key
        params_key = self.params_key if params_key is None else params_key

        return {
            name: {methods_key: method, params_key: params}
            for name, method, params in zip(
                self.method_names, self.methods, self.method_params
            )
        }

    def list_methods(self, include_callable: bool = True):
        for step_name, step_fun, step_prms in zip(
            self.methods_names, self.methods, self.methods_prms
        ):
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

        for method_fun, method_prms, method_name in zip(
            self.methods, self.methods_prms, self.methods_names
        ):

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
