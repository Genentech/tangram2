from abc import ABC
from collections import OrderedDict
from typing import Any, Dict

import cccv.evaluation.constants as C
import cccv.evaluation.dea_methods as dmet
import cccv.evaluation.grp_methods as gmet
import cccv.evaluation.map_methods as mmet
import cccv.evaluation.pred_methods as pmet
import cccv.evaluation.utils as ut
from cccv.evaluation._methods import MethodClass


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
    ):
        # list method names, for 'info' support
        self.methods = []

        # Ordered Dictionary for method objects
        self._methods = OrderedDict()

        # variables that will be available as input to a method in the chain
        available_vars = []

        # iterate over all methods in the specified workflow
        for k, (method_key, _method_name) in enumerate(methods.items()):
            # use fuzzy match for method classes if enabled
            if use_fuzzy_match:
                method_name = ut.get_fuzzy_key(_method_name, C.METHODS["OPTIONS"].value)
            else:
                method_name = _method_name

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
                    available_vars += method_fun.ins
                    available_vars += method_fun.outs

                # if not first method to be added
                else:
                    # check if the method to add is compatible with chain
                    is_comp = all([x in available_vars for x in method_fun.ins])
                    # if not compatible raise error
                    assert is_comp, "{} is not compatible with {}".format(
                        method_name, self.methods[k - 1]
                    )

                    # if compatible, add to list of methods
                    self._methods[method_key] = method_fun
                    # update set of available variables
                    available_vars += method_fun.outs
            else:
                raise NotImplementedError

    def run(
        self,
        input_dict: Dict[str, Any],
        experiment_name: str | None = None,
        **kwargs,
    ):

        # execute chain of methods in workflow
        for method_key in self._methods.keys():
            out = self._methods[method_key].run(
                input_dict,
                experiment_name=experiment_name,
                **kwargs.get(method_key, {}),
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


class Tangram2BaselineWorkflow(WorkFlowClass):
    """Implementation of Hejin's workflow"""

    flow = Composite(
        dict(
            map="map_tangram_v2",
            pred="prd_tangram_v2",
            group="grp_threshold",
            dea="dea_scanpy",
        )
    )
