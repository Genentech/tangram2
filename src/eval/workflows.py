from abc import ABC
from typing import Any, Dict

import eval.constants as C
import eval.dea_methods as dmet
import eval.grp_methods as gmet
import eval.map_methods as mmet
import eval.pred_methods as pmet
import eval.utils as ut
from eval._methods import MethodClass


def compose_workflow_from_input(source_d: Dict[str, str]):
    """compose workflow from a dictionary

    get {map,pred,group,dea} methods from the dictionary

    """

    # get map method, default None
    map_method = source_d.get("map")
    # get pred method, default None
    pred_method = source_d.get("pred")
    # get grouping method, default None
    group_method = source_d.get("group")
    # get dea method, default None
    dea_method = source_d.get("dea")

    # compose workflow
    workflow = Composite(
        map_method=map_method,
        pred_method=pred_method,
        group_method=group_method,
        dea_method=dea_method,
    )

    return workflow


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
        map_method: str | None = None,
        pred_method: str | None = None,
        group_method: str | None = None,
        dea_method: str | None = None,
    ):

        # list method names, for 'info' support
        self.methods = dict(
            map=map_method,
            pred=pred_method,
            group=group_method,
            dea=dea_method,
        )

        # set map function to choice, if None then IdentifyFun
        self.map = (
            C.METHODS["OPTIONS"].value[map_method]
            if map_method is not None
            else IdentityFun
        )
        # set pred function to choice, if None then IdentifyFun
        self.pred = (
            C.METHODS["OPTIONS"].value[pred_method]
            if pred_method is not None
            else IdentityFun
        )

        # set group function to choice, if None then IdentifyFun
        self.group = (
            C.METHODS["OPTIONS"].value[group_method]
            if group_method is not None
            else IdentityFun
        )
        # set dea function to choice, if None then IdentifyFun
        self.dea = (
            C.METHODS["OPTIONS"].value[dea_method]
            if dea_method is not None
            else IdentityFun
        )

    def run(
        self,
        input_dict: Dict[str, Any],
        map_args: Dict[str, Any] = dict(),
        pred_args: Dict[str, Any] = dict(),
        group_args: Dict[str, Any] = dict(),
        dea_args: Dict[str, Any] = dict(),
        out_dir: str | None = None,
        **kwargs,
    ):
        # map
        out = self.map.run(input_dict, **map_args)
        # update input dictionary
        input_dict.update(out)

        # predict
        out = self.pred.run(input_dict, **pred_args)
        # update input dictionary
        input_dict.update(out)

        # group
        out = self.group.run(input_dict, **group_args)
        # update input dictionary
        input_dict.update(out)

        # dea
        out = self.dea.run(input_dict, **dea_args)
        # update input dictionary
        input_dict.update(out)

        return input_dict

    def save(self, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        # save each of the outputs from
        # every element

        # map
        self.map.save(res_dict, out_dir, **kwargs)

        # predict
        self.pred.save(res_dict, out_dir, **kwargs)

        # group
        self.group.save(res_dict, out_dir, **kwargs)

        # dea
        self.dea.save(res_dict, out_dir, **kwargs)


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
        map_args: Dict[str, Any] = {},
        pred_args: Dict[str, Any] = {},
        group_args: Dict[str, Any] = {},
        dea_args: Dict[str, Any] = {},
        **kwargs,
    ):

        # run function equates to running the composition in "flow"
        out = cls.flow.run(
            input_dict=input_dict,
            map_args=map_args,
            pred_args=pred_args,
            group_args=group_args,
            dea_args=dea_args,
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


class HejinWorkflow(WorkFlowClass):
    """Implementation of Hejin's workflow"""

    flow = Composite(
        map_method="map_tangram_v2",
        pred_method="prd_tangram_v2",
        group_method="grp_threshold",
        dea_method="dea_scanpy",
    )
