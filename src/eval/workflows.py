from abc import ABC
from typing import Any, Dict

import eval.constants as C
import eval.dea_methods as dmet
import eval.grp_methods as gmet
import eval.map_methods as mmet
import eval.pred_methods as pmet
import eval.utils as ut
from eval._methods import MethodClass


class Composite:
    def __init__(
        self,
        map_method: str | None = None,
        pred_method: str | None = None,
        group_method: str | None = None,
        dea_method: str | None = None,
    ):

        self.methods = dict(map = map_method,
                            pred = pred_method,
                            group = group_method,
                            dea = dea_method,
                            )

        self.map = (
            C.METHODS["OPTIONS"].value[map_method]
            if map_method is not None
            else ut.identity_fun
        )
        self.pred = (
            C.METHODS["OPTIONS"].value[pred_method]
            if pred_method is not None
            else ut.identity_fun
        )
        self.group = (
            C.METHODS["OPTIONS"].value[group_method]
            if group_method is not None
            else ut.identity_fun
        )
        self.dea = (
            C.METHODS["OPTIONS"].value[dea_method]
            if dea_method is not None
            else ut.identity_fun
        )

    def run(
        self,
        input_dict: Dict[str, Any],
        map_args: Dict[str, Any],
        pred_args: Dict[str, Any],
        group_args: Dict[str, Any],
        dea_args: Dict[str, Any],
        **kwargs,
    ):
        out = self.map.run(input_dict, **map_args)
        input_dict.update(out)
        out = self.pred.run(input_dict, **pred_args)
        input_dict.update(out)
        out = self.group.run(input_dict, **group_args)
        input_dict.update(out)
        out = self.dea.run(input_dict, **dea_args)

        return out


class WorkFlowClass(MethodClass):
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
        out = cls.flow.run(
            input_dict=input_dict,
            map_args=map_args,
            pred_args=pred_args,
            group_args=group_args,
            dea_args=dea_args,
            **kwargs,
        )
        return out


class HejinWorkflow(WorkFlowClass):
    flow = Composite(
        map_method="map_tangram_v2",
        pred_method="prd_tangram_v2",
        group_method="grp_threshold",
        dea_method="dea_scanpy",
    )
