from abc import ABC
from typing import Any, Dict

import eval.constants as C
import eval.dea_methods as dmet
import eval.grp_methods as gmet
import eval.map_methods as mmet
import eval.pred_methods as pmet
import eval.utils as ut
from eval._methods import MethodClass


def compose_workflow_from_input(source_d: Dict[str, str], target_d: Dict[str, str]):
    map_method = source_d.get("map")
    pred_method = source_d.get("pred")
    group_method = source_d.get("group")
    dea_method = source_d.get("dea")

    workflow = Composite(
        map_method=map_method,
        pred_method=pred_method,
        group_method=group_method,
        dea_method=dea_method,
    )

    return workflow


class IdentityFun:
    @classmethod
    def run(cls, *args, **kwargs):
        return {}

    @classmethod
    def save(cls, *args, **kwargs):
        return None


class Composite:
    def __init__(
        self,
        map_method: str | None = None,
        pred_method: str | None = None,
        group_method: str | None = None,
        dea_method: str | None = None,
    ):
        self.methods = dict(
            map=map_method,
            pred=pred_method,
            group=group_method,
            dea=dea_method,
        )

        self.map = (
            C.METHODS["OPTIONS"].value[map_method]
            if map_method is not None
            else IdentityFun
        )
        self.pred = (
            C.METHODS["OPTIONS"].value[pred_method]
            if pred_method is not None
            else IdentityFun
        )
        self.group = (
            C.METHODS["OPTIONS"].value[group_method]
            if group_method is not None
            else IdentityFun
        )
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
        input_dict.update(out)

        # predict
        out = self.pred.run(input_dict, **pred_args)
        input_dict.update(out)

        # group
        out = self.group.run(input_dict, **group_args)
        input_dict.update(out)

        # dea
        out = self.dea.run(input_dict, **dea_args)
        input_dict.update(out)

        return input_dict

    def get_kwargs(*args, **kwargs):
        return {}

    def save(self, res_dict: Dict[str, Any], out_dir: str, **kwargs):
        # map
        self.map.save(res_dict, out_dir, **kwargs)

        # predict
        self.pred.save(res_dict, out_dir, **kwargs)

        # group
        self.group.save(res_dict, out_dir, **kwargs)

        # dea
        self.dea.save(res_dict, out_dir, **kwargs)


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

    @classmethod
    def save(
        cls,
        res_dict: Dict[str, Any],
        out_dir: str,
        **kwargs,
    ) -> None:
        cls.flow.save(res_dict, out_dir, **kwargs)


class HejinWorkflow(WorkFlowClass):
    flow = Composite(
        map_method="map_tangram_v2",
        pred_method="prd_tangram_v2",
        group_method="grp_threshold",
        dea_method="dea_scanpy",
    )
