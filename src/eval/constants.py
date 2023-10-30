from enum import Enum

import eval.dea_methods as dmet
import eval.grp_methods as gmet
import eval.map_methods as mmet
import eval.pred_methods as pmet

from . import metrics as mtx
from . import preprocess as pp


# from . import methods as met


class PREFIX(Enum):
    mapping = "map"
    pred = "prd"
    dea = "dea"
    group = "grp"
    workflow = "wf"


class EnumCustom(Enum):
    @classmethod
    def get_options(cls) -> None:
        return [x for x in cls.OPTIONS.value]


class METHODS(EnumCustom):
    _MAP_METHODS = dict(
        random=mmet.RandomMap,
        max_corr=mmet.ArgMaxCorrMap,
        tangram_v1=mmet.TangramV1Map,
        tangram_v2=mmet.TangramV2Map,
        CeLEry=mmet.CeLEryMap,
    )

    MAP_METHODS = {
        PREFIX.mapping.value + "_" + key: val for key, val in _MAP_METHODS.items()
    }

    _PRD_METHODS = dict(
        tangram_v1=pmet.TangramV1Pred,
        tangram_v2=pmet.TangramV2Pred,
    )

    PRD_METHODS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METHODS.items()
    }

    _GRP_METHODS = dict(threshold=gmet.ThresholdGroup)

    GRP_METHODS = {
        PREFIX.group.value + "_" + key: val for key, val in _GRP_METHODS.items()
    }

    _DEA_METHODS = dict(scanpy=dmet.ScanpyDEA)

    DEA_METHODS = {
        PREFIX.dea.value + "_" + key: val for key, val in _DEA_METHODS.items()
    }

    OPTIONS = MAP_METHODS | PRD_METHODS | GRP_METHODS | DEA_METHODS


class WORKFLOWS(EnumCustom):
    import eval.workflows as wf

    _OPTIONS = dict(hejin=wf.HejinWorkflow)

    OPTIONS = {PREFIX.workflow.value + "_" + key: val for key, val in _OPTIONS.items()}


class METRICS(EnumCustom):
    _MAP_METRICS = dict(
        jaccard=mtx.MapJaccardDist,
        accuracy=mtx.MapAccuracy,
        rmse=mtx.MapRMSE,
    )

    MAP_METRICS = {
        PREFIX.mapping.value + "_" + key: val for key, val in _MAP_METRICS.items()
    }

    _PRD_METRICS = dict()

    PRD_METRICS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METRICS.items()
    }

    _DEA_METRICS = dict(
        hypergeom=mtx.DEAHyperGeom,
        auc=mtx.DEAAuc,
    )

    DEA_METRICS = {
        PREFIX.dea.value + "_" + key: val for key, val in _DEA_METRICS.items()
    }

    DEV_METRICS = dict(dev_print=mtx.PrintMetric)

    OPTIONS = MAP_METRICS | PRD_METRICS | DEV_METRICS | DEA_METRICS


class PREPROCESS(EnumCustom):
    OPTIONS = dict(
        standard_scanpy=pp.StandardScanpy,
        normalize_totaly=pp.NormalizeTotal,
        CeLEry=pp.CeLEryPP,
        tangramv1=pp.StandardTangramV1,
        tangramv2=pp.StandardTangramV2,
    )
