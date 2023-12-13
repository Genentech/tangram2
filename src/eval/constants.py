from enum import Enum

import eval.dea_methods as dmet
import eval.grp_methods as gmet
import eval.map_methods as mmet
import eval.pred_methods as pmet

from . import metrics as mtx
from . import preprocess as pp


class PREFIX(Enum):
    # prefixes for each method
    mapping = "map"
    pred = "prd"
    dea = "dea"
    group = "grp"
    workflow = "wf"


class EnumCustom(Enum):
    # custom enum class that returns all
    # the options
    @classmethod
    def get_options(cls) -> list:
        return [x for x in cls.OPTIONS.value]


class METHODS(EnumCustom):
    # raw names of map methods
    _MAP_METHODS = dict(
        random=mmet.RandomMap,
        max_corr=mmet.ArgMaxCorrMap,
        tangram_v1=mmet.TangramV1Map,
        tangram_v2=mmet.TangramV2Map,
        CeLEry=mmet.CeLEryMap,
        SpaOTsc=mmet.SpaOTscMap,
    )

    # "prefixed" names of map methods
    MAP_METHODS = {
        PREFIX.mapping.value + "_" + key: val for key, val in _MAP_METHODS.items()
    }

    # raw names of prediction methods
    _PRD_METHODS = dict(
        tangram_v1=pmet.TangramV1Pred,
        tangram_v2=pmet.TangramV2Pred,
    )

    # "prefixed" names of predictions methods
    PRD_METHODS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METHODS.items()
    }

    # raw names of group methods
    _GRP_METHODS = dict(threshold=gmet.ThresholdGroup)

    # "prefixed" names of group methods
    GRP_METHODS = {
        PREFIX.group.value + "_" + key: val for key, val in _GRP_METHODS.items()
    }

    # raw names of DEA methods
    _DEA_METHODS = dict(scanpy=dmet.ScanpyDEA)

    # prefixed names of DEA methods
    DEA_METHODS = {
        PREFIX.dea.value + "_" + key: val for key, val in _DEA_METHODS.items()
    }

    # all available methods
    OPTIONS = MAP_METHODS | PRD_METHODS | GRP_METHODS | DEA_METHODS


class WORKFLOWS(EnumCustom):
    import eval.workflows as wf

    # raw workflow names
    _OPTIONS = dict(hejin=wf.HejinWorkflow)

    # "prefixed" workflow names
    OPTIONS = {PREFIX.workflow.value + "_" + key: val for key, val in _OPTIONS.items()}


class METRICS(EnumCustom):

    # raw map metrics names
    _MAP_METRICS = dict(
        jaccard=mtx.MapJaccardDist,
        accuracy=mtx.MapAccuracy,
        rmse=mtx.MapRMSE,
    )

    # "prefixed" map metrics names
    MAP_METRICS = {
        PREFIX.mapping.value + "_" + key: val for key, val in _MAP_METRICS.items()
    }

    # raw prediction metrics names
    _PRD_METRICS = dict()

    # "prefixed" prediction metrics names
    PRD_METRICS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METRICS.items()
    }

    # raw dea metrics names
    _DEA_METRICS = dict(
        hypergeom=mtx.DEAHyperGeom,
        auc=mtx.DEAAuc,
    )

    # "prefixed" dea metrics names
    DEA_METRICS = {
        PREFIX.dea.value + "_" + key: val for key, val in _DEA_METRICS.items()
    }

    # metrics to use in development
    DEV_METRICS = dict(dev_print=mtx.PrintMetric)

    # all available metrics
    OPTIONS = MAP_METRICS | PRD_METRICS | DEV_METRICS | DEA_METRICS


class PREPROCESS(EnumCustom):
    # preprocessing options
    OPTIONS = dict(
        standard_scanpy=pp.StandardScanpy,
        normalize_totaly=pp.NormalizeTotal,
        CeLEry=pp.CeLEryPP,
        tangramv1=pp.StandardTangramV1,
        tangramv2=pp.StandardTangramV2,
        SpaOTsc=pp.StandardSpaOTsc,
    )
