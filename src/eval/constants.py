from enum import Enum

# from . import methods as met

import eval.map_methods as mmet
import eval.pred_methods as pmet
import eval.dea_methods as dmet

from . import metrics as mtx
from . import preprocess as pp


class PREFIX(Enum):
    mapping = "map"
    pred = "prd"


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

    _PRD_METHODS = {}

    PRD_METHODS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METHODS.items()
    }

    OPTIONS = MAP_METHODS | PRD_METHODS


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

    OPTIONS = MAP_METRICS | PRD_METRICS


class PREPROCESS(EnumCustom):
    OPTIONS = dict(
        standard_scanpy=pp.StandardScanpy,
        normalize_totaly=pp.NormalizeTotal,
        CeLEry=pp.CeLEryPP,
    )

