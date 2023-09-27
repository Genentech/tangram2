from enum import Enum

# from . import methods as met
import eval.methods as met

from . import metrics as mtx
from . import preprocess as pp


class PREFIX(Enum):
    mapping = "map"
    pred = "prd"


# TODO: update methods with CeLEry pointing to met.CeLEryMap
class METHODS(Enum):
    _MAP_METHODS = dict(
        random=met.RandomMap,
        max_corr=met.ArgMaxCorrMap,
        tangram_v1=met.TangramV1Map,
        tangram_v2=met.TangramV2Map,
    )

    MAP_METHODS = {
        PREFIX.mapping.value + "_" + key: val for key, val in _MAP_METHODS.items()
    }

    _PRD_METHODS = {}

    PRD_METHODS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METHODS.items()
    }

    METHODS = MAP_METHODS | PRD_METHODS


#TODO: update metrics with rmse pointing to mtx.MapRMSE
class METRICS(Enum):
    _MAP_METRICS = dict(
        jaccard=mtx.MapJaccardDist,
        accuracy=mtx.MapAccuracy,

    )

    MAP_METRICS = {
        PREFIX.mapping.value + "_" + key: val for key, val in _MAP_METRICS.items()
    }

    _PRD_METRICS = dict()

    PRD_METRICS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METRICS.items()
    }

    METRICS = MAP_METRICS | PRD_METRICS


#TODO: update with CeLERry
class PreProcess(Enum):
    RECIPES = dict(
        standard_scanpy=pp.StandardScanpy,
        normalize_totaly=pp.NormalizeTotal,
    )

