from . import methods as met
from . import metrics as mtx
from enum import Enum


class PREFIX(Enum):
    mapping = "map"
    pred = "prd"


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
