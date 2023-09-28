from enum import Enum

# from . import methods as met
import eval.methods as met

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
        random=met.RandomMap,
        max_corr=met.ArgMaxCorrMap,
        tangram_v1=met.TangramV1Map,
        tangram_v2=met.TangramV2Map,
        CeLEry=met.CeLEryMap,
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


class CeLEry(Enum):
    x_coord = "x_pixel"
    y_coord = "y_pixel"
    filename = "model"
    spatial_key = "spatial"
