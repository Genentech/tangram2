from . import methods as met
from . import metrics as mtx
from enum import Enum


class METHODS(Enum):

    _MAP_METHODS = dict(
        random=met.RandomMap,
        max_corr=met.ArgMaxCorrMap,
        tangram_v1=met.TangramV1Map,
        tangram_v2=met.TangramV2Map,
    )

    _PRD_METHODS = dict()

    METHODS = dict(mapping = _MAP_METHODS,
                   pred = _PRD_METHODS,
                   )

class METRICS(Enum):

    _MAP_METRICS = dict(jaccard = mtx.MapJaccardDist,
                        accuracy = mtx.MapAccuracy,
                        )

    _PRD_METRICS = dict()

    METRICS = dict(mapping = _MAP_METRICS,
                   pred = _PRD_METRICS,
                   )
