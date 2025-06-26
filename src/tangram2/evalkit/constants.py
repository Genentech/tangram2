from enum import Enum

import tangram2.evalkit.methods.map_methods as mmet
import tangram2.evalkit.methods.pred_methods as pmet
from tangram2.evalkit.evaluation import metrics as mtx
from tangram2.evalkit.methods import preprocess as pp
from tangram2.evalkit.methods import save_methods as sm


class CONF(Enum):
    data = "data"
    wfs = "workflows"
    metrics = "metrics"
    pp = "preprocess"
    recipe = "recipe"
    params = "params"
    eval = "evaluation"
    save = "save"


class PREFIX(Enum):
    # prefixes for each method
    mapping = "map"
    pred = "prd"
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
        tangram_v1=mmet.Tangram1Map,
        tangram_v2=mmet.Tangram2Map,
        celery=mmet.CeLEryMap,
        spaotsc=mmet.SpaOTscMap,
        moscot=mmet.MoscotMap,
    )

    # "prefixed" names of map methods
    MAP_METHODS = {
        PREFIX.mapping.value + "_" + key: val for key, val in _MAP_METHODS.items()
    }

    # raw names of prediction methods
    _PRD_METHODS = dict(
        tangram_v1=pmet.TangramV1Pred,
        tangram_v2=pmet.TangramV2Pred,
        moscot=pmet.MoscotPred,
    )

    # "prefixed" names of predictions methods
    PRD_METHODS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METHODS.items()
    }

    # all available methods
    OPTIONS = MAP_METHODS | PRD_METHODS 


class METRICS(EnumCustom):

    # raw map metrics names
    _MAP_METRICS = dict(
        jaccard=mtx.MapJaccardDist,
        accuracy=mtx.MapAccuracy,
        rmse=mtx.MapRMSE,
        f1=mtx.MapF1,
    )

    # "prefixed" map metrics names
    MAP_METRICS = {
        PREFIX.mapping.value + "_" + key: val for key, val in _MAP_METRICS.items()
    }

    # raw prediction metrics names
    _PRD_METRICS = dict(
        loov=mtx.PredLeaveOutScore,
    )

    # "prefixed" prediction metrics names
    PRD_METRICS = {
        PREFIX.pred.value + "_" + key: val for key, val in _PRD_METRICS.items()
    }

    # metrics to use in development
    DEV_METRICS = dict(dev_print=mtx.PrintMetric)

    # all available metrics
    OPTIONS = MAP_METRICS | PRD_METRICS | DEV_METRICS 


class PREPROCESS(EnumCustom):
    # preprocessing options
    OPTIONS = dict(
        standard_scanpy=pp.StandardScanpy,
        normalize_total=pp.NormalizeTotal,
        celery=pp.CeLEryPP,
        tangram1=pp.StandardTangram1,
        tangram2=pp.StandardTangram2,
        spaotsc=pp.StandardSpaOTsc,
        moscot=pp.StandardMoscot,
    )
