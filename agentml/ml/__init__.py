"""Machine-learning engine: profiling, feature engineering, model zoo, deep learning, evaluation."""

from agentml.ml.deep import DeepTabularClassifier, DeepTabularRegressor, deep_info
from agentml.ml.evaluation import PRIMARY_METRIC, evaluate, feature_importance
from agentml.ml.features import FeaturePlan, build_preprocessor, propose_feature_plan
from agentml.ml.models import default_candidates, model_zoo
from agentml.ml.profiling import infer_target, infer_task, load_dataset, profile_dataframe

__all__ = [
    "PRIMARY_METRIC",
    "DeepTabularClassifier",
    "DeepTabularRegressor",
    "FeaturePlan",
    "build_preprocessor",
    "deep_info",
    "default_candidates",
    "evaluate",
    "feature_importance",
    "infer_target",
    "infer_task",
    "load_dataset",
    "model_zoo",
    "profile_dataframe",
    "propose_feature_plan",
]
