from typing import Any, Dict
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def make_model(name: str, params: Dict[str, Any]):
    if name == "lgbm":
        return lgb.LGBMClassifier(**params)
    if name == "xgb":
        return XGBClassifier(**params)
    if name == "cat":
        return CatBoostClassifier(verbose=0, **params)
    raise ValueError(f"Unknown model: {name}")