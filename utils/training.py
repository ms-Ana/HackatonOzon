from catboost import CatBoostClassifier
import pandas as pd

def train(catboost_params, train_pool, fit_params, save_path):
    model = CatBoostClassifier(**catboost_params)
    model.fit(train_pool, **fit_params)
    model.save_model(save_path)
    return model
    
def create_3cat_grouped(features: pd.DataFrame) -> pd.DataFrame:
    cat3_counts = features["cat_31"].value_counts().to_dict()
    cntr = 0
    for cat3 in cat3_counts:
        if cat3_counts[cat3] < 1_000:
            cntr += cat3_counts[cat3]
    features["cat3_grouped"] = features["cat_31"].apply(lambda x: x if cat3_counts[x] > 1000 else "rest")
    return features