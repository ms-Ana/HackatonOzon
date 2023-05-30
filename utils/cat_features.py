import pandas as pd
import ast
from tqdm import tqdm

def cat_features(features: pd.DataFrame):
    for i, row in tqdm(features.iterrows()):
        if row["characteristic_attributes_mapping"] is not None:
            cat_map = ast.literal_eval(row["characteristic_attributes_mapping"])
            cur_list = [f"{k}_{'_'.join(v)}" for k, v in cat_map.items()]
            features.at[i, "characteristic_attributes_mapping"]  = cur_list
    return features


