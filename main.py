import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor


housing_data = pd.read_csv("housing.csv")

MODEL_FILE = "housing_model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def final_pipeline(num_attribute , cat_attribute):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("Scaler" , StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("one_hot_encoder" , OneHotEncoder())
    ])

    full_pipeline = ColumnTransformer([
        ("numerical" , num_pipeline , num_attribute),
        ("categorical" , cat_pipeline , cat_attribute)
    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing_data["income_cat"] = pd.cut(housing_data["median_income"] ,
                                        bins = [0.0,1.5,3.0,4.5,6.0,np.inf],
                                        labels = [1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits = 1 , test_size = 0.2 , random_state = 42)

    for train_index , test_index in split.split(housing_data , housing_data["income_cat"]):
        housing_data.loc[test_index].drop("income_cat" , axis = 1).to_csv("input.csv" , index = False)
        housing = housing_data.loc[train_index].drop("income_cat" , axis = 1)

    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value" , axis = 1)

    num_attribute = housing.drop("ocean_proximity" , axis = 1).columns.to_list()
    cat_attribute = ["ocean_proximity"]

    full_pipeline = final_pipeline(num_attribute , cat_attribute)
    housing_prepared = full_pipeline.fit_transform(housing)

    model = RandomForestRegressor()
    model.fit(housing_prepared , housing_labels)

    # save the model and pipeline 

    joblib.dump(model , MODEL_FILE)
    joblib.dump(full_pipeline , PIPELINE_FILE)

else:

    model = joblib.load(MODEL_FILE)
    full_pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    input_prepared = full_pipeline.transform(input_data)
    predictions = model.predict(input_prepared)

    input_data["median_house_value"] = predictions
    input_data.to_csv("output2.csv", index=False)
    print("Inference complete. Results saved to output.csv")

