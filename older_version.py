import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# load the gousing.csv file 

data = pd.read_csv('housing.csv')

# lets do stratified shuffle split based on income_category

data["income_cat"] = pd.cut(data["median_income"],
                            bins = [0,1.5,3.0,4.5,6.0,np.inf],
                            labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index , test_index in split.split(data, data["income_cat"]):
    strat_train_set = data.loc[train_index].drop("income_cat" , axis = 1)
    strat_test_set = data.loc[test_index].drop("income_cat" , axis = 1)

# now lets work with test set

housing = strat_test_set.copy()

# lets separate the predictors and labels

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# print(housing)

# lets make num_attributes and cat_attributes
num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributes =["ocean_proximity"]


# lets create pipeline

# numberical pipeline

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
]) 
# 9880116361 - suresh...
# categorical pipeline

cat_pipeline = Pipeline([
    ('one_hot' , OneHotEncoder(handle_unknown='ignore'))
])


# full pipeline

full_pipeline = ColumnTransformer([
    ('num' , num_pipeline, num_attributes),
    ('cat' , cat_pipeline, cat_attributes)
])

# now pipeline is done now final transform and known as prepared

housing_prepared = full_pipeline.fit_transform(housing)

# print(housing_prepared)


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# Decision Tree
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)


# Random Forest
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# now lets predict

lin_preds = lin_reg.predict(housing_prepared)
tree_preds = tree_reg.predict(housing_prepared)
forest_preds = forest_reg.predict(housing_prepared)

# Calculate RMSE
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
tree_rmse = root_mean_squared_error(housing_labels, tree_preds)
forest_rmse = root_mean_squared_error(housing_labels, forest_preds)

print(f"Linear Regression RMSE: {lin_rmse}")
print(f"Decision Tree RMSE: {tree_rmse}")
print(f"Random Forest RMSE: {forest_rmse}")


from sklearn.model_selection import cross_val_score
import pandas as pd

# Evaluate Decision Tree with cross-validation
tree_rmses = -cross_val_score(
    tree_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

# WARNING: Scikit-Learn’s scoring uses utility functions (higher is better), so RMSE is returned as negative.
# We use minus (-) to convert it back to positive RMSE.
print("Decision Tree CV RMSEs:", tree_rmses)
print("\nCross-Validation Performance (Decision Tree):")
print(pd.DataFrame(tree_rmses).describe())








# Evaluate randforest with cross-validation
forest_rmses = -cross_val_score(
    forest_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)


print("Decision Tree CV RMSEs:", forest_rmses)
print("\nCross-Validation Performance (random forest):")
print(pd.DataFrame(forest_rmses).describe())









# Evaluate randforest with cross-validation
linear_rmses = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)


print("Decision Tree CV RMSEs:", linear_rmses)
print("\nCross-Validation Performance (linear regression ):")
print(pd.DataFrame(linear_rmses).describe())

# now since radnomforest has least rmse hence will be trained on that among these..
