import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from visualization import Visualization
from data_cleaning import DataCleaning

missing_values = ["n/a", "na", "--", "?"]
data = pd.read_csv('../dataset_diabetes/diabetic_data.csv', delimiter=',', na_values = missing_values)

data_cleaning = DataCleaning()
data = data_cleaning.clean_columns(data, missing_bound=0.2)

colsMissingValues = data_cleaning.get_cols_having_missing_values(data, False)
data = data_cleaning.fill_missing_values(data, colsMissingValues)

"""
data = data.values
features = []
for i in range(50):
	if isinstance(data[0][i], str):
		a = np.unique(data[:,i])
		features.append(a)
"""		

data = data.to_numpy()

print(data)
print(data.shape)

onehotencoder = OneHotEncoder(categories='auto')
data = onehotencoder.fit_transform(data).toarray()

print(data)
print(data.shape)		


