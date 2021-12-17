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

#Function to Perform K Fold Validation
def kFoldVal(X, Y, classifier, k):
	kf = KFold(n_splits=k)
	accSum = 0.0
	for train_index, test_index in kf.split(X):
		x_train, x_test = X[train_index], X[test_index]
		y_train, y_test = Y[train_index], Y[test_index]
		classifier.fit(x_train, y_train)
		y_pred = classifier.predict(x_test)
		accSum += accuracy_score(y_test,y_pred)*100
	
	return float(accSum/float(k))

#----- K Fold Cross Validation Function Ends	

def main():
#Data Cleaning

	missing_values = ["n/a", "na", "--", "?"]
	data = pd.read_csv('../dataset_diabetes/diabetic_data.csv', delimiter=',', na_values = missing_values)

	data_cleaning = DataCleaning()
	data = data_cleaning.clean_columns(data, missing_bound=0.2)

	colsMissingValues = data_cleaning.get_cols_having_missing_values(data, False)
	data = data_cleaning.fill_missing_values(data, colsMissingValues)

#Data Cleaning Done

	data = data.to_numpy()

	le = LabelEncoder()

	for i in range(50):
		if isinstance(data[0][i], str):
			data[:,i] = le.fit_transform(data[:,i])
	
	print(data)
	print(data.shape)

	X_train, X_test = data[0:80000,0:49], data[80000:101766,0:49]
	Y_train, Y_test = data[0:80000,49:50], data[80000:101766,49:50]
	Y_train, Y_test = Y_train.astype('int'), Y_test.astype('int')

	print(X_train)
	print(X_train.shape)
	print(Y_train)
	print(Y_train.shape)
    

	grid_params = {
		'criterion': ['gini','entropy'],
		'splitter': ['best','random'],
	    'max_depth': [2,4,6],
	    'min_samples_leaf': [0.02, 0.04],
	    'min_samples_split': [0.2,0.5,0.8]
	}

	dt = DecisionTreeClassifier(random_state=50)

	# Builds a model for each possible combination of all the hyperparamter values provided using cv = 5 (5 fold cross validation)
	# cv = 5, builds a 5 fold cross validated GridSearch Object
	# Set scoring parameter as accuracy as we choose the best model based on the accuracy value
	grid_object = GridSearchCV(estimator = dt, param_grid = grid_params, scoring = 'accuracy', cv = 5, n_jobs = -1)

	print "\nHyper Parameter Tuning Begins\n"
	# fit grid object to the training data
	grid_object.fit(X_train, Y_train)

	print "\n\nBest Param Values \t\t\n\n"
	print(grid_object.best_params_)

	#---- Hyper Parameter Tuning Ends ----


	#----- Reporting Accuracy on Test Set using Model with Best Parameters learned through Hyper Parameter Tuning -----------

	#Building Decision Tree With Best Parameters learned from Hyper Parameter Tuning
	best_params = grid_object.best_params_
	dt = DecisionTreeClassifier(criterion=best_params['criterion'],splitter=best_params['splitter'],max_depth=best_params['max_depth'],min_samples_leaf=best_params['min_samples_leaf'],min_samples_split=best_params['min_samples_split'],random_state=50)


	#dt = DecisionTreeClassifier(criterion='gini')
	dt.fit(X_train, Y_train)
	Y_pred = dt.predict(X_test)

	print "Accuracy score Test = ", accuracy_score(Y_test,Y_pred)*100

	print "Accuracy score 5-Fold = ", kFoldVal(X_train, Y_train, dt, 5)
	


if __name__ == "__main__":
	main()	