
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from util import Util
from my_statistics import My_Statistics


class My_SVM():
    def __init__(self, kernel='poly', gamma='scale', degree=3, shrinking=True, args=None):
        self.clf = SVC(kernel=kernel, gamma=gamma)
        self.args = args

    def train_validate_test(self, x_train_df, y_train, x_test_df, y_test, target_names=["0", "1", "2"]):
        """
        Overviews
        Params
            x_train_df: Expect a dataframe of features of train set. e.g. the dimension can be (81412, 1666)
            y_train: Expect a (xxxx, ) dimensional array, not a column vector. e.g. (81412,)
            x_test_df: Expect a dataframe of features of test set. e.g. the dimension can be (20354, 1666)
            y_test: Expect a (yyyy, ) dimensional array, not a column vector. e.g. (20354,)
            kernel: The kernel funtion used in classifier. ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
                    ‘precomputed’ or a callable. We can not use 'linear'.
            cv: integer
                Cross validation number
            target_names = ["0", "1", "2"]
        """
        clf = self.clf
        print("svm training ... ...")
        clf.fit(x_train_df, y_train)
        y_pred = clf.predict(x_train_df)
        my_statistics = My_Statistics()
        my_statistics.print_all_statistics(
            y_true=y_train, y_pred=y_pred, target_names=target_names)
        my_statistics.print_mean_accuracy(
            classifier=clf, x_df=x_train_df, y_array=y_train)
        print("svm testing ... ...")
        y_pred = clf.predict(x_test_df)
        my_statistics = My_Statistics()
        my_statistics.print_all_statistics(
            y_true=y_test, y_pred=y_pred, target_names=target_names)
        my_statistics.print_mean_accuracy(
            classifier=clf, x_df=x_test_df, y_array=y_test)

    def k_fold_cross_validation(self, x, y, k=5):
        print("running k-fold cross validation using svm ... ...")
        clf = self.clf
        scores = cross_val_score(clf, x, y, scoring='recall_macro', cv=k)
        print(scores)

    def train_by_LinearSVC(self, x_train_df, y_train, x_test_df, y_test, kernel='poly'):
        """
        Overviews
        Params
            x_train_df: Expect a dataframe of features of train set. e.g. the dimension can be (81412, 1666)
            y_train: Expect a (xxxx, ) dimensional array, not a column vector. e.g. (81412,)
            x_test_df: Expect a dataframe of features of test set. e.g. the dimension can be (20354, 1666)
            y_test: Expect a (yyyy, ) dimensional array, not a column vector. e.g. (20354,)
            kernel: The kernel funtion used in classifier. ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’,
                    ‘precomputed’ or a callable. We can not use 'linear'.
        """
        clf = LinearSVC(
            random_state=0, tol=1e-5)  # same as SVC(kernel='linear')
        print("svm training ... ...")
        svm_model = clf.fit(x_train_df, y_train)
        print("smv model: ", svm_model)
        print("svm testing ... ...")
        y_pred = clf.predict(x_test_df)
        print(confusion_matrix(y_test, y_pred))
