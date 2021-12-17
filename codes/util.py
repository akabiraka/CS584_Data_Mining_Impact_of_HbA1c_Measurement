
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class Util:
    def __init__(self, df=None):
        self.df = df

    def save_df(self, df, path):
        """
        Overviews:
            Save a pandas dataframe to the specified path as pickle(pkl) extension
        Params:
            df: dataframe
            path: string
                path to save df. e.g. ../path_to_save/name.pkl
        """
        df.to_pickle(path)

    def load_df(self, path):
        """
        Overviews:
            Read a pickle file as pandas dataframe.
        Params:
            path: string
                path to read pickle as df. e.g. ../path_to_save/name.pkl
        """
        df = pd.read_pickle(path)
        return df

    def do_train_test_split(self, df, class_name, test_size=0.2, is_y_col_vector=False):
        """
        Overviews:
            Split the pandas dataframe into train set dataframe, train label array, test set dataframe and test label array based on test_size
        Params:
            df: dataframe
            class_name: string
                a class name is allowed. e.g. "class_name"
            test_size: float
                test_set will consists of test_sixe% of data. And (1-test_size)% data will be train set
        Returns:
            x_train_df: dataframe
            y_train: array
            x_test_df: dataframe
            y_test: array
        """
        Y_df = df[[class_name]]
        X_df = df.drop(columns=[class_name])
        x_train_df, x_test_df, y_train_df, y_test_df = train_test_split(
            X_df, Y_df, test_size=test_size)
        if is_y_col_vector:
            print("Dataset split shapes: ", x_train_df.shape,
                  y_train_df.shape, x_test_df.shape, y_test_df.shape)
            return x_train_df, y_train_df, x_test_df, y_test_df
        else:
            print("Dataset split shapes: ", x_train_df.shape, y_train_df.values.ravel(
            ).shape, x_test_df.shape, y_test_df.values.ravel().shape)
            return x_train_df, y_train_df.values.ravel(), x_test_df, y_test_df.values.ravel()

    def do_train_validation_test_split(self, df, class_name, train_size=0.5, validation_size=0.3, is_y_col_vector=False):
        """
            Params:
                df: pandas dataframe
                    the dataframe to be split.
                class_name: string
                    a column of the df which will be Y (predicted class lebel) set. e.g. "class_name".
                train_size: float
                    Default 50%. % of data will be the train set.
                validation_size: float
                    Default 30%. % of the data will be validation set.
                1-(train_size+validation_size) will be the test size. Default 20%.
                is_y_col_vector: boolean
                    if true, class labels will be returned as column vector, else row array. sklearn confusion_matrix and other
                    statistics expects Y as array, so default is False
        """
        validation_size = validation_size * 2
        Y_df = df[[class_name]]
        X_df = df.drop(columns=[class_name])
        x_others_df, x_train_df, y_others_df, y_train_df = train_test_split(
            X_df, Y_df, test_size=train_size)
        x_test_df, x_validation_df, y_test_df, y_validation_df = train_test_split(
            x_others_df, y_others_df, test_size=validation_size)
        if is_y_col_vector:
            print("Dataset split shapes: ", x_train_df.shape, y_train_df.shape,
                  x_validation_df.shape, y_validation_df.shape, x_test_df.shape, y_test_df.shape)
            return x_train_df, y_train_df, x_validation_df, y_validation_df, x_test_df, y_test_df
        else:
            print("Dataset split shapes: ", x_train_df.shape, y_train_df.values.ravel().shape,
                  x_validation_df.shape, y_validation_df.values.ravel().shape, x_test_df.shape, y_test_df.values.ravel().shape)
            return x_train_df, y_train_df.values.ravel(), x_validation_df, y_validation_df.values.ravel(), x_test_df, y_test_df.values.ravel()

    def separate_class_cols(self, df, classes):
        """
            classes: ["diabetesMed", "readmitted"]
        """
        class_cols_df = df[classes]
        without_class_cols_df = df.drop(columns=classes)
        print("Dataset split shapes: ",
              without_class_cols_df.shape, class_cols_df.shape)
        return without_class_cols_df, class_cols_df

    def concat_two_dfs(self, df1, df2):
        df1.reset_index(drop=True)
        df2.reset_index(drop=True)
        big_df = pd.concat([df1, df2], axis=1)
        return big_df

    def concat_multi_cols(self, dfs):
        for df in dfs:
            df.reset_index(drop=True)
        big_df = pd.concat(dfs, axis=1)
        return big_df

    def convert_into_numeric_class_attr(self, df):
        df["readmitted"], readmitted_mapped_index = pd.Series(
            df["readmitted"]).factorize()  # readmitted_mapped_index=['NO', '>30', '<30']
        df["diabetesMed"], diabetesMed_mapped_index = pd.Series(
            df["diabetesMed"]).factorize()  # diabetesMed_mapped_index=['No', 'Yes']
        return df, readmitted_mapped_index, diabetesMed_mapped_index

    def convert_into_numeric_class_attr_readmitted(self, df):
        df["readmitted"], readmitted_mapped_index = pd.Series(
            df["readmitted"]).factorize()  # readmitted_mapped_index=['NO', '>30', '<30']
        return df, readmitted_mapped_index

    def one_hot(self, df):
        # making readmitted attribute values yes/no
        df.readmitted.replace(['>30', '<30'], ['YES', 'YES'], inplace=True)
        df_without_class_attr = df.drop(columns=["readmitted"])
        df_only_categorical_attr = df_without_class_attr.select_dtypes(include=[
                                                                       'object']).copy()
        print("convering into one hot ... ...")
        df_one_hot = pd.get_dummies(
            df_without_class_attr, columns=df_only_categorical_attr.columns)
        finalDf = pd.concat([df_one_hot, df["readmitted"]], axis=1)

        print("saving one hot into ''../only_calculated_datasets/categorical_1hot_numerical_asB4_only_yes_no_class.pkl' ... ...")
        self.save_df(
            finalDf, "../only_calculated_datasets/categorical_1hot_numerical_asB4_only_yes_no_class.pkl")

    def one_hot_and_standardize(self, df):
        # making readmitted attribute values yes/no
        df.readmitted.replace(['>30', '<30'], ['YES', 'YES'], inplace=True)
        df_without_class_attr = df.drop(columns=["readmitted"])
        df_only_categorical_attr = df_without_class_attr.select_dtypes(include=[
                                                                       'object']).copy()
        print("convering categorical columns into one hot only ... ...")
        df_one_hot = pd.get_dummies(
            df_without_class_attr, columns=df_only_categorical_attr.columns)
        print(df_one_hot.head())

        print("standardizing from one hot ... ... ")
        df_standardized = pd.DataFrame(StandardScaler().fit_transform(
            df_one_hot), index=df_one_hot.index, columns=df_one_hot.columns)
        finalDf = pd.concat([df_standardized, df["readmitted"]], axis=1)

        print(finalDf.head())
        print("saving standardized data ''../only_calculated_datasets/oneHot_standardized_for_nn.pkl' ... ...")
        self.save_df(
            finalDf, "../only_calculated_datasets/oneHot_standardized_for_nn.pkl")
