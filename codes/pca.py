
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from util import Util


class MY_PCA:
    def __init__(self, df=None):
        self.df = df

    def do_n_pca(self, df, n_components=2):
        # copies the object type columns
        df = df.select_dtypes(include=['object']).copy()
        df_without_class_attr = df.drop(columns=["diabetesMed", "readmitted"])
        print("doing one-hot-encoding of the categorical columns of the df... ...")
        # convert the object type columns into 1-hot-encoded dataframe, so will increase the column numbers
        df_one_hot = pd.get_dummies(
            df_without_class_attr, columns=df_without_class_attr.columns)
        # print(df_one_hot.head())
        # print(df_one_hot.shape)
        print("doing standerdization... ...")
        df_standardized = pd.DataFrame(StandardScaler().fit_transform(
            df_one_hot), index=df_one_hot.index, columns=df_one_hot.columns)
        # print(df_standardized.head())
        pca = PCA(n_components=n_components)
        print("doing pca over standardized df... ...")
        principalComponents = pca.fit_transform(df_standardized)
        pca_col_names = []
        for i in range(n_components):
            pca_col_names.append("Principal_Component_" + str(i))
        principalDf = pd.DataFrame(
            data=principalComponents, columns=pca_col_names)
        print("concatinating principal df to class attributes... ...")
        finalDf = pd.concat([principalDf, df["readmitted"]], axis=1)
        print("plottin 2 principal component vs readmitted class... ...")

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal_Component_1', fontsize=15)
        ax.set_ylabel('Principal_Component_2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        targets = ['NO', '>30', '<30']
        colors = ['r', 'g', 'b']
        for target, color in zip(targets, colors):
            indicesToKeep = finalDf['readmitted'] == target
            ax.scatter(finalDf.loc[indicesToKeep, 'Principal_Component_0'],
                       finalDf.loc[indicesToKeep, 'Principal_Component_1'], c=color, s=50)
        ax.legend(targets)
        ax.grid()
        plt.show()
        # pc0, pc1 = [0.001563   0.00125708] captures .1563% and .1257% variance
        print(pca.explained_variance_ratio_)

    def do_pca_with_one_hot_encoding(self, df, keep_variance=.80):
        # copies the object type columns
        # df = df.select_dtypes(include=['object']).copy()
        my_util = Util()
        df_without_class_attr, df_class_attr = my_util.separate_class_cols(df, [
                                                                           "readmitted"])
        print(df_without_class_attr.head())
        print(df_class_attr.head())
        print("doing one-hot-encoding of the all columns of the df... ...")
        df_one_hot = pd.get_dummies(
            df_without_class_attr, columns=df_without_class_attr.columns)
        # print(df_one_hot.head())
        print("after 1-hot-encoing shape: ", df_one_hot.shape)
        print("doing standerdization... ...")
        df_standardized = pd.DataFrame(StandardScaler().fit_transform(
            df_one_hot), index=df_one_hot.index, columns=df_one_hot.columns)
        # # print(df_standardized.head())
        pca = PCA(keep_variance)
        # #principalComponents = pca.fit(df_standardized)
        print("transforing the standardized DF into pca DF... ...")
        principalComponents = pca.fit_transform(df_standardized)
        pca_col_names = []
        for i in range(pca.n_components_):
            pca_col_names.append("pc_" + str(i))
        principalDf = pd.DataFrame(
            data=principalComponents, columns=pca_col_names)
        # #print(pca.n_components_)
        # #print(pca.explained_variance_)
        # print("saving explained variance into file ... ...")
        # a = np.asarray(pca.explained_variance_)
        # np.savetxt(
        #     "../only_calculated_datasets/explained_variance.csv", a, delimiter=",")

        return my_util.concat_multi_cols([principalDf, df_class_attr])
