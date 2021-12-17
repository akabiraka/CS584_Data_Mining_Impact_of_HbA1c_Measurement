
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns


class Visualization:
    def __init__(self, df=None):
        self.df = df

    def hello(self):
        return "Alhumdulillah, hello"

    def save_and_show_plot(self, path=None, is_show=True):
        if path:
            plt.savefig(path)
        if is_show:
            plt.show()

    def make_histograms(self, df, columns=None, figsize=(15, 15), cmap=None, image_path=None, is_show=True):
        """
        This function make histograms for each column name. Each of the column should be 'Nominal'
        Parameters
        ----------
        df: pandas dataframe
            Pandas dataframe associated with column names.
        columns: dataframe column names in array
            e.g. ['aColumn', 'bColumn']
        figsize: default=(15, 15)
        cmap: colormap codes for for custom colored bar or histogram
            defautls are:
                summer or green for numerical
                winter or blue for categorical
        Effects
        -------
        Creates columns.size() number of histograms.
        """
        nrows = 3
        ncols = 3
        plt.figure(figsize=figsize)
        for i, col in enumerate(df[columns]):
            plt.subplot(nrows, ncols, i + 1)
            if is_numeric_dtype(df[col]):
                df[col].plot(kind="hist", colormap=(
                    cmap or "summer")).set_title(col)
            else:
                df[col].value_counts().plot(
                    kind="bar", colormap=(cmap or "winter")).set_title(col)

        plt.subplots_adjust(top=.95, bottom=.05, hspace=.5, wspace=0.4)

        self.save_and_show_plot(image_path, is_show)

    def build_scatter_2attr_plot(self, df, x_col, y_col, figsize=(20, 20), cmap=None, image_path=None, is_show=True):
        # for i, col in enumerate(columns):
        df_normalized = pd.DataFrame({
            x_col: (df[x_col] - df[x_col].mean()) / df[x_col].std(),
            y_col: (df[y_col] - df[y_col].mean()) / df[y_col].std()
        })
        ax = df_normalized.plot(x=x_col, y=y_col, style='o')
        #ax = df.plot(x=x_col, y=y_col, style='o')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        self.save_and_show_plot(image_path, is_show)

    def build_3d_scatter_plot(self, df, x_axis_col, y_axis_col, z_axis_col, class_col, figsize=(30, 30), cmap=None, image_path=None, is_show=True):
        fig = px.scatter_3d(df, x=x_axis_col, y=y_axis_col, z=z_axis_col,
                            color=class_col, symbol=class_col, opacity=0.7)
        fig.update_layout(margin=dict(l=65, r=50, b=65, t=90))  #
        fig.show()
        self.save_and_show_plot(image_path, is_show)

    def make_boxplot(self, df, x, y, image_path):
        sns.boxplot(x=x, y=y, data=df)
        self.save_and_show_plot(image_path, is_show=True)
