
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import math

class DataCleaning():
    def __init__(self):
        pass

    def clean_columns(self, data, missing_bound=.5):
        """
        This function remove those columns whose missing value(in percentage) >= missing_bound
        data: the given data
        missing_bound: in percent (0, 1)
        """
        tmp_data = data
        for col in data.columns:
            missing_values_per_column = data[col].isnull().sum()
            total_rows = data[col].shape[0]
            missing_percentage = missing_values_per_column/total_rows

            if (missing_percentage >= missing_bound):
                print("Removing column: ", col, ", Missing value: ", missing_percentage*100)
                data = data.drop(columns=col)
        print("Number of columns removed: ", tmp_data.shape[1]-data.shape[1])
        return data


    # In[4]:


    def get_cols_having_missing_values(self, data, show_non_missing_columns=False):
        """
        Prints number of missing values in percentage columnwise.
        If show_non_missing_columns=False, it only shows columns that has at least 1 missing value.
        else shows all values.

        return the cols that has some missing columns
        """
        cols = []
        for col in data.columns:
            missing_values_per_column = data[col].isnull().sum()
            total_rows = data[col].shape[0]
            missing_percentage = float(missing_values_per_column)/float(total_rows)
            if show_non_missing_columns==False:
                if missing_percentage>0.0:
                    print("Column: ", col, ", Missing value: ", missing_percentage*100)
                    cols.append(col)
            else:
                print("Column: ", col, ", Missing value: ", missing_percentage*100)
                cols.append(col)

        return cols


    # In[5]:


    def fill_missing_values(self, data, columns=None):
        """
        columns: columns names in array format. e.g ['a', 'b', 'c']
        If data_type is 'object or string', fill the missing values with the mode.
        If data_type is 'numerical', fill the missing values with the mean.
        """
        for col in columns:
            if is_string_dtype(data[col]):
                data[col].fillna(data[col].mode()[0], inplace=True) #Pandas 0.24.0+ does not count NaN by default as mode value.
            elif is_numeric_dtype(data[col]):
                data[col].fillna(data[col].mean()[0], inplace=True) #fill the missing values with mean if the column is numeric data type

        return data

    def just_remove_columns(self, df, columns=None):
        df = df.drop(columns=columns)
        return df
