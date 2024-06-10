import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import Binarizer, KBinsDiscretizer, PowerTransformer, StandardScaler
from sklearn.decomposition import PCA
import os

class Preprocessing:
    def __init__(self, n_components = 2):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components)
        self.small_constant = 1e-6

    def init_info(self, df, head = True, info = False, describe = False):
        if head:
            print(df.head())
        
        if info:
            print(df.info())

        if describe:
            print(df.describe())

    def check_category(self, df):
        if (df.dtypes == "object").any():
            df_dummies = pd.get_dummies(df, columns = df.select_dtypes(include="object").columns)
            return df_dummies
        
        else:
            print("There are no categorical columns")

    def check_missing_value(self, df):
        total_num = []
        missing_col = {}
        for col in df.columns:
            num = np.sum(pd.isnull(df[col]).values)
            total_num.append(num)
            if num != 0 :
                missing_col[col] = num

        # print(missing_col)
        if any(total_num) == False:
            print(f"There aren't any missing values for this dataset")

        else:
            for key, value in missing_col:
                print(f"The {key} column has {value} missing values")

    def standardize(self, df, col_names):
        # To standarize some specific columns
        data_scaled = self.scaler.fit_transform(df[col_names])
        df[col_names] = data_scaled
        return df
    
    def apply_log_transformation(self, df, col_names):
        # The small_constant = 1e-16 is to ensure numerical stability and to handle any zero values.
        df[col_names] = df[col_names].apply(lambda x: np.log1p(x - x.min() + self.small_constant))
    
    def apply_PCA(self, df, col_names, drop = True):
        # To apply PCA for some specific columns (eg.some colinearity columns)
        principal_components = self.pca.fit_transform(df[col_names])
        PC_name = [f'PC{i+1}' for i in range(principal_components.shape[1])]
        pc_df = pd.DataFrame(data=principal_components, columns= PC_name)
        df.drop(columns = col_names, inplace = drop)
        df[PC_name] = pc_df
        return df
        

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    data_dir = os.path.join('.', 'data')
    data_path = os.path.join(data_dir, 'train.csv')
    df = pd.read_csv(data_path)
    target = df.iloc[:, -1]
    features = df.columns[:-1]

    data_preprocessing = Preprocessing()
    data_preprocessing.init_info(df, info = True)
    data_preprocessing.check_category(df)
    data_preprocessing.check_missing_value(df)
    nan_values = df.isna().sum()
    print(nan_values)
    df = data_preprocessing.standardize(df, features)
    # df = data_preprocessing.apply_PCA(df, features)
    data_preprocessing.init_info(df)
    # df.to_csv("data\\df_normalized.csv", index = False)