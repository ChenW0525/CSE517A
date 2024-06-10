import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from scipy import stats
import seaborn as sns
import os


class model_evaluate:
    def __init__(self, data, target, n_splits = 5, model = RandomForestRegressor(random_state=42)):
        self.data = data
        self.target = target
        self.n_splits = n_splits
        self.model = model

    def apply_k_fold(self, title,random_seed=42):
        kf = KFold(n_splits = self.n_splits, shuffle = True, random_state=random_seed)
        in_sample_errors = []
        out_sample_errors = []
        kth_folds = []

        for i, (train_index, val_index) in enumerate(kf.split(self.data)):
            # print(f"Fold {i}")
            X_train, X_val = self.data.iloc[train_index], self.data.iloc[val_index]
            y_train, y_val = self.target.iloc[train_index], self.target.iloc[val_index]

            self.model.fit(X_train, y_train)
            y_pred_train = self.model.predict(X_train)
            y_pred_val = self.model.predict(X_val)

            in_sample_errors.append(mean_squared_error(y_pred_train, y_train))
            out_sample_errors.append(mean_squared_error(y_pred_val, y_val))
            kth_folds.append(i+1)

        plt.figure(figsize=(10,6))
        plt.plot(kth_folds, in_sample_errors, label='In-Sample Error', marker='o')
        plt.plot(kth_folds, out_sample_errors, label='Out-of-Sample Error', marker='s')
        plt.legend()
        plt.title(title)
        plt.xlabel('Folds')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()

        return in_sample_errors, out_sample_errors


    def kde_zscores_plot(self, errors, label):
        print("Start to plot the kde")
        z_scores = stats.zscore(errors)
        sns.kdeplot(z_scores, label = label)
        plt.xlabel('Z-Score of Errors')

def perform_paired_t_tests(errors_dict):
    models = list(errors_dict.keys())
    p_values = {}
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            model_i_errors = errors_dict[models[i]]
            model_j_errors = errors_dict[models[j]]
            t_stat, p_value = stats.ttest_rel(model_i_errors, model_j_errors)
            p_values[f"{models[i]} vs {models[j]}"] = p_value
    return p_values


if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    data_dir = os.path.join('..', 'data')
    data_path = os.path.join(data_dir, 'df_normalized.csv')
    df = pd.read_csv(data_path)
    target = df.iloc[:, -1]
    features = df.columns[:-1]

    random_forest_regressor = model_evaluate(df[features], target, n_splits=10, model=RandomForestRegressor(random_state=42))
    linear_regressor = model_evaluate(df[features], target, n_splits=10, model = LinearRegression())
    neural_network = model_evaluate(df[features], target, n_splits=10, model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))

    in_sample_errors_random_forest, out_of_sample_errors_random_forest = random_forest_regressor.apply_k_fold('In-Sample vs. Out-of-Sample Errors for random forest', random_seed=42)
    in_sample_errors_linear, out_of_sample_errors_linear = linear_regressor.apply_k_fold('In-Sample vs. Out-of-Sample Errors for linear regression', random_seed=42)
    in_sample_errors_neural_network, out_of_sample_errors_neural_network = neural_network.apply_k_fold('In-Sample vs. Out-of-Sample Errors for neural network', random_seed=42)

    plt.figure(figsize=(10,6))
    random_forest_regressor.kde_zscores_plot(out_of_sample_errors_random_forest, 'Random Forest Regressor')
    linear_regressor.kde_zscores_plot(out_of_sample_errors_linear, "Linear Regressor")
    neural_network.kde_zscores_plot(out_of_sample_errors_neural_network, "Neural Network")
    plt.legend()
    plt.grid(True)
    plt.title('Out-of-Sample Errors KDE of each model')
    plt.show()

    in_sample_errors_dict = {
    'Random Forest': in_sample_errors_random_forest,
    'Linear Regression': in_sample_errors_linear,
    'Neural Network': in_sample_errors_neural_network
    }
    out_of_sample_errors_dict = {
    'Random Forest': out_of_sample_errors_random_forest,
    'Linear Regression': out_of_sample_errors_linear,
    'Neural Network': out_of_sample_errors_neural_network
    }
    
    # Perform a paired t-test comparing the in-sample errors of 3 models
    in_sample_p_values = perform_paired_t_tests(in_sample_errors_dict)
    print("Paired t-test p-values for In-Sample Errors:")
    for pair, p_value in in_sample_p_values.items(): 
        print(pair, ":", p_value)
    
    # Perform a paired t-test comparing the out-sample errors of 3 models
    out_of_sample_p_values = perform_paired_t_tests(out_of_sample_errors_dict)
    print("\nPaired t-test p-values for Out-of-Sample Errors:")
    for pair, p_value in out_of_sample_p_values.items():
        print(pair, ":",p_value)