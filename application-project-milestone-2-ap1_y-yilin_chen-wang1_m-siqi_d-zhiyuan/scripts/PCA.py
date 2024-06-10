import pandas as pd
import os

if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    data_dir = os.path.join('.', 'data')
    data_path = os.path.join(data_dir, 'df_normalized.csv')
    data = pd.read_csv(data_path)

    # # Load the data from the uploaded CSV file
    # data_path = '/mnt/data/df_normalized.csv'
    # data = pd.read_csv(data_path)
    #
    # # Display the first few rows of the dataset and its column information
    # data.head(), data.info()

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Defining the number of components for PCA
    n_components = 2

    # Preparing the data for PCA (excluding the target variable)
    features = data.drop('target', axis=1)

    # Performing PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(features)

    # Creating a DataFrame with the principal components
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['target'] = data['target']  # Adding the target variable for plotting

    # Plotting the 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['target'], c=pca_df['target'], marker='x', cmap='viridis')

    # Labels and title
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Target Variable')
    ax.set_title('3D Scatter Plot of PCA Components and Target Variable')

    # Color bar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Target Value')

    plt.show()

    pca_full = PCA()
    pca_full.fit(features)

    # Calculating the explained variance and cumulative variance
    explained_variance = pca_full.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    print(cumulative_variance)

    # Finding the minimum number of components with a reconstruction error tolerance of 1e-6
    reconstruction_error_tolerance = 1e-6
    num_components = (cumulative_variance >= (1 - reconstruction_error_tolerance)).argmax() + 1

    print(num_components, cumulative_variance[num_components - 1])

    pca_optimal = PCA(n_components=num_components)
    transformed_data = pca_optimal.fit_transform(features)

    # Creating a DataFrame with the transformed data
    transformed_df = pd.DataFrame(data=transformed_data,
                                  columns=[f'PC{i + 1}' for i in range(num_components)])

    # Displaying the first few rows of the transformed data
    print(transformed_df.head())

    # perform GP
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF

    # Setting up the Gaussian Process regressor with a Radial Basis Function (RBF) kernel
    gp_regressor = GaussianProcessRegressor(kernel=RBF(), random_state=42)

    # Using a small sample of the dataset for initial testing to manage computational demands
    small_sample_gp = transformed_df.sample(frac=0.05, random_state=42)
    small_sample_target = data.loc[small_sample_gp.index, 'target']

    # Fitting the Gaussian Process regressor on this small sample
    gp_regressor.fit(small_sample_gp, small_sample_target)

    # Scoring the model using the R^2 metric on the same small sample (simple self-check)
    gp_score = gp_regressor.score(small_sample_gp, small_sample_target)
    print(gp_score)

    from sklearn.model_selection import cross_val_score

    # Setting up the Gaussian Process regressor with an RBF kernel for cross-validation
    gp_regressor_cv = GaussianProcessRegressor(kernel=RBF(), random_state=42)

    # Performing 5-fold cross-validation using negative mean squared error as the scoring method
    gp_cv_scores = cross_val_score(gp_regressor_cv, small_sample_gp, small_sample_target, cv=5,
                                   scoring='neg_mean_squared_error')

    # Calculating the average MSE across all folds (convert negative MSE to positive)
    average_mse_gp = -gp_cv_scores.mean()
    print("The average MSE of PCA-transformed model is", average_mse_gp)
