import os
import pandas as pd
import numpy as np



if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    data_dir = os.path.join('..', 'data')
    data_path = os.path.join(data_dir, 'df_normalized.csv')
    df = pd.read_csv(data_path)
    target = df.iloc[:, -1]
    features = df.columns[:-1]
    # print(df)

    data_cleaned = df.copy()

    for column in df.columns:
        median = data_cleaned[column].median()
        std_dev = data_cleaned[column].std()
        mean = data_cleaned[column].mean()

        outliers = ((data_cleaned[column] - mean).abs() > 3 * std_dev)

        data_cleaned.loc[outliers, column] = median

    outliers_after_replacement = data_cleaned.apply(lambda x: ((x - x.mean()).abs() > 3 * x.std())).sum()

    # print(outliers_after_replacement[outliers_after_replacement > 0])

    from sklearn.cluster import KMeans
    import numpy as np
    import matplotlib.pyplot as plt

    kmeans = KMeans(n_clusters=2, random_state=42)
    data_cleaned['cluster'] = kmeans.fit_predict(data_cleaned.iloc[:, :-1])
    distances = kmeans.transform(data_cleaned.iloc[:, :-2])
    data_cleaned['distance_to_center1'] = distances[:, 0]
    data_cleaned['distance_to_center2'] = distances[:, 1]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(data_cleaned['distance_to_center1'], data_cleaned['distance_to_center2'],
                          c=data_cleaned['target'], cmap='viridis', alpha=0.6, edgecolor='k', marker='x',
                          s=50 + (20 * data_cleaned['target']))

    plt.xlabel('Distance to Cluster Center 1')
    plt.ylabel('Distance to Cluster Center 2')
    plt.colorbar(scatter, label='Target Value')
    plt.title('K-Means Clustering (k=2) Results')
    plt.grid(True)
    plt.show()





    # Plotting with specific colors for each cluster
    plt.figure(figsize=(10, 6))
    colors = np.where(data_cleaned['cluster'] == 0, 'red', 'blue')  # Red for cluster 0, Blue for cluster 1

    scatter = plt.scatter(data_cleaned['distance_to_center1'], data_cleaned['distance_to_center2'],
                          c=colors, alpha=0.6, edgecolor='k', marker='x',
                          s=50 + (20 * data_cleaned['target']))  # Size adjusted by target

    plt.xlabel('Distance to Cluster Center 1')
    plt.ylabel('Distance to Cluster Center 2')
    plt.title('K-Means Clustering (k=2) Results with Cluster Colors')
    plt.grid(True)

    from matplotlib.lines import Line2D

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Cluster 1',
                              markerfacecolor='red', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Cluster 2',
                              markerfacecolor='blue', markersize=10)]
    plt.legend(handles=legend_elements, title="Clusters")

    plt.show()

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    X = data_cleaned[['distance_to_center1', 'distance_to_center2']]
    y = data_cleaned['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)

    print(f"r2 is: {r2}")





    from sklearn.metrics import mean_squared_error

    k_values = range(2, 21)
    results = []

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X)
        distances = kmeans.transform(X)

        X_clust = distances
        y_clust = y

        X_train_clust, X_test_clust, y_train_clust, y_test_clust = train_test_split(X_clust, y_clust, test_size=0.2,
                                                                                    random_state=42)

        lr_clust = LinearRegression()
        lr_clust.fit(X_train_clust, y_train_clust)

        y_pred_clust = lr_clust.predict(X_test_clust)
        r2_clust = r2_score(y_test_clust, y_pred_clust)
        mse_clust = mean_squared_error(y_test_clust, y_pred_clust)

        results.append((k, r2_clust, mse_clust))

    results_df = pd.DataFrame(results, columns=['k', 'R^2', 'MSE'])
    # print(results_df)

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['k'], results_df['MSE'], marker='o', linestyle='-', color='b')
    plt.title('K-Means Clustering: k vs. MSE (Corrected Data)')
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('MSE (Mean Squared Error)')
    plt.grid(True)
    plt.show()

    from sklearn.model_selection import cross_val_score
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler




    def evaluate_k(X, y, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X)
        distances = kmeans.transform(X)

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

        scores = cross_val_score(pipeline, distances, y, cv=5, scoring='neg_mean_squared_error')
        return -scores.mean()

    mse_5 = evaluate_k(X, y, 5)
    mse_6 = evaluate_k(X, y, 6)

    print(f'MSE for k=5: {mse_5}')
    print(f'MSE for k=6: {mse_6}')




    from sklearn.decomposition import PCA

    kmeans = KMeans(n_clusters=5, random_state=42)
    data_cleaned['cluster'] = kmeans.fit_predict(data_cleaned.iloc[:, :-4])

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_cleaned.iloc[:, :-4])

    plt.figure(figsize=(10, 7))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=data_cleaned['cluster'], marker='x', cmap='viridis', alpha=0.5)
    plt.title('PCA Visualization of Clustering (k=5)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Cluster Label')
    plt.show()
