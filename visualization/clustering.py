from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 5. Customer Segmentation Based on Consumption Profiles
def customer_segmentation(df, consumption_cols=None):
    if consumption_cols is None:
        consumption_cols = [
            'PeakConsumption', 'StandardConsumption', 'OffPeakConsumption',
            'Block1Consumption', 'Block2Consumption', 'Block3Consumption',
            'Block4Consumption', 'NonTOUConsumption'
        ]
    # Aggregate average consumption per customer
    customer_profiles = df.groupby('CustomerID')[consumption_cols].mean()

    # Standardize data
    scaler = StandardScaler()
    scaled_profiles = scaler.fit_transform(customer_profiles)

    # Reduce dimensionality for visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_profiles)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(scaled_profiles)

    # Create DataFrame for visualization
    cluster_df = pd.DataFrame({
        'CustomerID': customer_profiles.index,
        'PC1': pca_result[:, 0],
        'PC2': pca_result[:, 1],
        'Cluster': clusters
    })

    # Plot clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='Cluster', palette='tab10', s=100)
    plt.title("Customer Segmentation Based on Average Consumption")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
