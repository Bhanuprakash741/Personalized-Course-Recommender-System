import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for plots
base_dir = "plots/task6_2_nmf_collaborative_filtering/"
os.makedirs(base_dir, exist_ok=True)

# Load ratings dataset
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
ratings_df = pd.read_csv(ratings_url)

# Create user-item matrix
user_item_matrix = ratings_df.pivot(index='user', columns='item', values='rating').fillna(0)

# NMF model
nmf = NMF(n_components=5, random_state=42, max_iter=500)
W = nmf.fit_transform(user_item_matrix)  # User latent factors
H = nmf.components_  # Item latent factors

# Reconstruct the matrix
reconstructed_matrix = pd.DataFrame(np.dot(W, H), index=user_item_matrix.index, columns=user_item_matrix.columns)

print("Original User-Item Matrix (first 5 rows and columns):")
print(user_item_matrix.iloc[:5, :5])
print("\nReconstructed Matrix (NMF, first 5 rows and columns):")
print(reconstructed_matrix.iloc[:5, :5])

# Recommend for the first user
user_idx = 0
user_id = user_item_matrix.index[user_idx]
user_rated = user_item_matrix.iloc[user_idx][user_item_matrix.iloc[user_idx] > 0].index
user_recommendations = reconstructed_matrix.iloc[user_idx]
user_recommendations = user_recommendations[~user_recommendations.index.isin(user_rated)].sort_values(ascending=False).head(5)

print(f"\nRecommendations for User {user_id}:")
print(user_recommendations)

# Save original vs reconstructed matrix (subset)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(user_item_matrix.iloc[:10, :10], annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('Original User-Item Matrix')
plt.subplot(1, 2, 2)
sns.heatmap(reconstructed_matrix.iloc[:10, :10], annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('Reconstructed Matrix (NMF)')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "original_vs_reconstructed.png"))
plt.close()
