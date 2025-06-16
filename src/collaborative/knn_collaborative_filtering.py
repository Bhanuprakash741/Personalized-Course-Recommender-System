import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for plots
base_dir = "plots/task6_1_knn_collaborative_filtering/"
os.makedirs(base_dir, exist_ok=True)

# Load ratings dataset
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
ratings_df = pd.read_csv(ratings_url)

# Create user-item matrix
user_item_matrix = ratings_df.pivot(index='user', columns='item', values='rating').fillna(0)
print("User-Item Matrix Shape:", user_item_matrix.shape)

# KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)

# Find similar users to the first user
user_idx = 0
distances, indices = knn.kneighbors([user_item_matrix.iloc[user_idx]], n_neighbors=4)
similar_users = user_item_matrix.index[indices.flatten()[1:]].tolist()
print(f"Users similar to User {user_item_matrix.index[user_idx]}: {similar_users}")

# Recommend courses based on similar users
similar_users_ratings = user_item_matrix.loc[similar_users]
recommendations = similar_users_ratings.mean().sort_values(ascending=False)
# Filter out courses the user has already rated
user_rated = user_item_matrix.iloc[user_idx][user_item_matrix.iloc[user_idx] > 0].index
recommendations = recommendations[~recommendations.index.isin(user_rated)].head(5)

print("Recommended courses:")
print(recommendations)

# Save a subset of the user-item matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(user_item_matrix.iloc[:10, :10], annot=True, cmap='YlGnBu', fmt='.1f')
plt.title('User-Item Matrix Heatmap (First 10 Users and Courses)')
plt.xlabel('Course ID')
plt.ylabel('User ID')
plt.savefig(os.path.join(base_dir, "user_item_heatmap.png"))
plt.close()
