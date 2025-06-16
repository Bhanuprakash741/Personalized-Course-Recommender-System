import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for plots
base_dir = "plots/task5_3_content_based_clustering/"
os.makedirs(base_dir, exist_ok=True)

# Load datasets
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
course_df = pd.read_csv(course_genre_url)
ratings_df = pd.read_csv(ratings_url)

# Use genre features for clustering
genre_columns = course_df.columns[2:]
genre_features = course_df[genre_columns]

# K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
course_df['cluster'] = kmeans.fit_predict(genre_features)

# Select a user and their courses
user_id = ratings_df['user'].iloc[0]
user_courses = ratings_df[ratings_df['user'] == user_id]['item']
user_course_clusters = course_df[course_df['COURSE_ID'].isin(user_courses)]['cluster']

# Most frequent cluster for the user's courses
most_common_cluster = user_course_clusters.mode().iloc[0]
recommended_courses = course_df[(course_df['cluster'] == most_common_cluster) & (~course_df['COURSE_ID'].isin(user_courses))]
recommended_courses = recommended_courses.head(5)

print(f"Top 5 recommended courses for user {user_id} from cluster {most_common_cluster}:")
print(recommended_courses[['COURSE_ID', 'TITLE', 'cluster']])

# Save clusters plot (using PCA for 2D visualization)
pca = PCA(n_components=2)
genre_pca = pca.fit_transform(genre_features)
course_df['pca1'] = genre_pca[:, 0]
course_df['pca2'] = genre_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', size='cluster', data=course_df, palette='Set2')
plt.title('Course Clusters Based on Genres (PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.savefig(os.path.join(base_dir, "course_clusters_pca.png"))
plt.close()
