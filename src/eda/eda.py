import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for plots
base_dir = "plots/task2_eda/"
os.makedirs(base_dir, exist_ok=True)

# Load datasets
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"

course_df = pd.read_csv(course_genre_url)
ratings_df = pd.read_csv(ratings_url)

# EDA: Summary statistics of ratings
print("Ratings Summary Statistics:")
print(ratings_df['rating'].describe())

# Save ratings distribution with a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(y=ratings_df['rating'], color='lightgreen')
plt.title('Boxplot of Ratings')
plt.ylabel('Rating')
plt.savefig(os.path.join(base_dir, "ratings_boxplot.png"))
plt.close()

# Analyze genre distribution (sum the binary genre columns)
genre_columns = course_df.columns[2:]  # Genre columns start after 'COURSE_ID' and 'TITLE'
genre_counts = course_df[genre_columns].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.title('Distribution of Course Genres')
plt.xlabel('Number of Courses')
plt.ylabel('Genre')
plt.savefig(os.path.join(base_dir, "genre_distribution.png"))
plt.close()

# User activity: Number of ratings per user
user_ratings_count = ratings_df.groupby('user')['rating'].count()
plt.figure(figsize=(10, 6))
user_ratings_count.hist(bins=20, color='salmon', edgecolor='black')
plt.title('Number of Ratings per User')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.savefig(os.path.join(base_dir, "ratings_per_user.png"))
plt.close()
