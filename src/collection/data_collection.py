import pandas as pd
import matplotlib.pyplot as plt
import os

# Create directory for plots
base_dir = "plots/task1_data_collection/"
os.makedirs(base_dir, exist_ok=True)

# Load datasets
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"

course_df = pd.read_csv(course_genre_url)
ratings_df = pd.read_csv(ratings_url)

# Basic understanding
print("Course Genre Dataset Info:")
print(course_df.info())
print("\nFirst few rows of Course Genre:")
print(course_df.head())
print("\nRatings Dataset Info:")
print(ratings_df.info())
print("\nFirst few rows of Ratings:")
print(ratings_df.head())

# Plot and save the distribution of ratings
plt.figure(figsize=(10, 6))
ratings_df['rating'].hist(bins=5, color='skyblue', edgecolor='black')
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.savefig(os.path.join(base_dir, "ratings_distribution.png"))
plt.close()

# Plot and save the number of ratings per course
course_ratings_count = ratings_df.groupby('item')['rating'].count()
plt.figure(figsize=(10, 6))
course_ratings_count.hist(bins=20, color='lightgreen', edgecolor='black')
plt.title('Number of Ratings per Course')
plt.xlabel('Number of Ratings')
plt.ylabel('Frequency')
plt.savefig(os.path.join(base_dir, "ratings_per_course.png"))
plt.close()