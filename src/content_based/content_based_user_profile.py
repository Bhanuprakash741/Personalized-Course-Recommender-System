import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for plots
base_dir = "plots/task5_2_content_based_user_profile/"
os.makedirs(base_dir, exist_ok=True)

# Load datasets
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
course_df = pd.read_csv(course_genre_url)
ratings_df = pd.read_csv(ratings_url)

# Select a user and get their rated courses
user_id = ratings_df['user'].iloc[0]  # First user for demo
user_ratings = ratings_df[ratings_df['user'] == user_id]
user_courses = course_df[course_df['COURSE_ID'].isin(user_ratings['item'])]

# Build user profile based on genres of rated courses
genre_columns = course_df.columns[2:]
user_profile = user_courses[genre_columns].sum()

# Calculate compatibility scores for all courses
compatibility_scores = course_df[genre_columns].dot(user_profile)
course_df['compatibility_score'] = compatibility_scores
recommended_courses = course_df[~course_df['COURSE_ID'].isin(user_courses['COURSE_ID'])]  
recommended_courses = recommended_courses.sort_values(by='compatibility_score', ascending=False).head(5)

print(f"Top 5 recommended courses for user {user_id}:")
print(recommended_courses[['COURSE_ID', 'TITLE', 'compatibility_score']])

# Save compatibility scores plot
plt.figure(figsize=(10, 6))
sns.barplot(x=recommended_courses['compatibility_score'], y=recommended_courses['TITLE'])
plt.title(f'Top 5 Course Recommendations for User {user_id}')
plt.xlabel('Compatibility Score')
plt.ylabel('Course Title')
plt.savefig(os.path.join(base_dir, "user_recommendations.png"))
plt.close()
