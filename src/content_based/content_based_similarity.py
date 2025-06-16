import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create directory for plots
base_dir = "plots/task5_1_content_based_similarity/"
os.makedirs(base_dir, exist_ok=True)

# Load course dataset
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
course_df = pd.read_csv(course_genre_url)

# BoW features and similarity
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(course_df['TITLE'])
similarity_matrix = cosine_similarity(bow_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=course_df['COURSE_ID'], columns=course_df['COURSE_ID'])

# Recommend top 3 similar courses for 'ML0101ENv3'
course = 'ML0101ENv3'
similar_courses = similarity_df[course].sort_values(ascending=False)[1:4]

# Filter and attach similarity scores
recommended_courses = course_df[course_df['COURSE_ID'].isin(similar_courses.index)][['COURSE_ID', 'TITLE']]
recommended_courses['similarity_score'] = similar_courses.values

print(f"Top 3 recommended courses for {course}:")
print(recommended_courses)

# Save recommendations plot
plt.figure(figsize=(8, 6))
sns.barplot(x=recommended_courses['similarity_score'], y=recommended_courses['TITLE'])
plt.title(f'Top 3 Similar Courses to {course}')
plt.xlabel('Similarity Score')
plt.ylabel('Course Title')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "similar_courses.png"))
plt.close()
