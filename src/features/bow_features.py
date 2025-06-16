import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create directory for plots
base_dir = "plots/task3_bow_features/"
os.makedirs(base_dir, exist_ok=True)

# Load course dataset
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
course_df = pd.read_csv(course_genre_url)

# Extract BoW features from course titles
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(course_df['TITLE'])
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=vectorizer.get_feature_names_out(), index=course_df['COURSE_ID'])

print("Bag of Words Features (first 5 rows):")
print(bow_df.head())

# Save the frequency of top words
word_sums = bow_df.sum().sort_values(ascending=False)[:10]
plt.figure(figsize=(10, 6))
sns.barplot(x=word_sums.values, y=word_sums.index)
plt.title('Top 10 Most Frequent Words in Course Titles')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.savefig(os.path.join(base_dir, "top_words_frequency.png"))
plt.close()
