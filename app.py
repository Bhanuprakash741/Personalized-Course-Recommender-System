import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load course dataset
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
course_df = pd.read_csv(course_genre_url)

# Compute course similarity based on titles
vectorizer = CountVectorizer(stop_words='english')
bow_matrix = vectorizer.fit_transform(course_df['TITLE'])
similarity_matrix = cosine_similarity(bow_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=course_df['COURSE_ID'], columns=course_df['COURSE_ID'])

# Streamlit app
st.title("Course Recommender System")

# Select a course
course_options = course_df['COURSE_ID'].tolist()
course = st.selectbox("Select a course you've taken:", course_options)
if course:
    # Get recommendations
    similar_courses = similarity_df[course].sort_values(ascending=False)[1:4]
    recommended_courses = course_df[course_df['COURSE_ID'].isin(similar_courses.index)][['COURSE_ID', 'TITLE']]
    recommended_courses['similarity_score'] = similar_courses.values

    st.subheader(f"Recommended Courses for {course}:")
    st.dataframe(recommended_courses)

    # Display course titles
    st.subheader("Course Details:")
    for _, row in recommended_courses.iterrows():
        st.write(f"**{row['TITLE']}** (ID: {row['COURSE_ID']}) - Similarity: {row['similarity_score']:.2f}")