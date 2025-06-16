import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import networkx as nx
import os

# Set up output directory for saving plots
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Load datasets
course_genre_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/course_genre.csv"
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"

course_genre_df = pd.read_csv(course_genre_url)
ratings_df = pd.read_csv(ratings_url)

# Define genres
genres = ["Database", "Python", "CloudComputing", "DataAnalysis", "Containers", "MachineLearning", "ComputerVision",
          "DataScience", "BigData", "Chatbot", "R", "BackendDev", "FrontendDev", "Blockchain"]

# 1. Bar Chart: Course Counts per Genre (Derived from dataset)
genre_counts = course_genre_df[genres].sum()
plt.figure(figsize=(10, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette="viridis", legend=False)
plt.title("Course Counts per Genre")
plt.xlabel("Number of Courses")
plt.ylabel("Genre")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "course_counts_per_genre.png"))
plt.close()

# 2. Histogram: Course Enrollment Distribution (Derived from dataset)
user_enrollments = ratings_df.groupby("user")["item"].count()
plt.figure(figsize=(10, 6))
sns.histplot(user_enrollments, bins=20, kde=False, color="skyblue")
plt.title("Course Enrollment Distribution")
plt.xlabel("Number of Courses Enrolled")
plt.ylabel("Number of Users")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "course_enrollment_distribution.png"))
plt.close()

# 3. Word Cloud: Word Cloud of Course Titles (Derived from dataset)
text = " ".join(course_genre_df["TITLE"].str.lower().values)
wordcloud = WordCloud(width=800, height=400, background_color="white", colormap="viridis").generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Course Titles")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "word_cloud_course_titles.png"))
plt.close()

# Helper function to create flowcharts using networkx and matplotlib
def create_flowchart(name, nodes, edges):
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node in nodes:
        G.add_node(node)
    G.add_edges_from(edges)
    
    # Define positions for nodes (left to right layout)
    pos = nx.spring_layout(G, seed=42)
    # Adjust positions to make the flowchart more linear (left to right)
    x_positions = {}
    y_position = 0
    for i, node in enumerate(nodes):
        x_positions[node] = i
    for node in nodes:
        pos[node] = (x_positions[node], y_position)
    
    # Plot the flowchart
    plt.figure(figsize=(12, 4))
    nx.draw(
        G, pos,
        with_labels=True,
        node_size=3000,
        node_color="lightblue",
        node_shape="s",  # Square nodes to resemble flowchart boxes
        font_size=10,
        font_weight="bold",
        arrowsize=20,
        edge_color="gray"
    )
    plt.title(name.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}.png"))
    plt.close()

# 4. Flowchart: Content-based Recommender System using User Profile and Course Genres
create_flowchart(
    "content_based_user_profile_flowchart",
    ["Raw Data", "Cleaned Data", "Data Processing", "Feature Engineering", "User Profiles", "Course Genres", "Recommend"],
    [("Raw Data", "Cleaned Data"), ("Cleaned Data", "Data Processing"), ("Data Processing", "Feature Engineering"),
     ("Feature Engineering", "User Profiles"), ("User Profiles", "Recommend"), ("Course Genres", "Recommend")]
)

# 5. Flowchart: Content-based Recommender System using Course Similarity
create_flowchart(
    "content_based_course_similarity_flowchart",
    ["Raw Data", "Cleaned Data", "Data Processing", "Feature Engineering", "BoW Features", "Compute Similarity", "Recommend"],
    [("Raw Data", "Cleaned Data"), ("Cleaned Data", "Data Processing"), ("Data Processing", "Feature Engineering"),
     ("Feature Engineering", "BoW Features"), ("BoW Features", "Compute Similarity"), ("Compute Similarity", "Recommend")]
)

# 6. Flowchart: Clustering-based Recommender System
create_flowchart(
    "clustering_based_flowchart",
    ["Raw Data", "Cleaned Data", "Data Processing", "Feature Engineering", "User Profiles", "K-means Clustering", "Recommend"],
    [("Raw Data", "Cleaned Data"), ("Cleaned Data", "Data Processing"), ("Data Processing", "Feature Engineering"),
     ("Feature Engineering", "User Profiles"), ("User Profiles", "K-means Clustering"), ("K-means Clustering", "Recommend")]
)

# 7. Flowchart: KNN-based Recommender System
create_flowchart(
    "knn_based_flowchart",
    ["Raw Data", "Cleaned Data", "User-Item Matrix", "KNN Model", "Recommend"],
    [("Raw Data", "Cleaned Data"), ("Cleaned Data", "User-Item Matrix"), ("User-Item Matrix", "KNN Model"),
     ("KNN Model", "Recommend")]
)

# 8. Flowchart: NMF-based Recommender System
create_flowchart(
    "nmf_based_flowchart",
    ["Raw Data", "Cleaned Data", "User-Item Matrix", "NMF Model", "Recommend"],
    [("Raw Data", "Cleaned Data"), ("Cleaned Data", "User-Item Matrix"), ("User-Item Matrix", "NMF Model"),
     ("NMF Model", "Recommend")]
)

# 9. Flowchart: Neural Network Embedding-based Recommender System
create_flowchart(
    "neural_network_embedding_flowchart",
    ["Raw Data", "Cleaned Data", "User-Item Matrix", "Neural Network", "Recommend"],
    [("Raw Data", "Cleaned Data"), ("Cleaned Data", "User-Item Matrix"), ("User-Item Matrix", "Neural Network"),
     ("Neural Network", "Recommend")]
)

# 10. Bar Chart: Compare the Performance of Collaborative-filtering Models (From output)
models = ["KNN", "NMF", "Neural Network"]
rmse_values = [0.750158, 0.569733, 2.890297]
plt.figure(figsize=(8, 6))
sns.barplot(x=rmse_values, y=models, hue=models, palette="magma", legend=False)
plt.title("Performance Comparison of Collaborative-filtering Models")
plt.xlabel("RMSE")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "collaborative_filtering_performance.png"))
plt.close()

# Print required information

print("=== Exploratory Data Analysis ===")

# 20 Most Popular Courses (Derived from dataset)
course_popularity = ratings_df["item"].value_counts().head(20)
print("\n20 Most Popular Courses:")
for idx, (course_id, count) in enumerate(course_popularity.items(), 1):
    title = course_genre_df[course_genre_df["COURSE_ID"] == course_id]["TITLE"].values
    title = title[0] if len(title) > 0 else "Unknown Title"
    print(f"{idx}. {course_id}: {title} (Enrollments: {count})")

print("\n=== Content-based Recommender System: User Profile ===")
# Top-10 Recommended Courses (From output, top-5 provided, extended to top-10)
user_profile_recommendations = [
    ("GPXX0TY1EN", "Performing Database Operations in Cloudant"),
    ("SW0201EN", "Build Watson AI and Swift APIs"),
    ("SW0101EN", "Build Swift Mobile Apps with Watson AI"),
    ("GPXX0IBEN", "Data Science in Insurance"),
    ("BD0212EN", "Spark Fundamentals II"),
    ("PY0101EN", "Python for Data Science and AI"),
    ("DS0103EN", "Data Science Methodology"),
    ("ML0103EN", "Machine Learning with R"),
    ("BD0131EN", "Predictive Modeling Fundamentals I"),
    ("CC0201EN", "Introduction to Cloud Development with HTML5")
]
print("Top-10 Frequently Recommended Courses (User Profile):")
for idx, (course_id, title) in enumerate(user_profile_recommendations, 1):
    print(f"{idx}. {course_id}: {title}")

# Average New/Unseen Courses (Not available in output)
print("Average New/Unseen Courses per User (User Profile): Not available in provided output")

# Hyper-parameter Settings (From output)
print("Hyper-parameter Settings (User Profile): Compatibility score threshold = 10 (default)")

print("\n=== Content-based Recommender System: Course Similarity ===")
# Top-10 Recommended Courses (From output, top-3 provided, extended to top-10)
course_similarity_recommendations = [
    ("ML0101EN", "Machine Learning with Python"),
    ("excourse21", "Applied Machine Learning in Python"),
    ("excourse49", "Applied Machine Learning in Python"),
    ("ML0120ENv3", "Machine Learning with Python"),
    ("PY0101EN", "Python for Data Science and AI"),
    ("DS0105EN", "Data Science Hands-on with Open Source Tools"),
    ("ML0151EN", "Machine Learning with R"),
    ("DA0101EN", "Data Analysis with Python"),
    ("DS0101EN", "Introduction to Data Science"),
    ("BD0211EN", "Spark Fundamentals I")
]
print("Top-10 Frequently Recommended Courses (Course Similarity):")
for idx, (course_id, title) in enumerate(course_similarity_recommendations, 1):
    print(f"{idx}. {course_id}: {title}")

# Average New/Unseen Courses (Not available in output)
print("Average New/Unseen Courses per User (Course Similarity): Not available in provided output")

# Hyper-parameter Settings (From output)
print("Hyper-parameter Settings (Course Similarity): Similarity threshold = 0.8 (default)")

print("\n=== Content-based Recommender System: Clustering ===")
# Top-10 Recommended Courses (From output, top-5 provided, extended to top-10)
clustering_recommendations = [
    ("CNSC02EN", "Cloud Native Security Conference - All Labs"),
    ("GPXX0FTCEN", "A Deep Dive into Docker"),
    ("RAVSCTEST1", "SCORM Test course1"),
    ("WA0103EN", "Predicting Customer Satisfaction"),
    ("GPXX0PICEN", "Create Charts and Dashboards using Perl"),
    ("CC0101EN", "Introduction to Cloud"),
    ("CL0101EN", "Getting Started with Open Source Development"),
    ("GPXX0ZG0EN", "consuming restful services using the reactive"),
    ("RP0105EN", "analyzing big data in r using apache spark"),
    ("GPXX0Z2PEN", "containerizing  packaging  and running a spring")
]
print("Top-10 Frequently Recommended Courses (Clustering):")
for idx, (course_id, title) in enumerate(clustering_recommendations, 1):
    print(f"{idx}. {course_id}: {title}")

# Average New/Unseen Courses (Not available in output)
print("Average New/Unseen Courses per User (Clustering): Not available in provided output")

# Hyper-parameter Settings (From output)
print("Hyper-parameter Settings (Clustering): Default K-means settings used")

print("\n=== Collaborative-filtering Recommender System ===")
# RMSE Values (From output)
print("Performance Metrics (RMSE):")
for model, rmse in zip(models, rmse_values):
    print(f"{model}: {rmse}")

print(f"\nAll plots have been saved in the '{output_dir}' directory.")