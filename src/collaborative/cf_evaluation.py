import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
import matplotlib.pyplot as plt
import os

# Create directory for plots
base_dir = "plots/task7_cf_evaluation/"
os.makedirs(base_dir, exist_ok=True)

# Load ratings dataset
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
ratings_df = pd.read_csv(ratings_url)

# Train-test split
train_df = ratings_df.sample(frac=0.8, random_state=42)
test_df = ratings_df.drop(train_df.index)

# Create user-item matrices
train_matrix = train_df.pivot(index='user', columns='item', values='rating').fillna(0)
test_matrix = test_df.pivot(index='user', columns='item', values='rating').fillna(0)

# Align indices and columns
common_users = train_matrix.index.intersection(test_matrix.index)
common_items = train_matrix.columns.intersection(test_matrix.columns)
train_matrix = train_matrix.loc[common_users, common_items]
test_matrix = test_matrix.loc[common_users, common_items]

# 1. KNN Predictions
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(train_matrix)
knn_predictions = np.zeros(test_matrix.shape)
for i, user in enumerate(test_matrix.index):
    distances, indices = knn.kneighbors([train_matrix.loc[user]], n_neighbors=3)
    similar_users = train_matrix.index[indices.flatten()[1:]]
    knn_predictions[i] = train_matrix.loc[similar_users].mean().values

# 2. NMF Predictions
nmf = NMF(n_components=5, random_state=42, max_iter=500)
W = nmf.fit_transform(train_matrix)
H = nmf.components_
nmf_predictions = np.dot(W, H)

# 3. Neural Network Predictions
user_ids = ratings_df['user'].unique()
course_ids = ratings_df['item'].unique()
user_id_map = {id: idx for idx, id in enumerate(user_ids)}
course_id_map = {id: idx for idx, id in enumerate(course_ids)}
train_df['user_idx'] = train_df['user'].map(user_id_map)
train_df['course_idx'] = train_df['item'].map(course_id_map)

n_users = len(user_ids)
n_courses = len(course_ids)

user_input = Input(shape=(1,))
course_input = Input(shape=(1,))
user_embedding = Embedding(n_users, 10, input_length=1)(user_input)
course_embedding = Embedding(n_courses, 10, input_length=1)(course_input)
user_vec = Flatten()(user_embedding)
course_vec = Flatten()(course_embedding)
concat = Concatenate()([user_vec, course_vec])
dense = Dense(128, activation='relu')(concat)
output = Dense(1)(dense)

nn_model = Model(inputs=[user_input, course_input], outputs=output)
nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(
    [train_df['user_idx'], train_df['course_idx']],
    train_df['rating'],
    epochs=10,
    batch_size=64,
    verbose=0
)

# Generate NN predictions
nn_predictions = np.zeros(test_matrix.shape)
for i, user in enumerate(test_matrix.index):
    user_idx = user_id_map[user]
    course_indices = np.array([course_id_map[item] for item in test_matrix.columns])
    preds = nn_model.predict([np.array([user_idx] * len(course_indices)), course_indices], verbose=0)
    nn_predictions[i] = preds.flatten()

# Calculate RMSE
actual = test_matrix.values
rmse_knn = sqrt(mean_squared_error(actual, knn_predictions))
rmse_nmf = sqrt(mean_squared_error(actual, nmf_predictions))
rmse_nn = sqrt(mean_squared_error(actual, nn_predictions))

# Create a DataFrame for visualization
rmse_df = pd.DataFrame({
    'Method': ['KNN', 'NMF', 'Neural Network'],
    'RMSE': [rmse_knn, rmse_nmf, rmse_nn]
})

print("RMSE for each method:")
print(rmse_df)

# Save RMSE plot
plt.figure(figsize=(8, 6))
plt.bar(rmse_df['Method'], rmse_df['RMSE'], color=['skyblue', 'lightgreen', 'salmon'])
plt.title('RMSE of Collaborative Filtering Methods')
plt.xlabel('Method')
plt.ylabel('RMSE')
plt.savefig(os.path.join(base_dir, "rmse_comparison.png"))
plt.close()
