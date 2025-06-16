import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
import matplotlib.pyplot as plt
import os

# Create directory for plots
base_dir = "plots/task6_3_nn_collaborative_filtering/"
os.makedirs(base_dir, exist_ok=True)

# Load ratings dataset
ratings_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML321EN-SkillsNetwork/labs/datasets/ratings.csv"
ratings_df = pd.read_csv(ratings_url)

# Prepare data for the neural network
user_ids = ratings_df['user'].unique()
course_ids = ratings_df['item'].unique()
user_id_map = {id: idx for idx, id in enumerate(user_ids)}
course_id_map = {id: idx for idx, id in enumerate(course_ids)}

ratings_df['user_idx'] = ratings_df['user'].map(user_id_map)
ratings_df['course_idx'] = ratings_df['item'].map(course_id_map)

n_users = len(user_ids)
n_courses = len(course_ids)

# Neural network model
user_input = Input(shape=(1,))
course_input = Input(shape=(1,))
user_embedding = Embedding(n_users, 10, input_length=1)(user_input)
course_embedding = Embedding(n_courses, 10, input_length=1)(course_input)
user_vec = Flatten()(user_embedding)
course_vec = Flatten()(course_embedding)
concat = Concatenate()([user_vec, course_vec])
dense = Dense(128, activation='relu')(concat)
output = Dense(1)(dense)

model = Model(inputs=[user_input, course_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

# Train the model
history = model.fit(
    [ratings_df['user_idx'], ratings_df['course_idx']],
    ratings_df['rating'],
    epochs=20,
    batch_size=64,
    verbose=0
)

# Predict ratings for the first user
user_idx = 0
user_id = user_ids[user_idx]
course_indices = np.arange(n_courses)
predictions = model.predict([np.array([user_idx] * n_courses), course_indices], verbose=0)
recommendations = pd.Series(predictions.flatten(), index=course_ids)

# Filter out courses the user has already rated
user_rated = ratings_df[ratings_df['user'] == user_id]['item']
recommendations = recommendations[~recommendations.index.isin(user_rated)].sort_values(ascending=False).head(5)

print(f"Recommendations for User {user_id}:")
print(recommendations)

# Save training loss plot
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Neural Network Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(base_dir, "training_loss.png"))
plt.close()
