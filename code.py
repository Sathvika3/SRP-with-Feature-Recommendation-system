# -*- coding: utf-8 -*-
"""code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Cur7VjmNLsUsqFufGUpUnWe9oQsN2r0y
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
excel_file_path = 'ASN3-Run1-Dataset.xlsx'
feature_dfs = [pd.read_excel(excel_file_path, sheet_name=f'H{i:02d}') for i in range(1, 13)]
all_features_df = pd.concat(feature_dfs, ignore_index=True)
all_features_df['Concatenated_Description'] = all_features_df['Feature Description']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(all_features_df['Concatenated_Description'].values)
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
sns.heatmap(tfidf_matrix.todense()[:, :10], annot=True, cmap='viridis')
plt.title("TF-IDF Matrix Heatmap")
plt.xlabel("Features")
plt.ylabel("Words")
plt.show()

# Visualize the Cosine Similarity Matrix
sns.heatmap(cosine_sim_matrix, annot=True, cmap='Blues')
plt.title("Cosine Similarity Matrix")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()


# RL Environment setup
class FeatureRecommendationEnvironment:
    def __init__(self, cosine_sim_matrix):
        self.cosine_sim_matrix = cosine_sim_matrix
        self.current_state = 0
        self.num_features = cosine_sim_matrix.shape[0]

    def step(self, action):
        next_state = action
        reward = calculate_reward(len(all_features_df['Concatenated_Description'].iloc[next_state]),
                                  self.cosine_sim_matrix[self.current_state, next_state])
        done = (next_state == self.num_features - 1)
        self.current_state = next_state
        return next_state, reward, done

    def reset(self):
        self.current_state = 0
        return self.current_state

# Reward calculation and precision functions
def calculate_reward(description_length, textual_similarity):
      length_reward = function_of_length(description_length)
      similarity_reward = function_of_similarity(textual_similarity)
      return 0.5 * length_reward + 0.5 * similarity_reward

def function_of_length(description_length):
    return 1 / (1 + np.exp(-description_length))

def function_of_similarity(textual_similarity):
    return textual_similarity

def is_decision_correct(reward, threshold):
    return reward > threshold
lengths = np.linspace(1, 100, 100)
similarities = np.linspace(0, 1, 100)

# Calculating rewards for a range of lengths and similarities
length_rewards = [calculate_reward(l, 0.5) for l in lengths]  # Fixed similarity
similarity_rewards = [calculate_reward(50, s) for s in similarities]  # Fixed length

# Plotting Reward Function behavior
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(lengths, length_rewards)
plt.title("Reward vs Description Length")
plt.xlabel("Description Length")
plt.ylabel("Reward")

plt.subplot(1, 2, 2)
plt.plot(similarities, similarity_rewards)
plt.title("Reward vs Textual Similarity")
plt.xlabel("Textual Similarity")
plt.ylabel("Reward")

plt.tight_layout()
plt.show()

# RL Training Function
def run_rl_training(learning_rate, num_episodes=10):
    Q_table = np.zeros((len(all_features_df), len(all_features_df)))
    env = FeatureRecommendationEnvironment(cosine_sim_matrix)
    episode_rewards = []
    precision_values = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        correct_decisions = 0
        total_decisions = 0

        while True:
            # ... [RL step code, action selection, state transition] ...

            next_state, reward, done = env.step(24)
            total_reward += reward
            total_decisions += 1
            if is_decision_correct(reward, 0.5):
                correct_decisions += 1

            # ... [State update, check if done] ...

        episode_rewards.append(total_reward)
        episode_precision = correct_decisions / total_decisions if total_decisions > 0 else 0
        precision_values.append(episode_precision)

    average_reward = np.mean(episode_rewards)
    stability = np.std(episode_rewards)
    return average_reward, stability, precision_values

# Test different learning rates
learning_rates = [0.01, 0.05, 0.1, 0.2]
metrics = {}

for lr in learning_rates:
    avg_reward, stability, precision = run_rl_training(lr)
    metrics[lr] = {'Average Reward': avg_reward, 'Stability': stability, 'Precision': precision}

# Plotting the results for Average Reward and Stability
fig, ax1 = plt.subplots()

ax1.set_xlabel('Learning Rate')
ax1.set_ylabel('Average Reward', color='tab:blue')
ax1.plot(learning_rates, [metrics[lr]['Average Reward'] for lr in learning_rates], color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()
ax2.set_ylabel('Stability', color='tab:red')
ax2.plot(learning_rates, [metrics[lr]['Stability'] for lr in learning_rates], color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

plt.title('Average Reward and Stability at Different Learning Rates')
plt.show()

# Plotting Precision for each trial
for lr, data in metrics.items():
    plt.plot(data['Precision'], label=f'LR: {lr}')

plt.xlabel('Episode')
plt.ylabel('Precision')
plt.title('Precision per Episode Across Different Learning Rates')
plt.legend()
plt.show()