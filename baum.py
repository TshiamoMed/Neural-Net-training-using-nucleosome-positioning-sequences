import numpy as np
import os
from hmmlearn import hmm
import joblib

# Step 1: Create the algorithm directory
output_dir = "algorithm"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#loading the files 
onehot_encoded_sequences = np.load('one_sequences.npy')  
target_values = np.load('target_values_stripped.npy') # the target values file was stripped to only have the nucleosome count, the chromosome and central positions columns were removed.

# Step 3: Reshape data to fit the HMM input format
n_sequences, n_bp, n_features = onehot_encoded_sequences.shape
sequences_for_hmm = onehot_encoded_sequences.reshape(n_sequences, n_bp * n_features)

# Step 4: Define and train the HMM using Baum-Welch
n_hidden_states = 2  

# Initialize the HMM
hmm_model = hmm.GaussianHMM(n_components=n_hidden_states, covariance_type="diag", n_iter=100)

# Fit the HMM to the sequence data (Baum-Welch algorithm)
# Note: We are using the sequences_for_hmm which is 2D
hmm_model.fit(sequences_for_hmm)

# Step 5: Use the trained HMM to predict hidden states for each sequence
# The hidden states will be used as additional features to the one hot encoding
hidden_states = hmm_model.predict(sequences_for_hmm)

# Step 6: Save the hidden states and model into the algorithm directory
hidden_states_file = os.path.join(output_dir, 'hidden_states.npy')
np.save(hidden_states_file, hidden_states)

# Save the trained HMM model for future use
hmm_model_file = os.path.join(output_dir, 'hmm_model.pkl')
joblib.dump(hmm_model, hmm_model_file)

# Step 7: Combine hidden states with one-hot encoded sequences for neural network training
combined_features = np.concatenate([sequences_for_hmm, hidden_states.reshape(-1, 1)], axis=1)

# Save the combined feature set for training neural network
combined_features_file = os.path.join(output_dir, 'combined_features.npy')
np.save(combined_features_file, combined_features)

# Save the target values for neural network training
target_values_file = os.path.join(output_dir, 'target_values.npy')
np.save(target_values_file, target_values)

print(f"All files have been saved to the '{output_dir}' directory.")
