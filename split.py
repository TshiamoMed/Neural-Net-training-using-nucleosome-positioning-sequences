import numpy as np
from sklearn.model_selection import train_test_split

# Load the aligned data
X = np.load('combined_features.npy')  # Load the one-hot encoded sequences
Y = np.load('target_values.npy')  # Load the target values

# Check the dimensions of the loaded data
print(f"Loaded one-hot encoded sequences shape: {X.shape}")
print(f"Loaded target values shape: {Y.shape}")

# Split the data into 80% training and 20% temporary (for validation and testing)
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.2, random_state=42)

# Now split the temporary set into validation and testing sets (50% of the temp, which is 10% of the original)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Check the shapes of the splits
print(f"Training set shape: {X_train.shape}, Validation set shape: {X_val.shape}, Test set shape: {X_test.shape}")
print(f"Training labels shape: {Y_train.shape}, Validation labels shape: {Y_val.shape}, Test labels shape: {Y_test.shape}")

# Save the split datasets
np.save('X_train.npy', X_train)
np.save('X_val.npy', X_val)
np.save('X_test.npy', X_test)
np.save('Y_train.npy', Y_train)
np.save('Y_val.npy', Y_val)
np.save('Y_test.npy', Y_test)

print("Data split and saved successfully.")
