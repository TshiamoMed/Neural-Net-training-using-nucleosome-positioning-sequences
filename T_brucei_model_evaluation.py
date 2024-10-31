import os
import numpy as np 
import tensorflow as tf
from tensorflow.keras import models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_gen_directory():
    """Create 'gen' directory if it doesn't exist."""
    if not os.path.exists('genT'):
        os.makedirs('genT')

def load_and_preprocess_data(X_test_path, Y_test_path):
    """Load and preprocess test data."""
    X_test = np.load(X_test_path)
    Y_test = np.load(Y_test_path)
    
    target_scaler = MinMaxScaler()
    
    # Scale Y_test for processing
    Y_test_scaled = target_scaler.fit_transform(Y_test.reshape(-1, 1))
    
    return X_test, Y_test, Y_test_scaled, target_scaler

def evaluate_predictions(Y_test, predictions):
    """Calculate and return evaluation metrics."""
    mse = mean_squared_error(Y_test, predictions)
    mae = np.mean(np.abs(Y_test - predictions))
    r2 = r2_score(Y_test, predictions)
    
    return mse, mae, r2

def save_predictions(predictions, actuals, output_file):
    """Save predictions and compare them with actual values."""
    with open(output_file, 'w') as f:
        f.write("Predicted\tActual\tDifference\tRounded_Pred\tCorrect\n")
        
        # Flatten predictions and actuals if needed
        predictions = predictions.flatten()
        actuals = actuals.flatten()
        
        for i in range(min(50, len(predictions))):  
            pred = predictions[i]
            actual = actuals[i]
            difference = pred - actual
            rounded_pred = np.round(pred)
            correct = rounded_pred == np.round(actual)
            f.write(f"{pred:.4f}\t{actual:.4f}\t{difference:.4f}\t{rounded_pred}\t{correct}\n")


def plot_actual_vs_predicted(Y_test, predictions):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(Y_test, predictions, alpha=0.5)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2, color='red')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.savefig('gen/actual_vs_predicted.png') 
    plt.show()

def main():
    # Create the 'gen' directory
    create_gen_directory()

    # File paths
    model_paths = ['best_model.h5']
    X_test_path = 'combined_features.npy'
    Y_test_path = 'target_values.npy'
    predictions_output = 'gen/new_predictions_detailed.txt'  
    
    print("Loading and preprocessing data...")
    X_test, Y_test, Y_test_scaled, target_scaler = load_and_preprocess_data(X_test_path, Y_test_path)
    
    all_predictions = []
    
    for model_path in model_paths:
        print(f"\nLoading model from {model_path}...")
        model = models.load_model(model_path, compile=False)
        
        print("\nMaking predictions...")
        predictions_scaled = model.predict(X_test)
        
        # Reverse scaling
        predictions = target_scaler.inverse_transform(predictions_scaled)
        
        all_predictions.append(predictions)
    
    average_predictions = np.mean(all_predictions, axis=0)

    mse, mae, r2 = evaluate_predictions(Y_test, average_predictions)
    
    print("\nTest Set Metrics (Averaged Predictions):")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Save predictions
    print("\nSaving detailed predictions...")
    save_predictions(average_predictions, Y_test, predictions_output)
    
    # Plot actual vs predicted
    print("\nPlotting actual vs predicted values...")
    plot_actual_vs_predicted(Y_test, average_predictions)
    

if __name__ == "__main__":
    main()
