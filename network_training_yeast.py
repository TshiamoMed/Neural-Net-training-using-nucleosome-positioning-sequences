import numpy as np 
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

output_dir = 'yeast_NN'
os.makedirs(output_dir, exist_ok=True)

def preprocess_data(Y_train, Y_val, Y_test):
    """Enhanced preprocessing with log transform and standardization"""
    epsilon = 1e-8
    Y_train_log = np.log1p(Y_train + epsilon)
    Y_val_log = np.log1p(Y_val + epsilon)
    Y_test_log = np.log1p(Y_test + epsilon)
    
    scaler = StandardScaler()
    Y_train_scaled = scaler.fit_transform(Y_train_log.reshape(-1, 1))  
    Y_val_scaled = scaler.transform(Y_val_log.reshape(-1, 1))
    Y_test_scaled = scaler.transform(Y_test_log.reshape(-1, 1))
    
    return Y_train_scaled, Y_val_scaled, Y_test_scaled, scaler


def create_model(input_shape=(645,)):
    """Model with stronger regularization"""
    l2_reg = regularizers.l2(0.001)  # Stronger L2 regularization

    inputs = layers.Input(shape=input_shape)
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(1024, kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.3)(x)

    units = [512, 256, 128, 64]
    for unit in units:
        shortcut = layers.Dense(unit)(x)
        x = layers.Dense(unit, kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dense(unit, kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([shortcut, x])
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
    return model

def custom_loss():
    """Huber loss with MSE"""
    def loss(y_true, y_pred):
        huber = tf.keras.losses.Huber(delta=1.0)(y_true, y_pred)
        mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
        return 0.7 * huber + 0.3 * mse
    return loss

def inverse_transform_predictions(predictions, scaler):
    predictions_unscaled = scaler.inverse_transform(predictions)
    return np.expm1(predictions_unscaled)

def plot_metrics(history):
    """Plot training and validation loss and MAE."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_plot.png')
    plt.show()


def plot_actual_vs_predicted(Y_test, predictions_final):
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test.flatten(), predictions_final.flatten(), c='blue', label='Predicted')
    plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red', linestyle='--', label='Perfect Fit')
    plt.title('Actual vs Predicted')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.legend(loc='best')
    plt.savefig(f'{output_dir}/actual_vs_predicted.png')
    plt.show()

# Cross-validation
def cross_validation(X, Y, k=2):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_no = 1
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Model creation
        model = create_model(input_shape=(X_train.shape[1],))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=custom_loss(),
            metrics=['mae']
        )

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-4),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1),
            callbacks.ModelCheckpoint(f'{output_dir}/best_model_fold_{fold_no}.h5', monitor='val_loss', save_best_only=True, verbose=1)
        ]

        print(f"Training fold {fold_no}...")
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=120,
            batch_size=300,
            callbacks=callbacks_list,
            verbose=1
        )

        plot_metrics(history)
        fold_no += 1

# Main execution
if __name__ == "__main__":
    print("Loading data...")
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    X_val = np.load('X_val.npy')
    Y_val = np.load('Y_val.npy')
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')

    # Preprocess target variable
    Y_train_scaled, Y_val_scaled, Y_test_scaled, target_scaler = preprocess_data(Y_train, Y_val, Y_test)


    # Perform cross-validation
    cross_validation(X_train, Y_train_scaled, k=2)

    # Train final model on full data
    model = create_model(input_shape=(X_train.shape[1],))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=custom_loss(),
        metrics=['mae']
    )

    # Train and evaluate final model
    print("\nFinal model training...")
    history = model.fit(
        X_train, Y_train_scaled,
        epochs=120,
        batch_size=300,
        validation_split=0.1,
        callbacks=[
            callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, min_delta=1e-4),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6, verbose=1)
        ],
        verbose=1
    )

    # Predictions and inverse transformation
    predictions_final = model.predict(X_test)
    predictions_final = inverse_transform_predictions(predictions_final, target_scaler)
    Y_test = inverse_transform_predictions(Y_test_scaled, target_scaler)

    # Calculate and print metrics
    plot_actual_vs_predicted(Y_test, predictions_final)
    calculate_and_print_stats(Y_test, predictions_final)
    plot_error_distribution(Y_test, predictions_final)
    
    mse = mean_squared_error(Y_test, predictions_final.flatten())
    mae = np.mean(np.abs(Y_test.flatten() - predictions_final.flatten()))
    r2 = r2_score(Y_test, predictions_final.flatten())


    print("\nFinal Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R-squared Score: {r2:.4f}")
  

def save_predictions_and_errors(Y_test, predictions_final):
    """Save actual values, predictions, and differences (errors) to a CSV file."""
    errors = Y_test.flatten() - predictions_final.flatten()

    # Create a DataFrame with Actual, Predicted, and Error values
    results_df = pd.DataFrame({
        'Actual': Y_test.flatten(),
        'Predicted': predictions_final.flatten(),
        'Error': errors
    })

    # Save the DataFrame to a CSV file
    results_df.to_csv(f'{output_dir}/predictions_and_errors.csv', index=False)

    print(f"Predictions, actual values, and errors saved to {output_dir}/predictions_and_errors.csv")

# Main execution (part of the existing main section)

# After predictions and stats calculation:
save_predictions_and_errors(Y_test, predictions_final)

