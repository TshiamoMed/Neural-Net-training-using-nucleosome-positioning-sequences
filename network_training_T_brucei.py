import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Directory for saving outputs
output_dir = 'training_outputs'
os.makedirs(output_dir, exist_ok=True)

def preprocess_data(Y_train, Y_val, Y_test):
    """Preprocess only target data since X is already normalized in a one hot encoding"""
    target_scaler = MinMaxScaler()
    Y_train_scaled = target_scaler.fit_transform(Y_train.reshape(-1, 1))
    Y_val_scaled = target_scaler.transform(Y_val.reshape(-1, 1))
    Y_test_scaled = target_scaler.transform(Y_test.reshape(-1, 1))
    
    return Y_train_scaled, Y_val_scaled, Y_test_scaled, target_scaler

def create_model(input_shape):
    """Enhanced model architecture with skip connections"""
    inputs = layers.Input(shape=input_shape)
    
    # Initial processing
    x = layers.Dense(512)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.3)(x)
    
    # Multiple blocks with skip connections
    for units in [256, 128, 64]:
        skip = x
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dense(units)(x)
        x = layers.BatchNormalization()(x)
        
        # Add skip connection if shapes match
        if skip.shape[-1] == units:
            x = layers.Add()([x, skip])
        x = layers.ReLU()(x)
        x = layers.Dropout(0.2)(x)
    
    # Output layer
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs, outputs)
    return model

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{output_dir}/training_loss.png')
    plt.close()

def plot_predictions(y_true, y_pred):
    """Plot actual vs predicted values with correlation line"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.grid(True)
    plt.savefig(f'{output_dir}/predictions.png')
    plt.close()

def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive set of metrics"""
    mse = mean_squared_error(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'R2': r2
    }

if __name__ == "__main__":
    print("Loading data...")
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    X_val = np.load('X_val.npy')
    Y_val = np.load('Y_val.npy')
    X_test = np.load('X_test.npy')
    Y_test = np.load('Y_test.npy')

    # Preprocess only Y data
    Y_train_scaled, Y_val_scaled, Y_test_scaled, target_scaler = preprocess_data(
        Y_train, Y_val, Y_test)

    # Create and compile model
    model = create_model(input_shape=(X_train.shape[1],))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='huber',  # Huber loss is more robust to outliers
        metrics=['mae', 'mse']
    )
    model.summary()

    # Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            min_delta=1e-6
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            f'{output_dir}/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, Y_train_scaled,
        validation_data=(X_val, Y_val_scaled),
        epochs=200,
        batch_size=128,
        callbacks=callbacks_list,
        verbose=1
    )

    # Plot training history
    plot_training_history(history)

    # Make predictions
    print("\nMaking predictions...")
    predictions_scaled = model.predict(X_test)
    predictions = target_scaler.inverse_transform(predictions_scaled)

    # Calculate and display metrics
    metrics = calculate_metrics(Y_test, predictions)
    print("\nFinal Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Plot predictions
    plot_predictions(Y_test, predictions)

    # Save detailed predictions
    with open(f'{output_dir}/predictions_detailed.txt', 'w') as f:
        f.write("Predicted\tActual\tDifference\tRounded_Pred\tCorrect\n")
        for i in range(min(50, len(predictions))):
            pred = predictions[i][0]
            actual = Y_test[i][0]
            difference = pred - actual
            rounded_pred = round(pred)
            correct = rounded_pred == round(actual)
            f.write(f"{pred:.4f}\t{actual:.4f}\t{difference:.4f}\t{rounded_pred}\t{correct}\n")
            print(f"{pred:.4f}\t{actual:.4f}\t{difference:.4f}\t{rounded_pred}\t{correct}")