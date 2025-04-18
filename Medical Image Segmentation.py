# Medical Image Segmentation
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler

# Generate synthetic image data with circular features for segmentation
def generate_synthetic_data(num_samples=100, img_size=(128, 128)):
    images, masks = [], []
    
    for _ in range(num_samples):
        img = np.zeros(img_size, dtype=np.uint8)
        mask = np.zeros(img_size, dtype=np.uint8)
        center = (np.random.randint(32, 96), np.random.randint(32, 96))
        radius = np.random.randint(10, 30)
        
        # Add circular feature to the image
        cv2.circle(img, center, radius, 255, -1)
        # Create the corresponding mask with the circle
        cv2.circle(mask, center, radius, 255, -1)
        
        images.append(img / 255.0)  # Normalize the image
        masks.append(mask / 255.0)  # Normalize the mask
        
    return np.expand_dims(np.array(images), -1), np.expand_dims(np.array(masks), -1)

# Generate the dataset
X, y = generate_synthetic_data()

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1)

# Define the U-Net model architecture
def unet_model(input_size=(128, 128, 1)):
    inputs = tf.keras.Input(input_size)

    # Encoder block
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, 3, activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D()(c3)

    # Bottleneck block
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    # Decoder block
    u5 = layers.UpSampling2D()(c4)
    u5 = layers.concatenate([u5, c3])
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(u5)
    c5 = layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u6 = layers.UpSampling2D()(c5)
    u6 = layers.concatenate([u6, c2])
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D()(c6)
    u7 = layers.concatenate([u7, c1])
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(16, 3, activation='relu', padding='same')(c7)

    # Output layer with sigmoid activation for binary segmentation
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(c7)

    return Model(inputs, outputs)

# Initialize the model
model = unet_model()

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Set up callbacks for model checkpointing and early stopping
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

# Learning rate scheduler function to decrease learning rate after 5 epochs
def lr_scheduler(epoch, lr):
    return lr if epoch < 5 else lr * 0.9

lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# Train the model with the training data and validate with the validation set
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=8,
    callbacks=[checkpoint, early_stopping, lr_scheduler_callback]
)

# Plot training and validation loss to visualize the model's learning progress
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# Function to display the input image, ground truth mask, and predicted mask
def predict_and_display(index=0):
    sample_img = X_val[index]
    ground_truth = y_val[index]

    # Predict the mask using the trained model
    prediction = model.predict(np.expand_dims(sample_img, axis=0))[0]
    prediction = (prediction > 0.5).astype(np.uint8)

    # Convert input image to RGB for visualization
    rgb_img = np.repeat(sample_img, 3, axis=-1)

    # Plot the input image, ground truth, and predicted mask side by side
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title("Input Image")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth.squeeze(), cmap='plasma')
    plt.title("Ground Truth Mask")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plt.subplot(1, 3, 3)
    plt.imshow(prediction.squeeze(), cmap='viridis')
    plt.title("Predicted Mask")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")

    plt.tight_layout()
    plt.show()

# Test the model prediction and display the results
predict_and_display(index=0)
