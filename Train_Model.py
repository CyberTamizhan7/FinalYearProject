import os
import json
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

#Hi


# Start Time
start = time.perf_counter()


# Parameters
IMAGE_SIZE = (100, 100)
BATCH_SIZE = 32
EPOCHS = 16

# Dataset path
dataset_path = r"C:\Users\admin\Desktop\Final Year Project\Dataset\chess_pieces"

# === No need to preprocess folders, since they are already flattened ===

# Load dataset
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Class names
class_names = list(train_generator.class_indices.keys())
print("Classes:", class_names)

# Build CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Save class labels in same order as training
with open("class_labels.json", "w") as f:
    json.dump(class_names, f)

# Save model
model.save("Chess_22.h5")
print("Model saved as Chess_22.h5")

# Plot accuracy and loss
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.show()



# End Time
end = time.perf_counter()

total_seconds = end-start

hours = int(total_seconds//3600)
minutes = int((total_seconds%3600)//60)
seconds = total_seconds%60

print(f"Total Running Time : {hours}hours {minutes}minutes {seconds:.2f}seconds")
