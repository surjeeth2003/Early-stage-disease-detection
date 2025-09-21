# train
# src/train.py

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# --- 1. SETTINGS AND HYPERPARAMETERS ---

# Image dimensions
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32

# Directories
# Assumes you run the script from the root of your project directory
BASE_DIR = os.getcwd() 
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'results', 'models', 'baseline_model.h5')

# --- 2. DATA PREPARATION ---

# For the baseline, we ONLY rescale the images. No augmentation!
# Rescaling pixel values from [0, 255] to [0, 1] is a standard practice.
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators from the directories
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # for multi-class classification
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get the number of classes automatically from the generator
num_classes = len(train_generator.class_indices)
print(f"Found {num_classes} classes.")


# --- 3. MODEL BUILDING (TRANSFER LEARNING) ---

# Load the MobileNetV2 model, pre-trained on ImageNet
# include_top=False means we don't include the final classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))

# Freeze the layers of the base model so we don't retrain them
for layer in base_model.layers:
    layer.trainable = False

# Add our custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x) # Averages the spatial information
x = Dense(1024, activation='relu')(x) # A fully-connected layer
# The final layer has 'num_classes' neurons and 'softmax' for probability distribution
predictions = Dense(num_classes, activation='softmax')(x)

# This is the final model we will train
model = Model(inputs=base_model.input, outputs=predictions)


# --- 4. MODEL COMPILATION ---

# We use the Adam optimizer, and since it's multi-class, categorical_crossentropy is the loss function.
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()


# --- 5. MODEL TRAINING ---

print("Starting model training...")
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=10, # Start with 10 epochs for the baseline
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

print("Training finished.")

# --- 6. SAVE THE MODEL ---

# Ensure the directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")