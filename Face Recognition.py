
import os
import numpy as np
import tarfile
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# Extract the dataset
with tarfile.open('C:/Users/musta/Desktop/FaceRecognition/lfw-funneled.tgz', 'r:gz') as file:
    file.extractall(path='lfw-funneled/lfw_funneled')



# Parameters
img_width, img_height = 96, 96

channels = 3

# Paths
data_dir = 'lfw-funneled/lfw_funneled'


# Lists to hold images and labels
images = []
labels = []

# Load and preprocess the images
valid_extensions = ['.jpg', '.jpeg', '.png']  # You can add more extensions if necessary

for dir_name in os.listdir(data_dir):
    print("Processing directory:", dir_name)

    dir_path = os.path.join(data_dir, dir_name)
    if os.path.isdir(dir_path):
        print(dir_name, "is a directory.")

        for img_name in os.listdir(dir_path):
            if any(img_name.endswith(ext) for ext in valid_extensions):  # Check if the file has a valid image extension
                img_path = os.path.join(dir_path, img_name)
                print("Loading image:", img_name)

                img = load_img(img_path, target_size=(img_width, img_height))
                img_array = img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(dir_name)


images = np.array(images)
labels = np.array(labels)

# Convert string labels to integers
encoder = LabelEncoder()
encoded_labels = encoder.fit_transform(labels)
categorical_labels = to_categorical(encoded_labels)



# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow(images, categorical_labels, subset='training')
val_generator = datagen.flow(images, categorical_labels, subset='validation')

# Revised Model with Batch Normalization and weight initializers
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', input_shape=(img_width, img_height, channels)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Optional: Add another convolutional layer without max-pooling
model.add(Conv2D(128, (2, 2), activation='relu', kernel_initializer='he_normal'))  # Note the filter size is reduced to 2x2
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(len(encoder.classes_), activation='softmax'))


# Compile the model with a custom learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# Early stopping & checkpointing the best model
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
]
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * (drop ** np.floor((1 + epoch) / epochs_drop))
    return lrate

lrate_schedule = LearningRateScheduler(step_decay)
callbacks.append(lrate_schedule)  # Add the new scheduler to the list of callbacks

# Train the model using data augmentation
model.fit(
    train_generator,
    validation_data=val_generator,
    batch_size=32,
    epochs=50,
    callbacks=callbacks
)

# To predict on a new image
def predict_image(image_path):
    new_img = load_img(image_path, target_size=(img_width, img_height))
    new_img_array = img_to_array(new_img) / 255.0
    new_img_array = np.expand_dims(new_img_array, axis=0)
    prediction = model.predict(new_img_array)
    predicted_class = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_class[0]

# Example usage of prediction
# predicted_person = predict_image('path_to_new_image.jpg')
# print(predicted_person)

