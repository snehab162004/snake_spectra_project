import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import accuracy_score, classification_report

# Load and update the dataset
df = pd.read_csv('datasets_hackathon.csv')

# Function to check and update the Image_Path
def update_image_path(row):
    expected_path = f"snakes_images/{row['Name'].replace(' ', '_')}.jpeg"
    if not os.path.exists(row['Image_Path']):
        return expected_path
    return row['Image_Path']

# Correct the paths in the Image_Path column
df['Image_Path'] = df.apply(update_image_path, axis=1)

# Save the updated dataset to a new CSV file
df.to_csv('datasets_hackathon_updated.csv', index=False)

# Load the CSV files for training, validation, and testing
train_df = pd.read_csv('train_data.csv')
val_df = pd.read_csv('val_data.csv')
test_df = pd.read_csv('test_data.csv')

# Function to load and preprocess images
def load_and_preprocess_image(image_path, target_size=(128, 128)):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None
    img = load_img(image_path, target_size=target_size)  # Ensure all images are the same size
    img_array = img_to_array(img)
    img_array = img_array / 255.0 
    return img_array

# Load images and labels for training, validation, and testing
def load_data(df):
    images = []
    labels = []
    for index, row in df.iterrows():
        image_path = row['Image_Path']
        label = row['Label']
        img_array = load_and_preprocess_image(image_path)
        if img_array is not None:
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)

X_train, y_train = load_data(train_df)
X_val, y_val = load_data(val_df)
X_test, y_test = load_data(test_df)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
datagen.fit(X_train)

# Build a CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# Save the model
model.save('snake_detection_model.keras')

# Plot the training and validation accuracy and loss
history_df = pd.DataFrame(history.history)
history_df[['accuracy', 'val_accuracy']].plot()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

history_df[['loss', 'val_loss']].plot()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Load the trained model
model = load_model('snake_detection_model.keras')

# Function to predict the label of a single image
def predict_image(image_path):
    img = load_and_preprocess_image(image_path)
    if img is None:
        return None
    img = np.expand_dims(img, axis=0)  # Expand dimensions to match the input shape
    prediction = model.predict(img)
    return 1 if prediction[0] > 0.5 else 0

# Predict on the test dataset
y_pred = []
for image_path in test_df['Image_Path'].values:
    pred = predict_image(image_path)
    if pred is not None:
        y_pred.append(pred)
y_true = test_df['Label'].values[:len(y_pred)]

# Evaluate predictions
accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=['Non-Venomous', 'Venomous'])

print(f"Test accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Function to display snake details and image, and predict venomous status
def display_snake_details_and_image(color, scales, model):
    color = color.capitalize()
    scales = scales.capitalize()

    filtered_snakes = df[(df['Color'] == color) & (df['Scales'] == scales)].sort_values(by='Name')

    if filtered_snakes.empty:
        print("No matching snake found.")
    else:
        for index, row in filtered_snakes.iterrows():
            print(f"Snake Name: {row['Name']}")
            venomous = 'Yes' if row['Venomous'] == 'Yes' else 'No'
            print(f"Venomous: {venomous}")
            print(f"Snake Category: {row['Category']}")
            print(f"Most commonly found in: {row['Location']}")
            print(f"Family: {row['Family']}")
            print(f"Image Path: {row['Image_Path']}")
            print()

            # Display the image
            image_path = row['Image_Path']
            if os.path.exists(image_path):
                img = mpimg.imread(image_path)
                plt.imshow(img)
                plt.title(f"{row['Name']}")
                plt.axis('off')  # Hide axes for better visualization
                plt.show()
            else:
                print(f"Image not found for {row['Name']} at path: {image_path}")

            # Predict venomous or non-venomous
            prediction = 'Venomous' if predict_image(image_path) == 1 else 'Non-Venomous'
            print(f"Predicted: {prediction}")

            if prediction == 'Venomous':
                bite_location = input("Where has the snake bitten (e.g., arm, leg)?: ").strip().capitalize()
                print(f"First Aid for Venomous Snake Bite on {bite_location}:")
                print(venomous_first_aid)
            else:
                print("First Aid for Non-Venomous Snake Bite:")
                print(non_venomous_first_aid)

            print()

venomous_first_aid = """
1. Call emergency services immediately.
2. Keep the person calm and still to slow the spread of venom.
3. Position the bitten limb below the heart.
4. Remove tight clothing and jewelry near the bite area.
5. Clean the wound, but don't apply ice or a tourniquet.
6. Avoid using suction devices or cutting the bite area to remove venom, as this can cause more harm.
7. Clean the bite area with soap and water. Avoid applying ice or a tourniquet, as these can cause further tissue damage.
"""
non_venomous_first_aid = """
1. Clean the bite area with soap and water.
2. Apply an antiseptic.
3. Cover with a clean bandage.
4. Monitor for signs of infection.
"""

# Get user input for the color and type of scales
color_input = input("Enter the color of the snake: ").strip().capitalize()
scales_input = input("Enter the type of scales (Smooth/Scaly): ").strip().capitalize()

# Display snake details and image based on user input
display_snake_details_and_image(color_input, scales_input, model)
