# DL - Human Emotion Detection

## Introduction
This project aims to detect human emotions from images using deep learning techniques. We build and train a convolutional neural network (CNN) to classify images into one of seven emotions: anger, contempt, disgust, fear, happiness, sadness, and surprise.

## Table of Contents
1. [Libraries and Dependencies](#libraries-and-dependencies)
2. [Data Loading and Preprocessing](#data-loading-and-preprocessing)
3. [Model Architecture](#model-architecture)
4. [Training the Model](#training-the-model)
5. [Model Evaluation](#model-evaluation)
6. [Real-Time Emotion Detection](#real-time-emotion-detection)
7. [Conclusion](#conclusion)

## Libraries and Dependencies
The project utilizes several key libraries for data handling, model building, and evaluation:
- `os`
- `cv2`
- `matplotlib`
- `numpy`
- `sklearn`
- `tensorflow`
- `keras`

### Importing Required Libraries
```python
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle
from keras.utils import np_utils
import tensorflow as tf
import tensorflow.keras as k
from sklearn.model_selection import train_test_split
import pickle
```

## Data Loading and Preprocessing

### Loading Data
We load the image data from the specified directory and preprocess it by resizing and scaling the images.
```python
path_image = r"D:\Courses language programming\6_Deep Learning\Deep Learning Project\Folder\Human Emotion Detection from Images\CK+48"
data_list = os.listdir(path_image)
img_data = []

for data_set in data_list:
    img_list = os.listdir(path_image + "/" + data_set)
    print(f"Loading image - {data_set}")
    for image in img_list:
        input_img = cv2.imread(path_image + "/" + data_set + "/" + image)
        input_img_new = cv2.resize(input_img, (100, 100))
        img_data.append(input_img_new)

data = np.array(img_data).astype("float32") / 255.0
print(data.shape)
```

### Preparing Data
We prepare the data by labeling the images according to their respective emotions and splitting the data into training and testing sets.
```python
num_class = len(data_list)
num_sample = data.shape[0]
label = np.ones((num_sample,), dtype="int64")

label[0:134] = 0  # Anger
label[135:188] = 1  # Contempt
label[189:365] = 2  # Disgust
label[366:440] = 3  # Fear
label[441:647] = 4  # Happy
label[648:731] = 5  # Sadness
label[732:980] = 6  # Surprise

names = data_list
image_label = np_utils.to_categorical(label, num_class)
x_img, y_img = shuffle(data, image_label, random_state=2)
x_train, x_test, y_train, y_test = train_test_split(x_img, y_img, train_size=0.8, random_state=42)
```

### Sample Data Visualization
We visualize some sample images from the dataset.
```python
plt.figure(figsize=(10, 10))
for i, j in enumerate(list(np.random.randint(0, len(x_img), 36))):
    plt.subplot(6, 6, i + 1)
    plt.imshow(x_img[j])
    plt.axis("off")
    plt.title(names[list(y_img[j].astype(int)).index(1)])
plt.show()
```

## Model Architecture

### Building the CNN Model
We construct a CNN model using TensorFlow and Keras.
```python
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation

model_CNN = k.models.Sequential([
    Conv2D(6, (5, 5), input_shape=(100, 100, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(32, (5, 5), padding="valid", activation="relu"),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), padding="same", activation="relu"),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(7, activation="softmax")
])

model_CNN.summary()
model_CNN.compile(optimizer="adam", loss=k.losses.CategoricalCrossentropy(), metrics=["accuracy"])
```

## Training the Model
We train the CNN model with the training data and validate it using the testing data.
```python
history = model_CNN.fit(x_train, y_train, epochs=100, validation_split=0.4, validation_data=(x_test, y_test), validation_steps=1)
```

### Training and Validation Loss
We plot the training and validation loss over epochs.
```python
plt.plot(history.history["loss"], label="Loss")
plt.plot(history.history["val_loss"], label="Val_Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Vs Epochs")
plt.legend()
plt.grid()
plt.show()
```

### Training and Validation Accuracy
We plot the training and validation accuracy over epochs.
```python
plt.plot(history.history["accuracy"], label="Accuracy")
plt.plot(history.history["val_accuracy"], label="Val_Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy Vs Epochs")
plt.legend()
plt.grid()
plt.show()
```

## Model Evaluation
We evaluate the model's performance on the testing data.
```python
loss, accuracy = model_CNN.evaluate(x_test, y_test)
print(f"The Testing Accuracy is ==> {round(accuracy * 100, 3)}%")
print(f"The Testing Loss is ==> {round(loss, 3)}")
```

## Real-Time Emotion Detection

### Saving the Model
We save the trained model for future use.
```python
pickle.dump(model_CNN, open(r"D:\Pycharm\model_pickle\human_emotion.bin", "wb"))
```

### Loading the Model
We can load the saved model for predictions.
```python
model = pickle.load(open(r"D:\Pycharm\model_pickle\human_emotion.bin", "rb"))
```

### Predicting Emotion from an Image
We implement a system to predict emotions from a single image.
```python
path_image = input("Please Enter The Path of Image: ")
image = cv2.imread(path_image)
image = cv2.resize(image, (100, 100))
new_image = np.reshape(image, [1, 100, 100, 3])

new_predict = model_CNN.predict(new_image)
print(f"The Predicted Emotion is ==> {names[new_predict.argmax()]}")
plt.imshow(image, cmap="gray")
plt.title(names[new_predict.argmax()])
plt.show()
```

### Real-Time Emotion Detection
We implement a real-time emotion detection system using OpenCV.
```python
camera = cv2.VideoCapture(0)
face_detect = cv2.CascadeClassifier(r"D:\Pycharm\Computer Vision\haar cascade files\haarcascade_frontalface_default.xml")

while True:
    _, images = camera.read()
    face = face_detect.detectMultiScale(images)

    for (x, y, w, h) in face:
        cv2.rectangle(images, (x, y), (x + w, y + h), (0, 255, 0), 1)
        face_pre = images[y:y + h, x:x + w]
        images_a = cv2.resize(face_pre, (100, 100))
        new_image = np.reshape(images_a, [1, 100, 100, 3])
        new_predict = model_CNN.predict(new_image)
        print(names[new_predict.argmax()])

    cv2.imshow("image", images)
    if cv2.waitKey(60) & 0xff == ord("o"):
        break
```

## Conclusion
This project showcases the implementation of a deep learning model to detect human emotions from images. The model achieved an impressive accuracy, demonstrating the potential of deep learning in emotion recognition tasks. Further improvements could involve experimenting with different network

 architectures and optimizing hyperparameters.
