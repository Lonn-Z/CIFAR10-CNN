import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

# images = 32 x 32 x 3 (RGB), 10 labels (0-9)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize color intensity for R, G, B
train_images, test_images = train_images / 255, test_images / 255

# Each name aligns with encoded class num (e.g. 0 = Plane)
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Image visualization
for image in range(16):
    plt.subplot(4, 4, image+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[image], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[image][0]])

plt.show()

# Train and test caps
max_train = int(input("How many training examples do you wish to use? Max = 50,000\n"))
max_test = int(input("How many testing examples do you wish to use? Max = 10,000\n"))
train_images, train_labels = train_images[:max_train], train_labels[:max_train]
test_images, test_labels = test_images[:max_test], test_labels[:max_test]

# Building the CNN
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=train_images.shape[1:]))

model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.add(layers.Flatten()) # To make outputs into a rolled vector
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(class_names), activation='softmax'))

# Training the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluations
loss, accuracy = model.evaluate(test_images, test_labels)
accuracy = accuracy * 100
print(f"Loss: {loss}\nAccuracy: {accuracy}%")
model.save("Image_classifier.model")