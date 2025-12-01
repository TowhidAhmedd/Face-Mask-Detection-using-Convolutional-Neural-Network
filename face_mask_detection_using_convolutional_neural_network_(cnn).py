
# Work Flow = Dataset -> Image processing -> Train Test Split -> CNN -> Evaluation

# Dependency
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from google.colab.patches import cv2_imshow
from PIL import Image
from sklearn.model_selection import train_test_split
from google.colab import drive
drive.mount('/content/drive')

# MyDrive
print(os.listdir("/content/drive/MyDrive"))

with_mask= "/content/drive/MyDrive/with_mask"
print(os.listdir(with_mask))  # folder

without_mask = "/content/drive/MyDrive/without_mask"
print(os.listdir(without_mask))  # subfolders




print('Number of with mask images:', len(with_mask)) # length of mask dataset
print('Number of without mask images:', len(without_mask)) # length of without mask dataset

"""Creating Labels for the two class of Images

with mask --> 1

without mask --> 0
"""

# create the label
# Dynamically get the number of files from the lists created in the image processing step
with_mask_files = os.listdir(with_mask)
without_mask_files = os.listdir(withot_mask)

num_with_mask_images = len(with_mask_files)
num_without_mask_images = len(without_mask_files)

# create the label for the two class of images
with_mask_label = [1] * num_with_mask_images
without_mask_label = [0] * num_without_mask_images

print(with_mask_label[0:5])
print(without_mask_label[0:5])

print(len(with_mask_label))
print(len(without_mask_label))


# combine with_mask_label + without_mask_label
labels = with_mask_label + without_mask_label
print(len(labels))
print(labels[0:5])
print(labels[-5:])

"""Diaplaing the images"""

# displaying with mask image
img = mpimg.imread(os.path.join(with_mask, 'with_mask_347.jpg')) # Using an image from the listed files in 'with_mask'
imgplot = plt.imshow(img)
plt.show()

# displaying without mask image
img = mpimg.imread(os.path.join(without_mask, 'without_mask_3555.jpg')) # Using an image from the listed files in 'without_mask'
imgplot = plt.imshow(img)
plt.show()

"""Image Processing

1.Resize the Images

2.Convert the images to numpy arrays
"""

# convert images to numpy arrays
data = []

with_mask_path = "/content/drive/MyDrive/with_mask/"
for img_file in with_mask_files:
  image = Image.open(with_mask_path+img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)


with_mask_path = "/content/drive/MyDrive/with_mask/"
for img_file in with_mask_files:
  image = Image.open(with_mask_path+img_file)
  image = image.resize((128,128))
  image = image.convert('RGB')
  image = np.array(image)
  data.append(image)

len(data) # length of data
type(data) # type of data
data[0] # print first data
type(data[0]) # first datatype
data[0].shape # first datatype shape


# convert images to numpy arrays
data = []
IMG_SIZE = 128 # Define a standard image size for resizing

# Process with_mask images
with_mask_files = os.listdir(with_mask)
for img_file in with_mask_files:
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(with_mask, img_file)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # Ensure all images are 3 channels (RGB)
                if len(img.shape) == 2: # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 1: # Grayscale image with 1 channel
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 3: # Color image (BGR from imread)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
                else: # Handle other cases if necessary, or skip this image
                    print(f"Skipping {img_file} due to unsupported channel format: {img.shape}")
                    continue
                data.append(img)
            else:
                print(f"Warning: Could not read image file: {img_file} from with_mask")
        except Exception as e:
            print(f"Error processing with_mask/{img_file}: {e}")

# Process without_mask images
without_mask_files = os.listdir(without_mask)
for img_file in without_mask_files:
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(without_mask, img_file)
        try:
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                # Ensure all images are 3 channels (RGB)
                if len(img.shape) == 2: # Grayscale image
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 1: # Grayscale image with 1 channel
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                elif img.shape[2] == 3: # Color image (BGR from imread)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
                else: # Handle other cases if necessary, or skip this image
                    print(f"Skipping {img_file} due to unsupported channel format: {img.shape}")
                    continue
                data.append(img)
            else:
                print(f"Warning: Could not read image file: {img_file} from without_mask")
        except Exception as e:
            print(f"Error processing without_mask/{img_file}: {e}")


# converting image list and lavel list to numpy arrays
X = np.array(data)
Y = np.array(labels)

print(X.shape)
print(Y.shape)
print(Y)


"""Train Test Split"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

print(X.shape, X_train.shape, X_test.shape,Y_train.shape, Y_test.shape)


# scaling the data

X_train_scaled = X_train/255
X_test_scaled = X_test/255

X_train[0]
X_train_scaled[0]


"""Building a Convolutional Neural Networks (CNN)"""

# !pip install tensorflow
import tensorflow as tf
from tensorflow import keras

num_of_classes = 2

model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))


# compile the neural network
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])


# training the neural network
history = model.fit(X_train_scaled, Y_train, validation_split=0.1, epochs=5)


"""Model Evaluation"""

loss, accuracy = model.evaluate(X_test_scaled, Y_test)
print('Test Accuracy =', accuracy)


# show line using matplotlib
h = history

# plot the loss value
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label = 'validation loss')
plt.legend()
plt.show()

# plot the accuracy value
plt.plot(h.history['acc'], label='train accuracy')
plt.plot(h.history['val_acc'], label='validation accuracy')
plt.legend()  # Display the legend
plt.show()


"""Predictive System"""

input_image_path = input('Path of the image to be predicted: ')

input_image = cv2.imread(input_image_path)

cv2_imshow(input_image)

input_image_resized = cv2.resize(input_image,(128,128))

input_image_scaled = input_image_resized/255

input_image_reshaped = np.reshape(input_image_scaled,[1,128,128,3])

input_prediction = model.predict(input_image_reshaped)

print(input_prediction)


input_pred_label = np.argmax(input_prediction)
print(input_pred_label)

if input_pred_label == 1:
  print('The person in the image is wearing a mask')

else:
  print('The person in the image is not wearing a mask')









