import os
import cv2
import numpy as np
import pickle

'''
This file is for taking images and converting them so that the
machine learning models can make predictions with them

This script doesnt work quite yet but the initiative is there
'''
# Function to resize image to 32x32
def resize_image(image):
    blur_image = cv2.blur(image, (10, 10))
    return cv2.resize(blur_image, (32, 32))

# Function to load images from folder, resize, and convert to numpy array
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            if image is not None:
                resized_image = resize_image(image)
                images.append(resized_image)
    return np.array(images)

input_folder = "./input"
output_file = "output.pkl"
images = load_images(input_folder)

with open(output_file, 'wb') as f:
    pickle.dump(images, f)
