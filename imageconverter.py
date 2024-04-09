from PIL import Image
import numpy as np
from numpy import asarray
import os
import pickle

'''
This file is for taking images and converting them so that the
machine learning models can make predictions with them

This script doesnt work quite yet but the initiative is there
'''

all_images = np.array()

files = os.listdir("./input")
for file in files:
    image = Image.open(file)
    resized_image = image.resize((32,32))
    numpydata = asarray(resized_image)
    all_images.append(numpydata)
    
