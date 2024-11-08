import torch

from fastapi import FastAPI, Body
from fastapi.staticfiles import StaticFiles
from myapp.model import Model
#import cv2
import numpy as np
import logging
from torchvision.transforms import ToTensor, Compose, Resize, Normalize
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def custom_dilate(img, kernel):
    dilated_img = np.zeros_like(img)
    rows, cols = img.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if np.any(img[i-1:i+2, j-1:j+2].dot(kernel)):
                dilated_img[i, j] = 255

    return dilated_img


# load model
model = Model()

# app
app = FastAPI(title='Symbol detection', docs_url='/docs')

# api
@app.post('/api/predict')
def predict(image: str = Body(..., description='image pixels list')):

    image = np.reshape(list(map(int, image[1:-1].split(','))),(28, 28))
    image = np.array(image, np.uint8)
    print(image.shape)
    image = image.transpose()
    print(image.shape)
    
    p = 3
    kernel = np.ones((p, p), np.uint8)
    #image = cv2.dilate(image, kernel, iterations=1)
    image = custom_dilate(image, kernel)
    transform = Compose([
    ToTensor(),
    Normalize([0.5], [0.5])
    ])

    image = transform(image).unsqueeze(1)
    
    
    
    pred = model.predict(image)
    return {'prediction': pred}

# static files
app.mount('/', StaticFiles(directory='static', html=True), name='static')
