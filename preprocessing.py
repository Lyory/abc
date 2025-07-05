# Quân

import numpy as np
import cv2
import glob

def maximize_contrast_clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)

def preprocess_image(img_input):
    img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    img_contrast = maximize_contrast_clahe(img_gray)
    img_blurred = cv2.GaussianBlur(img_contrast, (5, 5), 0)
    _, img_binary = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_binary

img = cv2.imread('samples/ArUco.jpg')  
img_result = preprocess_image(img)

cv2.imshow('Original', img)
cv2.imshow('Preprocessed', img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
