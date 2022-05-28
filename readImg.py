import numpy as np
import cv2 as cv
from PIL import Image


img = Image.open('Arthur.png')
img = img.convert('L')
    
cols,rows = img.size
    
Value = [[0]*cols for i in range(rows)]
    
for x in range(0,rows):
    for y in range(0,cols):
        img_array = np.array(img)
        v = img_array[x,y]
        Value[x][y] = v
    
print(Value)
