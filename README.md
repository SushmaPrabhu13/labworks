**1.Develop a program to display grayscale image using read and write operation.**
import cv2
import numpy as np

image = cv2.imread('cat.jpg')
image = cv2.resize(image, (0, 0), None, 1.00, 1.00)

grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
grey_3_channel = cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)

numpy_horizontal = np.hstack((image, grey_3_channel))
numpy_horizontal_concat = np.concatenate((image, grey_3_channel), axis=1)

cv2.imshow('cat', numpy_horizontal_concat)
cv2.waitKey()

**output**
![image](https://user-images.githubusercontent.com/72405086/105021141-057c5680-5a6a-11eb-8a84-cbbf6439c83c.png)


