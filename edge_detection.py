import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("./notebook/images/road_in_norway.jpg", 0)
image_noise_removed = cv.GaussianBlur(image, (3, 3), 0)

# Laplacian
laplacian = cv.Laplacian(image_noise_removed, cv.CV_64F)

# Sobelx
sobelx = cv.Sobel(image_noise_removed, cv.CV_64F, 1, 0, ksize=5)

# Sobely
sobely = cv.Sobel(image_noise_removed, cv.CV_64F, 0, 1, ksize=5)

# Canny Edge Detection
canny = cv.Canny(image_noise_removed,  100, 300)

plt.subplot(2, 2, 1), plt.imshow(image, cmap='gray')
plt.title("Original")
plt.subplot(2, 2, 2), plt.imshow(canny, cmap='gray')
plt.title("Canny Edge Detection")
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title("Sobel X")
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title("Sobel Y")

plt.show()
