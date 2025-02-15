import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

original_img = cv.imread("./notebook/images/neuron.jpg", 0)

_, mask = cv.threshold(original_img, 60, 255, cv.THRESH_BINARY)

# Dilation
# kernel = np.ones((5, 5), np.uint8)
# delated_img = cv.dilate(mask, kernel, iterations=2)

# Erosion
# kernel = np.ones((3, 3), np.uint8)
# eroded_img = cv.erode(mask, kernel, iterations=1)

# kernel = np.ones((2, 2), np.uint8)
# eroded_img2 = cv.erode(mask, kernel, iterations=1)

# Closing
# kernel1 = np.ones((9, 9), np.uint8)
# kernel2 = np.ones((20, 20), np.uint8)
# closed_img1 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel1)
# closed_img2 = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel2)

# Opening
# opened_img1 = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel1)
# opened_img2 = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel2)

# Gradient
gradient = cv.morphologyEx(mask, cv.MORPH_GRADIENT, kernel=np.ones((5, 5), np.uint8))

fig, (img1, img2, img3) = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
cmap_val = "gray"
img1.axis("Off")
img1.title.set_text("Original Mask")
img2.axis("Off")
img2.title.set_text("Eroded Image")
img3.axis("Off")
img3.title.set_text("Eroded Image")
img1.imshow(mask, cmap=cmap_val)
img2.imshow(gradient, cmap=cmap_val)
img3.imshow(gradient, cmap=cmap_val)

plt.show()