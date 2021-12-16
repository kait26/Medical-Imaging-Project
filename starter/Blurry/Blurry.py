import cv2
import numpy as np

image = cv2.imread('blurry1.jpg', flags=cv2.IMREAD_COLOR)

cv2.imshow('blurry1', image)
cv2.waitKey()
#cv2.destroyAllWindows()

kernel = np.array([[0, -1, 0],
                [-1, 5,-1],
                [0, -1, 0]])
image_sharp = cv2.filter2D(src=image, ddepth=-10, kernel=kernel)
cv2.imshow('blurry1 Sharpened', image_sharp)
cv2.waitKey()
cv2.destroyAllWindows()
