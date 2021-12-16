import numpy as np
import cv2
from Noise.Noise_Gen import NoiseGen

# Displays image
def display_image(before, after):
    cv2.imshow("BEFORE", before)
    cv2.imshow("AFTER", after)
    cv2.waitKey(0)

def main():
    # Read in image
    img = cv2.imread("sample.png", 0)

    # Pick Noise
    noisyImage_obj = NoiseGen(img, "gaussian", 0, 0.001)
    # noisyImage_obj = NoiseGen(img, "bipolar", 0, 0.1)

    # Generate Noise and Display
    output = noisyImage_obj.generator()
    display_image(img, output)

if __name__ == "__main__":
    main()

