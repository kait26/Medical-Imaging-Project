import numpy as np
import math
import random
import cv2

class NoiseGen:

    def __init__(self, image, noise_name, mean, var):
        self.image = image
        self.noise_name = noise_name
        self.mean = mean
        self.var = var

        if noise_name == 'gaussian':
            self.noise = self.noiseGaussian
        if noise_name == 'bipolar':
            self.noise = self.noiseBipolar
    
    def noiseGaussian(self, mean, var):
        image = np.array(self.image/255, dtype=float)
        x, y = image.shape[0], image.shape[1]
        sigma = var ** 0.5
        gaussian = np.random.normal(mean, sigma ** 0.5, (x, y))
        noisy_image = image + gaussian
        return noisy_image

    def noiseBipolar( self, noise_proba, noise_probb):
        noisy_image = self.image.copy()
        rows, cols = self.image.shape

        for i in range(rows):
            for j in range(cols):
                n = np.random.random()
                if n < 0.5:
                    n = np.random.random()
                    if n < noise_proba:
                        noisy_image[i][j] = 0
                    else:
                        noisy_image[i][j] = self.image[i][j]
                else:
                    n = np.random.random()
                    if n < noise_probb:
                        noisy_image[i][j] = 255
                    else:
                        noisy_image[i][j] = self.image[i][j]

        return noisy_image


    def generator(self):
        result = self.image

        if self.noise_name == 'gaussian':
            result = NoiseGen.noiseGaussian(self, self.mean, self.var)
        
        if self.noise_name == 'bipolar':
            result = NoiseGen.noiseBipolar(self, self.mean, self.var)
        
        return result

