import io
import cv2
import time
import numpy as np

from PIL import Image

class JPEG():
    def __init__(self, img, N=8):
        self.img = np.array(img.convert())

        # Constants
        self.N = N
        self.T = np.array([self.cosine(x, np.arange(N)) for x in range(N)])
        self.Q = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68,109,103, 77],
            [24, 35, 55, 64, 81,104,113, 92],
            [49, 64, 78, 87,103,121,120,101],
            [72, 92, 95, 98,112,100,103, 99]
        ])
        self.compression = 8

        self.rgb_to_ycbcr()
        self.downsample()
        self.split_blocks()

    def rgb_to_ycbcr(self):
        R = self.img[:,:,0]/256.
        G = self.img[:,:,1]/256.
        B = self.img[:,:,2]/256.

        self.img[:,:,0] = 16. + 65.738 * R + 129.057 * G + 25.064 * B
        self.img[:,:,2] = 128. - 37.945 * R - 74.494 * G + 112.439 * B
        self.img[:,:,1] = 128. + 112.439 * R - 94.154 * G - 18.285 * B

    def downsample(self, scale=4):
        img = self.img
        cb = img[::scale,::scale,1]
        cr = img[::scale,::scale,2]

        y = img[:,:,0]
        cb = np.kron(cb, np.ones((scale, scale)))[:img.shape[0],:img.shape[1]]
        cr = np.kron(cr, np.ones((scale, scale)))[:img.shape[0],:img.shape[1]]

        self.img = np.stack([y, cb, cr], axis=2)

    def split_blocks(self, N=8):
        self.compression = 0

        while True:
            img = np.copy(self.img)
            x_overflow = img.shape[0] % N
            y_overflow = img.shape[1] % N
            img = img[:-x_overflow,:-y_overflow,:]

            # Access an 8x8 block from the image: 
            # x, y = the indices for each 8x8 block
            # ch = channel from index 0 to 2 for y, cb, cr
            index_block = lambda x, y, ch: img[x*N:(x+1)*N, y*N:(y+1)*N,ch]

            self.compression += 1
            print(self.compression)

            for ch in range(3):
                for x in range(img.shape[0] // N - 1):
                    for y in range(img.shape[1] // N - 1):
                        block = index_block(x, y, ch)
                        img[x*N:(x+1)*N, y*N:(y+1)*N, ch] = self.DCT(block)

            for ch in range(3):
                for x in range(img.shape[0] // N - 1):
                    for y in range(img.shape[1] // N - 1):
                        block = index_block(x, y, ch)
                        img[x*N:(x+1)*N, y*N:(y+1)*N, ch] = self.IDCT(block)

            self.show(img)

    def cosine(self, i, j, N=8):
        """Row-wise function for populating DCT matrix"""
        if i == 0:
            out = np.ones(N) / np.sqrt(N)
        else:
            out = np.sqrt(2/N) * np.cos(((2*j+1) * i * np.pi) / (2 * N))

        return out

    def DCT(self, block, N=8):
        """Performs the DCT operation on an 8x8 block"""
        D = np.matmul(np.matmul(self.T, block - 128.), self.T.T)
        C = np.rint(D / (self.Q * self.compression))
        return C

    def IDCT(self, block, N=8):
        """Performs the inverse DCT operation on an 8x8 block"""
        R = (self.Q * self.compression) * block
        out = np.rint(np.matmul(np.matmul(self.T.T, R), self.T) + 128.)

        return out

    def show(self, img):
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_YCrCb2RGB)
        cv2.imshow('image', img)
        cv2.waitKey(1)

img = Image.open("images/lion.png")
jpeg = JPEG(img)

