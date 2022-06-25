import io
import cv2
import time
import numpy as np

from PIL import Image

class JPEG():
    def __init__(self, fp, compression=8, N=8):
        self.img = np.array(Image.open(fp).convert())

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
        ])[:N,:N]
        self.compression = compression
        self.rgb_to_ycbcr()
        self.downsample()

    def rgb_to_ycbcr(self):
        R = self.img[:,:,0]/256.
        G = self.img[:,:,1]/256.
        B = self.img[:,:,2]/256.
        self.img[:,:,0] = 16. + 65.738 * R + 129.057 * G + 25.064 * B
        self.img[:,:,1] = 128. - 37.945 * R - 74.494 * G + 112.439 * B
        self.img[:,:,2] = 128. + 112.439 * R - 94.154 * G - 18.285 * B

    def downsample(self, scale=4):
        img = self.img
        cb = img[::scale,::scale,1]
        cr = img[::scale,::scale,2]
        y = img[:,:,0]
        cb = np.kron(cb, np.ones((scale, scale)))[:img.shape[0],:img.shape[1]]
        cr = np.kron(cr, np.ones((scale, scale)))[:img.shape[0],:img.shape[1]]
        self.img = np.stack([y, cb, cr], axis=2)

    def encode(self):
        img = np.copy(self.img)
        for ch in range(3):
            for x in range(self.img.shape[0] // self.N - 1):
                for y in range(self.img.shape[1] // self.N - 1):
                    img[x*self.N:(x+1)*self.N, y*self.N:(y+1)*self.N,ch] = self.DCT(img, x, y, ch)
                    img[x*self.N:(x+1)*self.N, y*self.N:(y+1)*self.N,ch] = self.IDCT(img, x, y, ch)
        self.show(img)

    def cosine(self, i, j):
        """Row-wise function for populating DCT matrix"""
        if i == 0:
            out = np.ones(self.N) / np.sqrt(self.N)
        else:
            out = np.sqrt(2/self.N) * np.cos(((2*j+1) * i * np.pi) / (2 * self.N))

        return out

    def DCT(self, img, x, y, ch):
        """Performs the DCT operation on an 8x8 block
        x, y: the indices for each 8x8 block
        ch: channel from index 0 to 2 for y, cb, cr
        """
        block = img[x*self.N:(x+1)*self.N, y*self.N:(y+1)*self.N,ch]
        D = np.matmul(np.matmul(self.T, block - 128.), self.T.T)
        C = np.rint(D / (self.Q * self.compression))

        return C

    def IDCT(self, img, x, y, ch):
        """Performs the inverse DCT operation on an 8x8 block
        x, y: the indices for each 8x8 block
        ch: channel from index 0 to 2 for y, cb, cr
        """
        block = img[x*self.N:(x+1)*self.N, y*self.N:(y+1)*self.N,ch]
        R = (self.Q * self.compression) * block
        out = np.rint(np.matmul(np.matmul(self.T.T, R), self.T) + 128.)

        return out

    def show(self, img):
        img = cv2.cvtColor(np.uint8(img), cv2.COLOR_YCrCb2RGB)
        cv2.imshow('image', img)
        cv2.waitKey(1)


if __name__=="__main__":
    jpeg = JPEG("images/lion.png", compression=0, N=8)

    while True:
        jpeg.compression += 0.1
        jpeg.encode()

        print(jpeg.compression)

