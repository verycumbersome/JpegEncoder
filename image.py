import io
import numpy as np
import cv2
from PIL import Image


def rgb_to_ycbcr(img):
    img = np.array(img.convert())

    R = img[:,:,0]/256.
    G = img[:,:,1]/256.
    B = img[:,:,2]/256.

    img[:,:,0] = 16 + 65.738 * R + 129.057 * G + 25.064 * B
    img[:,:,2] = 128 - 37.945 * R - 74.494 * G + 112.439 * B
    img[:,:,1] = 128 + 112.439 * R - 94.154 * G - 18.285 * B

    return img


def downsample(img, scale=4):
    cb = img[::scale,::scale,1]
    cr = img[::scale,::scale,2]

    y = img[:,:,0]
    cb = np.kron(cb, np.ones((scale, scale)))[:img.shape[0],:img.shape[1]]
    cr = np.kron(cr, np.ones((scale, scale)))[:img.shape[0],:img.shape[1]]

    img = np.stack([y, cb, cr], axis=2).astype(np.uint8)

    return img


img = Image.open("images/cat.png")
img = rgb_to_ycbcr(img)
img = downsample(img)


img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
PIL_image = Image.fromarray(img).convert("YCbCr")
PIL_image.show()
