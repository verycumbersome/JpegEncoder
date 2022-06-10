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

    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    PIL_image = Image.fromarray(img).convert("YCbCr")
    PIL_image.show()

    # print(img)


img = Image.open("images/wood.png")
rgb_to_ycbcr(img)
