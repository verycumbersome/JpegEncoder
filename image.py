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

    img = np.stack([y, cb, cr], axis=2)

    return img


def split_blocks(img, size=8):
    x_overflow = img.shape[0] % size
    y_overflow = img.shape[1] % size
    img = img[:-x_overflow,:-y_overflow,:]

    y = img[:,:,0]
    print(y[0:8,0:8,])

    y = np.ravel(y)
    y = np.split(y, y.shape[0] // size)
    print(y)

    # print()
    # print()
    # print()
    # print()
    # print()
    # print()
    # print()
    # print()
    # y = y.reshape(8, 8, -1, order="F")
    # print(y.shape)
    # print(y[0:8,0:8,0])
    # cb = img[:,:,1]
    # cr = img[:,:,2]

    # y = [x.reshape(8, 8) for x in np.split(y, y.shape[0] // size**2)]
    # print(y[0])
    # print(y.shape[0])

    # y = np.split(y.ravel(), img.shape[])
    # print(len(y))
    # for i in y:
        # print(i.shape)


# def DCT(block):
    # block -= 128


img = Image.open("images/wood.png")
img = rgb_to_ycbcr(img)
img = downsample(img)
split_blocks(img)
# y, cb, cr = split_blocks(img)
# img = DCT(blocks[0])


# img = cv2.cvtColor(np.uint8(img), cv2.COLOR_YCrCb2RGB)
# PIL_image = Image.fromarray(img).convert("YCbCr")
# PIL_image.show()
