import io
import numpy as np
import cv2

from PIL import Image

quantize_mat = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68,109,103, 77],
    [24, 35, 55, 64, 81,104,113, 92],
    [49, 64, 78, 87,103,121,120,101],
    [72, 92, 95, 98,112,100,103, 99]
]) * 8

def rgb_to_ycbcr(img):
    img = np.array(img.convert())

    R = img[:,:,0]/256.
    G = img[:,:,1]/256.
    B = img[:,:,2]/256.

    img[:,:,0] = 16. + 65.738 * R + 129.057 * G + 25.064 * B
    img[:,:,2] = 128. - 37.945 * R - 74.494 * G + 112.439 * B
    img[:,:,1] = 128. + 112.439 * R - 94.154 * G - 18.285 * B

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

    # Access an 8x8 block from the image: 
    # x, y = the indices for each 8x8 block
    # ch = channel from index 0 to 2 for y, cb, cr
    index_block = lambda x, y, ch: img[x*size:(x+1)*size, y*size:(y+1)*size,ch]

    for ch in range(1, 3):
        for x in range(len(img) // size):
            for y in range(len(img) // size):
                try:
                    block = index_block(x, y, ch)
                    img[x*size:(x+1)*size, y*size:(y+1)*size, ch] = DCT(block)
                except:
                    continue

    img = cv2.cvtColor(np.uint8(img), cv2.COLOR_YCrCb2RGB)
    PIL_image = Image.fromarray(img).convert("YCbCr")
    PIL_image.show()



def cosine(i, j, N=8):
    """Row-wise function for populating DCT matrix"""
    if i == 0:
        out = np.ones(N) / np.sqrt(N)
    else:
        out = np.sqrt(2/N) * np.cos(((2*j+1) * i * np.pi) / (2 * N))

    return out


def DCT(block, N=8):
    T = np.array([cosine(x, np.arange(N)) for x in range(8)])

    block -= 128

    D = np.matmul(np.matmul(T, block), T.T)

    C = np.rint(D / quantize_mat)

    R = quantize_mat * C

    out = np.rint(np.matmul(np.matmul(T.T, R), T) + 128)

    return out


img = Image.open("images/lion.png")
img.show()
img = rgb_to_ycbcr(img)
img = downsample(img)
split_blocks(img)
# y, cb, cr = split_blocks(img)
# img = DCT(blocks[0])


# img = cv2.cvtColor(np.uint8(img), cv2.COLOR_YCrCb2RGB)
# PIL_image = Image.fromarray(img).convert("YCbCr")
# PIL_image.show()
