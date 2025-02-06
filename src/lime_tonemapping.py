import numpy as np
import cv2
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import bicgstab
from scipy.ndimage import gaussian_filter1d, zoom
from scipy.signal import convolve2d


def compute_weights_rog(image, sigma1, sigma2, eps=1e-6):
    # Source: https://caibolun.github.io/papers/RoG.pdf

    dx = np.diff(image, axis=1)
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='constant')
    dy = np.diff(image, axis=0)
    dy = np.pad(dy, ((1, 0), (0, 0)), mode='constant')

    gdx1 = gaussian_filter1d(dx, sigma1, axis=1)
    gdy1 = gaussian_filter1d(dy, sigma1, axis=0)

    gdx2 = gaussian_filter1d(dx, sigma2, axis=1)
    gdy2 = gaussian_filter1d(dy, sigma2, axis=0)

    wx = 1 / (np.abs(gdx1) * np.abs(gdx2) + eps)
    wy = 1 / (np.abs(gdy1) * np.abs(gdy2) + eps)
    wx = gaussian_filter1d(wx, sigma1/2, axis=1)
    wy = gaussian_filter1d(wy, sigma1/2, axis=0)

    # Remove the last column and row to match the input image shape
    wx[:, :-1] = 0
    wy[:-1, :] = 0

    return wx, wy


def compute_weights(image, sigma, sharpness):
    """
    Compute texture weights for the input image.

    Args:
    fin (numpy.ndarray): Input image.
    sigma (float): Standard deviation for the Gaussian filter.
    sharpness (float): Sharpness parameter.

    Returns:
    W_h (numpy.ndarray): Horizontal texture weights.
    W_v (numpy.ndarray): Vertical texture weights.
    """
    # range_val = 5
    # dt0_v = np.diff(image, axis=0)
    # dt0_v = np.vstack((dt0_v, image[1:, :] - image[:-1, :]))
    # dt0_h = np.diff(image, axis=1)
    # dt0_h = np.hstack((dt0_h, image[:, 1:] - image[:, :-1]))

    # mid = np.ceil(range_val / 2)
    # temp = np.power(np.arange(range_val) - mid, 2)
    # fil = np.exp(-temp / (2 * sigma ** 2))

    # gauker_h = np.convolve(dt0_h.flatten(), fil, mode='same').reshape(dt0_h.shape)
    # gauker_v = np.convolve(dt0_v.flatten(), fil, mode='same').reshape(dt0_v.shape)

    # W_h = np.sum(fil) / (np.abs(gauker_h * dt0_h) + sharpness)
    # W_v = np.sum(fil) / (np.abs(gauker_v * dt0_v) + sharpness)

    # return W_h, W_v

    # Compute vertical and horizontal differences
    dt0_v = np.roll(image, -1, axis=0) - image
    dt0_v[-1, :] = image[0, :] - image[-1, :]
    dt0_h = np.roll(image, -1, axis=1) - image
    dt0_h[:, -1] = image[:, 0] - image[:, -1]

    # Create a filter kernel
    kernel_h = np.ones((1, sigma)) / sigma
    kernel_v = np.ones((sigma, 1)) / sigma

    # Apply the filter
    gauker_h = convolve2d(dt0_h, kernel_h, mode='same')
    gauker_v = convolve2d(dt0_v, kernel_v, mode='same')

    # Compute texture weights
    W_h = 1 / (np.abs(gauker_h) * np.abs(dt0_h) + sharpness)
    W_v = 1 / (np.abs(gauker_v) * np.abs(dt0_v) + sharpness)

    return W_h, W_v


def solve_linear_equation(image, wx, wy, lambda_val):
    # Source: https://www.researchgate.net/publication/320728375_A_New_Low-Light_Image_Enhancement_Algorithm_Using_Camera_Response_Model
    # Source: https://ieeexplore.ieee.org/document/7782813

    height, width = image.shape
    k = height * width

    dx = -lambda_val * wx.flatten()
    dy = -lambda_val * wy.flatten()

    tempx = np.concatenate([wx[:, -1:], wx[:, :-1]], axis=1)
    tempy = np.concatenate([wy[-1:, :], wy[:-1, :]], axis=0)
    dxa = -lambda_val * tempx.flatten()
    dya = -lambda_val * tempy.flatten()

    tempx = np.concatenate([wx[:, -1:], np.zeros((height, width - 1))], axis=1)
    tempy = np.concatenate([wy[-1:, :], np.zeros((height - 1, width))], axis=0)
    dxd1 = -lambda_val * tempx.flatten()
    dyd1 = -lambda_val * tempy.flatten()

    dxd2 = -lambda_val * wx.flatten()
    dyd2 = -lambda_val * wy.flatten()

    data = np.array([dxd1, dxd2, dyd1, dyd2, 1 - (dx + dy + dxa + dya)])
    diags = np.array([-k + height, -height, -height + 1, -1, 0])
    A = csc_matrix(spdiags(data, diags, k, k))

    i = [0]
    f = lambda x: (i.append(i[-1] + 1), i.pop(0), print(i[-1], x))

    tin = image.flatten()
    tout, _ = bicgstab(A, tin, atol=1e-5, callback=f)
    image_out = tout.reshape((height, width))

    return image_out


def LECARM(image, camera_model='sigmoid', downsampling=0.5, ratio_max=7.0, lambda_val=0.15, sigma=2, sharpness=0.001, scaling=1):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    illumination = hsv_image[:, :, 2].astype(np.float32) / 255

    # Apply the LIME estimator
    T_downsampled = zoom(illumination, downsampling, order=0)
    wx, wy = compute_weights(T_downsampled, sigma, sharpness)
    T_estimated = solve_linear_equation(T_downsampled, wx, wy, lambda_val)
    T_upsampled = zoom(T_estimated, 1/downsampling, order=0)

    # Compute the scaling factor K
    K = np.minimum(1.0 / T_upsampled, ratio_max)
    K = np.tile(K[:, :, np.newaxis], (1, 1, image.shape[2]))

    # Apply the brightness transfer function
    image_out = image.astype(np.float32) / 255

    if camera_model == 'sigmoid':
        n = 0.90
        sigma = 0.60
        image_out = ((sigma + sigma ** 2) * scaling ** n * image_out) / (scaling ** n * sigma * image_out + (1 + sigma - image_out) * sigma)
    elif camera_model == 'gamma':
        image_out = image_out * np.power(scaling, 0.8)
    elif camera_model == 'betagamma':
        beta = np.exp((1 - np.power(scaling, -0.3293)) * 1.1258)
        gamma = np.power(scaling, -0.3293)
        image_out = np.power(image_out, gamma) * beta
    else:  # preferred
        cf = np.power(image_out, 1 / 0.14)
        ka = np.power(scaling, 4.35)
        image_out = (cf * ka / (cf * (ka - 1) + 1)) ** 0.14

    return np.clip(255 * image_out, 0, 255).astype(np.uint8)


def hdr_tonemapping(image, downsampling=0.5, sigma1=1, sigma2=3, epsilon=1e-6):
    # Source: https://www.mdpi.com/1424-8220/20/16/4378
    # Source: https://www.researchgate.net/publication/320728375_A_New_Low-Light_Image_Enhancement_Algorithm_Using_Camera_Response_Model

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_out = hsv_image[:, :, 2].astype(np.float32) / 255

    lambda_val = 0.001

    T_downsampled = zoom(image_out, downsampling, order=0)
    wx, wy = compute_weights_rog(T_downsampled, sigma1, sigma2, eps=epsilon)
    T_estimated = solve_linear_equation(T_downsampled, wx, wy, lambda_val)
    T_upsampled = zoom(T_estimated, 1/downsampling, order=0)

    # Enhancement
    r_ratio = 1 / (T_upsampled + epsilon)

    p1 = 1 + np.std(image_out)
    p2 = -p1 / 4

    gamma = np.power(r_ratio, p2)
    beta = np.exp(p1 * (1 - gamma))

    image_out = beta * np.power(image_out, gamma)
    
    hsv_image[:, :, 2] = np.clip(255 * image_out, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)