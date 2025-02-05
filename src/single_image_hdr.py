import numpy as np
import cv2


def exposure_ratio(image, window_size=5, sigma=0.5, lambda_factor=0.001, epsilon=1e-6):
    # Source: https://www.mdpi.com/1424-8220/20/16/4378

    # Extract Value Channel (L) from HSV transformation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    L = hsv_image[:, :, 2].astype(np.float64) / 255.0  # Normalize L to [0, 1]

    # Define Gradient Operator (∆D)
    dx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
    delta_DL = np.sqrt(dx**2 + dy**2)  # Gradient magnitude

    # Define Gaussian Kernel (Gσ) - Example: σ = 1/2
    gaussian_kernel = cv2.getGaussianKernel(window_size, sigma)  # Adjust kernel size as needed
    gaussian_delta_dl = cv2.filter2D(delta_DL, -1, gaussian_kernel @ gaussian_kernel.T)

    WD = 1 / (np.abs(gaussian_delta_dl) + epsilon)
    WD = gaussian_kernel / WD

    balancing_coefficient = 0

    H, W = L.shape
    for y in range(H):
        for x in range(W):
            y_min = max(0, y - window_size)
            y_max = min(H - 1, y + window_size)
            x_min = max(0, x - window_size)
            x_max = min(W - 1, x + window_size)

            window_delta_DL = delta_DL[y_min:y_max+1, x_min:x_max+1]
            balancing_coefficient += WD / (np.linalg.norm(window_delta_DL) + epsilon)

    T = np.sum(L - lambda_factor * balancing_coefficient/2)

    return 1 / np.maximum(T, epsilon)


def single_image_hdr(image):
    # Apply logarithmic transformation to the input image to bound the irradiance information to a range from 0 to 1
    max_val = np.max(image)
    log_image = np.log(image + 1) / np.log(max_val + 1)

    # Apply RGB-to-HSV transformation to the log image
    hsv_image = cv2.cvtColor((log_image * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV)

    exposure_ratio()