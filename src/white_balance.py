import numpy as np
import cv2


def white_balance_gray_world(image):
    # Source: https://www.researchgate.net/publication/235350557_Combining_Gray-World_assumption_White-Point_correction_and_power_transformation_for_automatic_white_balance

    image_out = image.astype(np.float32)

    mean_r = np.mean(image_out[:, :, 2])
    mean_g = np.mean(image_out[:, :, 1])
    mean_b = np.mean(image_out[:, :, 0])

    # Mean value of all channels
    gain = np.mean((mean_r, mean_g, mean_b)) / 255

    max_r = np.max(image_out[:, :, 2])
    max_g = np.max(image_out[:, :, 1])
    max_b = np.max(image_out[:, :, 0])

    # logarithm base to which each channel will be raised
    base_r = mean_r / max_r
    base_g = mean_g / max_g
    base_b = mean_b / max_b
    
    # the power to which each channel will be raised
    power_r = np.log(gain) / np.log(base_r)
    power_g = np.log(gain) / np.log(base_g)
    power_b = np.log(gain) / np.log(base_b)
    
    # separately applying different color correction powers to each channel
    image_out[:, :, 2] = (image_out[:, :, 2] / max_r) ** power_r
    image_out[:, :, 1] = (image_out[:, :, 1] / max_g) ** power_g
    image_out[:, :, 0] = (image_out[:, :, 0] / max_b) ** power_b

    return (255 * image_out).astype(np.uint8)


def white_balance_percentile(image, percentile=97.5):
    image_out = image.astype(np.float32)

    # Calculate the percentile values for each channel
    gain = 255 / np.percentile(image, percentile, axis=(0, 1))

    # Apply the gain to each color channel
    image_out[:, :, 2] *= gain[2]
    image_out[:, :, 1] *= gain[1]
    image_out[:, :, 0] *= gain[0]

    return np.clip(image_out, 0, 255).astype(np.uint8)


def white_balance_lab(image):
    # Source:https://isprs-archives.copernicus.org/articles/XL-5-W5/25/2015/isprsarchives-XL-5-W5-25-2015.pdf
    # Reference: https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=cvtcolor#cvtcolor

    # Convert the image from BGR to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Reset alpha and beta range from [0, 255] to [-128, 127]
    lab_image[:, :, 1:] -= 128

    # Ensure the values are within the valid range for LAB to BGR conversion
    lab_image[:, :, 1] = np.clip(lab_image[:, :, 1] - np.mean(lab_image[:, :, 1]), -128, 127)  # a* component
    lab_image[:, :, 2] = np.clip(lab_image[:, :, 2] - np.mean(lab_image[:, :, 2]), -128, 127)  # b* component

    # Reset alpha and beta range from [-128, 127] to [0, 255]
    lab_image[:, :, 1:] += 128

    # Convert the modified LAB image back to BGR
    return cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)