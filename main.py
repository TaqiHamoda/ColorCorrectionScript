import numpy as np
import cv2, argparse, os, yaml


def adjust_brightness_saturation(image, brightness_factor=0, saturation_factor=1):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adjust the brightness and saturation
    hsv_image[:, :, 2] += brightness_factor
    hsv_image[:, :, 1] *= saturation_factor

    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def contrast_enhancement(image, kernel_size=5, clip_limit=2.0):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply CLAHE to the value channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kernel_size, kernel_size))
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


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


def denoise(image, diameter=5, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)


# def exposure_ratio(image, window_size=5, sigma=0.5, lambda_factor=0.001, epsilon=1e-6):
#     # Source: https://www.mdpi.com/1424-8220/20/16/4378

#     # Extract Value Channel (L) from HSV transformation
#     hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     L = hsv_image[:, :, 2].astype(np.float64) / 255.0  # Normalize L to [0, 1]

#     # Define Gradient Operator (∆D)
#     dx = cv2.Sobel(L, cv2.CV_64F, 1, 0, ksize=3)
#     dy = cv2.Sobel(L, cv2.CV_64F, 0, 1, ksize=3)
#     delta_DL = np.sqrt(dx**2 + dy**2)  # Gradient magnitude

#     # Define Gaussian Kernel (Gσ) - Example: σ = 1/2
#     gaussian_kernel = cv2.getGaussianKernel(window_size, sigma)  # Adjust kernel size as needed
#     gaussian_delta_dl = cv2.filter2D(delta_DL, -1, gaussian_kernel @ gaussian_kernel.T)

#     WD = 1 / (np.abs(gaussian_delta_dl) + epsilon)
#     WD = gaussian_kernel / WD

#     balancing_coefficient = 0

#     H, W = L.shape
#     for y in range(H):
#         for x in range(W):
#             y_min = max(0, y - window_size)
#             y_max = min(H - 1, y + window_size)
#             x_min = max(0, x - window_size)
#             x_max = min(W - 1, x + window_size)

#             window_delta_DL = delta_DL[y_min:y_max+1, x_min:x_max+1]
#             balancing_coefficient += WD / (np.linalg.norm(window_delta_DL) + epsilon)

#     T = np.sum(L - lambda_factor * balancing_coefficient/2)

#     return 1 / np.maximum(T, epsilon)


# def single_image_hdr(image):
#     # Apply logarithmic transformation to the input image to bound the irradiance information to a range from 0 to 1
#     max_val = np.max(image)
#     log_image = np.log(image + 1) / np.log(max_val + 1)

#     # Apply RGB-to-HSV transformation to the log image
#     hsv_image = cv2.cvtColor((log_image * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV)

#     exposure_ratio()


if __name__ == '__main__':
    # import matplotlib.pyplot as plt

    # image = cv2.imread("images/uxo_enhanced.jpg")

    # kernel = 5

    # results = []
    # for i in range(6):
    #     results.append([])

    #     clip = 1.0
    #     for j in range(5):
    #         results[-1].append(contrast_enhancement(image, kernel_size=kernel, clip_limit=clip))
    #         clip += 1

    #     kernel += 5

    # fig, axes = plt.subplots(nrows=len(results), ncols=len(results[0]), figsize=(15, 10))

    # kernel = 5
    # for i, row in enumerate(results):
    #     clip = 1
    #     for j, image in enumerate(row):
    #         axes[i, j].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #         axes[i, j].set_title(f"Kernel: {kernel}, Clip: {clip}")
    #         axes[i, j].axis('off')

    #         clip += 1

    #     kernel += 5

    # plt.tight_layout()
    # plt.show()

    # exit()


    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image enhancement script')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input folder with images')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()

    # Load configuration from YAML file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Validate config structure (optional, but recommended)
    required_keys = ['white_balance_type']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required key '{key}' in config file.")

    # Create output folder
    output_folder = os.path.join(os.path.dirname(args.input_folder), f"{os.path.basename(args.input_folder)}_output")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Build filename suffix
    filename_suffix = ""

    # Add parameters to filename suffix
    if 'brightness_factor' in config and config['brightness_factor'] != 0:
        filename_suffix += f"_b{config['brightness_factor']}"
    if 'saturation_factor' in config and config['saturation_factor'] != 1.0:
        filename_suffix += f"_s{config['saturation_factor']}"

    if config['white_balance_type'] == 'percentile':
        filename_suffix += f"-percentile{config['white_balance_percentile']}"
    elif config['white_balance_type'] == 'gray_world':
        filename_suffix += "-grayworld"

    if not config['skip_clahe']:
        filename_suffix += f"_clahe-kernel{config['clahe_kernel_size']}-clip{config['clahe_clip_limit']}"

    if not config['skip_denoising']:
        filename_suffix += f"_denoise-d{config['denoise_diameter']}-sc{config['denoise_sigma_color']}-ss{config['denoise_sigma_space']}"

    # Load images from input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')): # More concise way to check multiple extensions
            image_path = os.path.join(args.input_folder, filename)
            image = cv2.imread(image_path)

            # Adjust brightness and saturation
            image = adjust_brightness_saturation(image, config['brightness_factor'], config['saturation_factor'])

            # Apply white balancing
            wb_type = config['white_balance_type']
            if wb_type == 'percentile':
                image = white_balance_percentile(image, config['white_balance_percentile'])
            elif wb_type == 'gray_world':
                image = white_balance_gray_world(image)

            # Apply denoising
            if not config['skip_denoising']:
                image = denoise(image, config['denoise_diameter'], config['denoise_sigma_color'], config['denoise_sigma_space'])

            # Apply CLAHE
            if not config['skip_clahe']:
                image = contrast_enhancement(image, config['clahe_kernel_size'], config['clahe_clip_limit'])

            # Save output image
            name, ext = os.path.splitext(filename)
            output_filename = name + filename_suffix + ext
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, image)
