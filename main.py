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


def white_balance(image, gain):
    image_out = image.astype(np.float32)

    # Apply the gain to each color channel
    image_out[:, :, 2] *= gain[2]
    image_out[:, :, 1] *= gain[1]
    image_out[:, :, 0] *= gain[0]

    return np.clip(image_out, 0, 255).astype(np.uint8)


def white_balance_patch(image, patch_column, patch_row, patch_width, patch_height):
    # Extract the "true" white patch from the image
    patch = image[patch_row: patch_row + patch_height, patch_column: patch_column + patch_width]
    mean_color = np.mean(patch, axis=(0, 1))

    return white_balance(image, 255 / mean_color)


def white_balance_percentile(image, percentile=97.5):
    # Calculate the percentile values for each channel
    percentile_values = np.percentile(image, percentile, axis=(0, 1))

    return white_balance(image, 255 / percentile_values)


def white_balance_gray_world(image):
    gain = np.mean(image) / np.mean(image, axis=(0, 1))

    return white_balance(image, gain)


def denoise(image, diameter=5, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)


if __name__ == '__main__':
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
        if filename.endswith(('.jpg', '.png')): # More concise way to check multiple extensions
            image_path = os.path.join(args.input_folder, filename)
            image = cv2.imread(image_path)

            # Adjust brightness and saturation
            image = adjust_brightness_saturation(image, config['brightness_factor'], config['saturation_factor'])

            # Apply white balancing
            wb_type = config['white_balance_type']
            if wb_type == 'patch':
                image = white_balance_patch(image, config['white_balance_patch_column'], config['white_balance_patch_row'], config['white_balance_patch_width'], config['white_balance_patch_height'])
            elif wb_type == 'percentile':
                image = white_balance_percentile(image, config['white_balance_percentile'])
            elif wb_type == 'gray_world':
                image = white_balance_gray_world(image)

            # Apply CLAHE
            if not config['skip_clahe']:
                image = contrast_enhancement(image, config['clahe_kernel_size'], config['clahe_clip_limit'])

            # Apply denoising
            if not config['skip_denoising']:
                image = denoise(image, config['denoise_diameter'], config['denoise_sigma_color'], config['denoise_sigma_space'])

            # Save output image
            name, ext = os.path.splitext(filename)
            output_filename = name + filename_suffix + ext
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, image)
