import numpy as np
import cv2, argparse, os, yaml

from src.white_balance import white_balance_gray_world, white_balance_percentile, white_balance_lab
from src.spatial_scale_enhancement import apply_local_contrast_enhancement, apply_spatial_tonemapping


def adjust_brightness_saturation(image, brightness_factor=0, saturation_factor=1):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adjust the brightness and saturation
    hsv_image[:, :, 2] += brightness_factor
    hsv_image[:, :, 1] *= saturation_factor

    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def apply_clahe(image, kernel_size=5, clip_limit=2.0):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply CLAHE to the value channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(kernel_size, kernel_size))
    hsv_image[:, :, 2] = clahe.apply(hsv_image[:, :, 2])

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


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
    required_keys = ['brightness_factor', 'saturation_factor']
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

    if 'white_balance' in config and config['white_balance']['enabled']:
        if config['white_balance']['algorithm'] == 'percentile':
            filename_suffix += f"-wb_percentile{config['white_balance']['percentile']}"
        elif config['white_balance']['algorithm'] == 'grayworld':
            filename_suffix += "-wb_grayworld"
        elif config['white_balance']['algorithm'] == 'lab':
            filename_suffix += "-wb_lab"

    if 'clahe' in config and config['clahe']['enabled']:
        filename_suffix += f"_clahe-kernel{config['clahe']['kernel_size']}-clip{config['clahe']['clip_limit']}"

    if 'denoising' in config and config['denoising']['enabled']:
        filename_suffix += f"_denoise-d{config['denoising']['diameter']}-sc{config['denoising']['sigma_color']}-ss{config['denoising']['sigma_space']}"

    if 'local_contrast_enhancement' in config and config['local_contrast_enhancement']['enabled']:
        filename_suffix += f"_lce-degree{config['local_contrast_enhancement']['degree']}-smoothing{config['local_contrast_enhancement']['smoothing']}"

    if 'spatial_tonemapping' in config and config['spatial_tonemapping']['enabled']:
        filename_suffix += f"_stm-smoothing{config['spatial_tonemapping']['smoothing']}-mid_tone{config['spatial_tonemapping']['mid_tone']}-tonal_width{config['spatial_tonemapping']['tonal_width']}-areas_dark{config['spatial_tonemapping']['areas_dark']}-areas_bright{config['spatial_tonemapping']['areas_bright']}"

    # Load images from input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')): 
            image_path = os.path.join(args.input_folder, filename)
            image = cv2.imread(image_path)

            # Apply local contrast enhancement
            if 'local_contrast_enhancement' in config and config['local_contrast_enhancement']['enabled']:
                image = apply_local_contrast_enhancement(image, 
                                                        degree=config['local_contrast_enhancement']['degree'], 
                                                        smoothing=config['local_contrast_enhancement']['smoothing'])

            # Apply spatial tonemapping
            if 'spatial_tonemapping' in config and config['spatial_tonemapping']['enabled']:
                image = apply_spatial_tonemapping(image, 
                                                  smoothing=config['spatial_tonemapping']['smoothing'], 
                                                  mid_tone=config['spatial_tonemapping']['mid_tone'], 
                                                  tonal_width=config['spatial_tonemapping']['tonal_width'], 
                                                  areas_dark=config['spatial_tonemapping']['areas_dark'], 
                                                  areas_bright=config['spatial_tonemapping']['areas_bright'], 
                                                  preserve_tones=config['spatial_tonemapping']['preserve_tones'])

            # Apply denoising
            if 'denoising' in config and config['denoising']['enabled']:
                image = denoise(image, config['denoising']['diameter'], config['denoising']['sigma_color'], config['denoising']['sigma_space'])

            # Apply CLAHE
            if 'clahe' in config and config['clahe']['enabled']:
                image = apply_clahe(image, config['clahe']['kernel_size'], config['clahe']['clip_limit'])

            # Adjust brightness and saturation
            image = adjust_brightness_saturation(image, config['brightness_factor'], 1)

            # Apply white balancing
            if 'white_balance' in config and config['white_balance']['enabled']:
                wb_algorithm = config['white_balance']['algorithm']
                if wb_algorithm == 'percentile':
                    image = white_balance_percentile(image, config['white_balance']['percentile'])
                elif wb_algorithm == 'grayworld':
                    image = white_balance_gray_world(image)
                elif wb_algorithm == 'lab':
                    image = white_balance_lab(image)

            image = adjust_brightness_saturation(image, 0, config['saturation_factor'])

            # Save output image
            name, ext = os.path.splitext(filename)
            output_filename = name + filename_suffix + ext
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, image)