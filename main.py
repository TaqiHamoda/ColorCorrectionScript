import numpy as np
import cv2, argparse, os
from distutils.util import strtobool


def adjust_brightness_saturation(image, brightness_factor=0, saturation_factor=1):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Adjust the brightness and saturation
    hsv_image[:, :, 2] += brightness_factor
    hsv_image[:, :, 1] *= saturation_factor

    hsv_image = np.clip(hsv_image, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def local_contrast_enhancement(image, kernel_size=5, clip_limit=2.0):
    # Convert the image to LAB color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply CLAHE to the L channel
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
    parser.add_argument('--brightness_factor', type=int, default=0, help='Brightness adjustment factor')
    parser.add_argument('--saturation_factor', type=float, default=1.0, help='Saturation adjustment factor')
    parser.add_argument('--skip_clahe', type=lambda x: bool(strtobool(x)), default=False, help='Skip the CLAHE processing step')
    parser.add_argument('--clahe_kernel_size', type=int, default=5, help='CLAHE kernel size')
    parser.add_argument('--clahe_clip_limit', type=float, default=2.0, help='CLAHE clip limit')
    parser.add_argument('--white_balance_type', type=str, required=True, choices=['patch', 'percentile', 'gray_world'], help='Type of white balancing')
    parser.add_argument('--white_balance_patch_column', type=int, default=0, help='Column of the white patch')
    parser.add_argument('--white_balance_patch_row', type=int, default=0, help='Row of the white patch')
    parser.add_argument('--white_balance_patch_width', type=int, default=10, help='Width of the white patch')
    parser.add_argument('--white_balance_patch_height', type=int, default=10, help='Height of the white patch')
    parser.add_argument('--white_balance_percentile', type=float, default=97.5, help='Percentile value for white balancing')
    parser.add_argument('--skip_denoising', type=lambda x: bool(strtobool(x)), default=False, help='Skip the denoising step')
    parser.add_argument('--denoise_diameter', type=int, default=5, help='Diameter for denoising')
    parser.add_argument('--denoise_sigma_color', type=float, default=50.0, help='Sigma color for denoising')
    parser.add_argument('--denoise_sigma_space', type=float, default=50.0, help='Sigma space for denoising')
    args = parser.parse_args()

    # Create output folder
    output_folder = os.path.join(os.path.dirname(args.input_folder), 'output')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load images from input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(args.input_folder, filename)
            image = cv2.imread(image_path)

            # Adjust brightness and saturation
            image = adjust_brightness_saturation(image, args.brightness_factor, args.saturation_factor)

            # Apply white balancing
            if args.white_balance_type == 'patch':
                image = white_balance_patch(image, args.white_balance_patch_column, args.white_balance_patch_row, args.white_balance_patch_width, args.white_balance_patch_height)
            elif args.white_balance_type == 'percentile':
                image = white_balance_percentile(image, args.white_balance_percentile)
            elif args.white_balance_type == 'gray_world':
                image = white_balance_gray_world(image)

            # Apply CLAHE
            if not args.skip_clahe:
                image = local_contrast_enhancement(image, args.clahe_kernel_size, args.clahe_clip_limit)

            # Apply denoising
            if not args.skip_denoising:
                image = denoise(image, diameter=args.denoise_diameter, sigmaColor=args.denoise_sigma_color, sigmaSpace=args.denoise_sigma_space)

            # Save output image
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)
