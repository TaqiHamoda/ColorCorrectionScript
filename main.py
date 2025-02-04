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


def get_sigmoid_lut(resolution, threshold, non_linearirty):
    max_value = resolution - 1  # the maximum attainable value
    thr = threshold * max_value  # threshold in the range [0,resolution-1]
    alpha = non_linearirty * max_value  # controls non-linearity degree

    beta = max_value - thr
    if beta == 0:
        beta = 0.001
    
    lut = np.zeros(resolution, dtype=np.float32)
    for i in range(resolution):
        i_comp = i - thr  # complement of i

        # upper part of the piece-wise sigmoid function
        if i >= thr:
            lut[i] = (((((alpha + beta) * i_comp) / (alpha + i_comp)) * 
                         (1 / (2 * beta))) + 0.5)
        else:  # lower part of the piece-wise sigmoid function
            lut[i] = (alpha * i) / (alpha - i_comp) * (1 / (2 * thr))

    return lut


def get_photometric_mask(image_intensity, smoothing, resolution, threshold, non_linearirty):
    # internal parameters
    THR_A = smoothing
    THR_B = threshold 
    NON_LIN = non_linearirty
    LUT_RES = resolution
    
    # get sigmoid LUTs
    lut_a = get_sigmoid_lut(
            resolution=LUT_RES, 
            threshold=THR_A, 
            non_linearirty=NON_LIN, 
            )
    lut_a_max = len(lut_a) - 1
    lut_b = get_sigmoid_lut(
            resolution=LUT_RES, 
            threshold=THR_B, 
            non_linearirty=NON_LIN, 
            )
    lut_b_max = len(lut_b) - 1

    image_ph_mask = np.expand_dims(image_intensity, axis=2)
    
    # robust recursive envelope

    # up -> down
    for i in range(1, image_ph_mask.shape[0]-1):
        d = np.abs(image_ph_mask[i-1,:,:] - image_ph_mask[i+1,:,:])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        image_ph_mask[i,:,:] = ((image_ph_mask[i,:,:] * d) + 
                              (image_ph_mask[i-1,:,:] * (1-d)))

    # left -> right
    for j in range(1, image_ph_mask.shape[1]-1):
        d = np.abs(image_ph_mask[:,j-1,:] - image_ph_mask[:,j+1,:])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        image_ph_mask[:,j,:] = ((image_ph_mask[:,j,:] * d) + 
                              (image_ph_mask[:,j-1,:] * (1-d)))
        
    # down -> up
    for i in range(image_ph_mask.shape[0]-2, 1, -1):
        d = np.abs(image_ph_mask[i-1,:,:] - image_ph_mask[i+1,:,:])  # diff
        d = lut_a[(d * lut_a_max).astype(int)]
        image_ph_mask[i,:,:] = ((image_ph_mask[i,:,:] * d) + 
                              (image_ph_mask[i+1,:,:] * (1-d)))
        
    # right -> left
    for j in range(image_ph_mask.shape[1]-2, 1, -1):
        d = np.abs(image_ph_mask[:,j-1,:] - image_ph_mask[:,j+1,:])  # diff
        d = lut_b[(d * lut_b_max).astype(int)]
        image_ph_mask[:,j,:] = ((image_ph_mask[:,j,:] * d) + 
                              (image_ph_mask[:,j+1,:] * (1-d)))
          
    # up -> down
    for i in range(1, image_ph_mask.shape[0]-1):
        d = np.abs(image_ph_mask[i-1,:,:] - image_ph_mask[i+1,:,:])  # diff
        d = lut_b[(d * lut_b_max).astype(int)]
        image_ph_mask[i,:,:] = ((image_ph_mask[i,:,:] * d) + 
                              (image_ph_mask[i-1,:,:] * (1-d)))

    # convert back to 2D if grayscale is needed
    return np.squeeze(image_ph_mask)


def apply_local_contrast_enhancement(image, degree=1.5, smoothing=0.2, resolution=256, threshold=0.04, non_linearirty=0.12):
    # https://www.researchgate.net/publication/221145067_Multi-Scale_Image_Contrast_Enhancement

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    DARK_BOOST = 0.2
    THRESHOLD_DARK_TONES = 100 / 255
    detail_amplification_global = degree

    image_intensity = hsv_image[:, :, 2] / 255

    image_ph_mask = get_photometric_mask(
        image_intensity,
        smoothing=smoothing,
        resolution=resolution,
        threshold=threshold,
        non_linearirty=non_linearirty
    )

    image_details = image_intensity - image_ph_mask  # image details

    # special treatment for dark regions
    detail_amplification_local = image_ph_mask / THRESHOLD_DARK_TONES
    detail_amplification_local[detail_amplification_local > 1] = 1
    detail_amplification_local = ((1 - detail_amplification_local) * DARK_BOOST) + 1

    # apply all detail adjustements
    image_details = (
        image_details * 
        detail_amplification_global * 
        detail_amplification_local
    )

    # add details back to the local neighborhood
    hsv_image[:, :, 2] = np.clip(255 * (image_details + image_ph_mask), 0, 255).astype(np.uint8)

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


def denoise(image, diameter=5, sigmaColor=50, sigmaSpace=50):
    return cv2.bilateralFilter(image, diameter, sigmaColor, sigmaSpace)


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


# def single_image_hdr(image):
#     # Apply logarithmic transformation to the input image to bound the irradiance information to a range from 0 to 1
#     max_val = np.max(image)
#     log_image = np.log(image + 1) / np.log(max_val + 1)

#     # Apply RGB-to-HSV transformation to the log image
#     hsv_image = cv2.cvtColor((log_image * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV)

#     exposure_ratio()


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
        filename_suffix += f"_lce-degree{config['local_contrast_enhancement']['degree']}-smoothing{config['local_contrast_enhancement']['smoothing']}-resolution{config['local_contrast_enhancement']['resolution']}-threshold{config['local_contrast_enhancement']['threshold']}-non_linearity{config['local_contrast_enhancement']['non_linearity']}"

    # Load images from input folder
    for filename in os.listdir(args.input_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')): 
            image_path = os.path.join(args.input_folder, filename)
            image = cv2.imread(image_path)

            # Adjust brightness and saturation
            image = adjust_brightness_saturation(image, config['brightness_factor'], config['saturation_factor'])

            # Apply white balancing
            if 'white_balance' in config and config['white_balance']['enabled']:
                wb_algorithm = config['white_balance']['algorithm']
                if wb_algorithm == 'percentile':
                    image = white_balance_percentile(image, config['white_balance']['percentile'])
                elif wb_algorithm == 'grayworld':
                    image = white_balance_gray_world(image)
                elif wb_algorithm == 'lab':
                    image = white_balance_lab(image)

            # Apply local contrast enhancement
            if 'local_contrast_enhancement' in config and config['local_contrast_enhancement']['enabled']:
                image = apply_local_contrast_enhancement(image, 
                                                        degree=config['local_contrast_enhancement']['degree'], 
                                                        smoothing=config['local_contrast_enhancement']['smoothing'], 
                                                        resolution=config['local_contrast_enhancement']['resolution'], 
                                                        threshold=config['local_contrast_enhancement']['threshold'], 
                                                        non_linearity=config['local_contrast_enhancement']['non_linearity'])

            # Apply denoising
            if 'denoising' in config and config['denoising']['enabled']:
                image = denoise(image, config['denoising']['diameter'], config['denoising']['sigma_color'], config['denoising']['sigma_space'])

            # Apply CLAHE
            if 'clahe' in config and config['clahe']['enabled']:
                image = contrast_enhancement(image, config['clahe']['kernel_size'], config['clahe']['clip_limit'])

            # Save output image
            name, ext = os.path.splitext(filename)
            output_filename = name + filename_suffix + ext
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, image)