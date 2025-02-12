import numpy as np
import cv2


def map_value(value, range_in, range_out, invert, non_lin_convex, non_lin_concave):
    '''

    '''
    
    # truncate value to within input range limits
    if value > range_in[1]: value = range_in[1]
    if value < range_in[0]: value = range_in[0]
    
    # map values linearly to [0,1]
    value = (value - range_in[0]) / (range_in[1] - range_in[0])
    
    # invert values
    if invert is True: value = 1 - value
    
    # apply convex non-linearity 
    if non_lin_convex is not None:
        value = (value * non_lin_convex) / (1 + non_lin_convex - value)
         
    # apply concave non-linearity
    if non_lin_concave is not None:
        value = ((1 + non_lin_concave) * value) / (non_lin_concave + value)
    
    # mapping value to the output range in a linear way
    value = value * (range_out[1] - range_out[0]) + range_out[0]
    
    return value


def get_sigmoid_lut(resolution, threshold, non_linearity):
    max_value = resolution - 1  # the maximum attainable value
    thr = threshold * max_value  # threshold in the range [0,resolution-1]
    alpha = non_linearity * max_value  # controls non-linearity degree

    beta = max_value - thr
    if beta == 0:
        beta = 0.001

    lut = np.zeros(resolution, dtype='float')
    for i in range(resolution):
        i_comp = i - thr  # complement of i

        # upper part of the piece-wise sigmoid function
        if i >= thr:
            lut[i] = (((((alpha + beta) * i_comp) / (alpha + i_comp)) * 
                         (1 / (2 * beta))) + 0.5)
        else:  # lower part of the piece-wise sigmoid function
            lut[i] = (alpha * i) / (alpha - i_comp) * (1 / (2 * thr))

    return lut


def get_photometric_mask(illumination, smoothing):
    '''
    source: 

    Intuition about the threshold and non_linearirty values of the LUTs
    threshold: 
        The larger it is, the stronger the blurring, the better the local 
        contrast but also more halo artifacts (less edge preservation).
    non_linearity: 
        The lower it is, the more it preserves the edges, but also has more 
        'bleeding' effects.
    '''

    # internal parameters
    THR_A = smoothing
    THR_B = 0.04  # ~10/255
    NON_LIN = 0.12  # ~30/255
    LUT_RES = 256

    # get sigmoid LUTs
    lut_a = get_sigmoid_lut(
            resolution=LUT_RES, 
            threshold=THR_A, 
            non_linearity=NON_LIN
            )
    lut_a_max = len(lut_a) -1

    lut_b = get_sigmoid_lut(
            resolution=LUT_RES, 
            threshold=THR_B, 
            non_linearity=NON_LIN
            )
    lut_b_max = len(lut_b) -1

    # expand dimensions to 3D for code compatibility (filtering assumes a 3D image)
    image_ph_mask = np.expand_dims(illumination.copy(), axis=2)

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
    image_ph_mask = np.squeeze(image_ph_mask)

    return image_ph_mask


def spatial_tonemapping(image, smoothing=0.2, mid_tone=0.5, tonal_width=0.5, areas_dark=0.5, areas_bright=0.5, preserve_tones=True, eps = 1 / 256):
    '''
    Source: Vassilios Vonikakis, Stefan Winkler, "A center-surround framework for spatial image processing"  in Proc. IS&T Int’l. Symp. on Electronic Imaging: Retinex at 50,  2016,  https://doi.org/10.2352/ISSN.2470-1173.2016.6.RETINEX-020
    '''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    illumination = hsv_image[:, :, 2].astype(np.float32) / 255

    image_ph_mask = get_photometric_mask(illumination, smoothing=smoothing)

    # adjust range and non-linear response of parameters
    mid_tone = map_value(
            value=mid_tone, 
            range_in=(0,1), 
            range_out=(0,1), 
            invert=False, 
            non_lin_convex=None, 
            non_lin_concave=None
            )

    tonal_width = map_value(
            value=tonal_width, 
            range_in=(0,1), 
            range_out=(eps,1), 
            invert=False, 
            non_lin_convex=0.1, 
            non_lin_concave=None
            )

    areas_dark = map_value(
            value=areas_dark, 
            range_in=(0,1), 
            range_out=(0,5), 
            invert=True, 
            non_lin_convex=0.05, 
            non_lin_concave=None
            )

    areas_bright = map_value(
            value=areas_bright, 
            range_in=(0,1), 
            range_out=(0,5), 
            invert=True, 
            non_lin_convex=0.05, 
            non_lin_concave=None
            )

    # lower tones (below mid_tone level)
    image_lower = illumination.copy()   
    image_lower[image_lower >= mid_tone] = 0
    alpha = np.power(image_ph_mask, 2) / tonal_width
    tone_continuation_factor = mid_tone / np.maximum(mid_tone - image_ph_mask, eps)
    alpha = alpha * tone_continuation_factor + areas_dark
    image_lower = image_lower * (alpha + 1) / (alpha + image_lower)

    # upper tones (above mid_tone level)
    image_upper = illumination.copy()
    image_upper[image_upper < mid_tone] = 0
    image_ph_mask_inv = 1 - image_ph_mask
    alpha = np.power(image_ph_mask_inv, 2) / tonal_width
    tone_continuation_factor = mid_tone / np.maximum(1 - mid_tone - image_ph_mask_inv, eps)
    alpha = alpha * tone_continuation_factor + areas_bright 
    image_upper = (image_upper * alpha) / (1 + alpha - image_upper)

    image_tonemapped = image_lower + image_upper
    if preserve_tones is True:
        preservation_degree = np.abs(0.5 - image_ph_mask) / 0.5
        image_tonemapped = preservation_degree * image_tonemapped + illumination * (1 - preservation_degree)

    hsv_image[:, :, 2] = np.clip(255 * image_tonemapped, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def local_contrast_enhancement(image, degree=1.5, smoothing=0.2):
    '''
    Source: V. Vonikakis and I. Andreadis, "Multi-scale image contrast enhancement," 2008 10th International Conference on Control, Automation, Robotics and Vision, Hanoi, Vietnam, 2008, pp. 856-861, doi: 10.1109/ICARCV.2008.4795629.
    '''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    illumination = hsv_image[:, :, 2].astype(np.float32) / 255
    image_ph_mask = get_photometric_mask(illumination, smoothing=smoothing)

    image_details = illumination - image_ph_mask

    # special treatment for dark regions
    detail_amplification_local = 255 * image_ph_mask / 100
    detail_amplification_local[detail_amplification_local > 1] = 1
    detail_amplification_local = 1 + 0.2 * (1 - detail_amplification_local)  # [1, 1.2]

    # apply all detail adjustements and merge them back to the original image
    image_details = (degree * image_details * detail_amplification_local)
    hsv_image[:, :, 2] = np.clip(255 * (image_ph_mask + image_details), 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)