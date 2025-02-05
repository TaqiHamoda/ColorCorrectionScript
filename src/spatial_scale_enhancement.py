import numpy as np
import cv2


def map_value(value, range_in=(0,1), range_out=(0,1), invert=False, non_lin_convex=None, non_lin_concave=None):
    '''
    ---------------------------------------------------------------------------
         Map a scalar value to an output range in a linear/non-linear way
    ---------------------------------------------------------------------------

    Map scalar values to a particular range, in a linear or non-linear way.
    This can be helpful for adjusting the range and nonlinear response of 
    parameters. 

    For more info on the non-linear functions check:
    Vonikakis, V., Winkler, S. (2016). A center-surround framework for spatial 
    image processing. Proc. IS&T Human Vision & Electronic Imaging.


    INPUTS
    ------
    value: float
        Input value to be mapped.
    range_in: tuple (min,max)
        Range of input value. The min and max values that the input value can 
        attain. 
    range_out: tuple (min,max)
        Range of output value. The min and max values that the mapped input 
        value can attain. 
    invert: Bool
        Invert or not the input value. If invert, then min->max and max->min.
    non_lin_convex: None or float (0,inf)
        If None, no non-linearity is applied. If float, then a convex 
        non-linearity is applied, which lowers the values, while not affecting
        the min and max. non_lin_convex controls the steepness of the 
        non-linear mapping. Small values near zero, result in a steeper curve.
    non_lin_concave: None or float (0,inf)
        If None, no non-linearity is applied. If float, then a concave 
        non-linearity is applied, which increases the values, while not 
        affecting min and max. non_lin_concave controls the steepness of the 
        non-linear mapping. Small values near zero, result in a steeper curve.

    OUTPUT
    ------
    Mapped value 
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


def get_photometric_mask(image, smoothing):
    '''
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
    image_ph_mask = np.expand_dims(image.copy(), axis=2)

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


def apply_spatial_tonemapping(image, image_ph_mask, mid_tone=0.5, tonal_width=0.5, areas_dark=0.5, areas_bright=0.5, preserve_tones=True):
    '''
    ---------------------------------------------------------------------------
       Apply spatially variable tone mapping based on the local neighborhood
    ---------------------------------------------------------------------------
    
    Applies different tone mapping curves in each pixel based on its surround.
    For surround, the photometric mask is used. Alternatively, other filters
    could be used, like gaussian, bilateral filter, edge-avoiding wavelets etc.
    Dark pixels are brightened, bright pixels are darkened, and pixels in the 
    mid_tonedle of the tone range are minimally affected. More information 
    about the technique can be found in the following papers:
    
    Related publications: 
    Vonikakis, V., Andreadis, I., & Gasteratos, A. (2008). Fast centre-surround 
    contrast modification. IET Image processing 2(1), 19-34.
    Vonikakis, V., Winkler, S. (2016). A center-surround framework for spatial 
    image processing. Proc. IS&T Human Vision & Electronic Imaging.
    
    
    INPUTS
    ------
    image: numpy array of WxH of float [0,1]
        Input grayscale image with values in the interval [0,1].
    image_ph_mask: numpy array of WxH of float [0,1]
        Grayscale image whose values represent the neighborhood of the pixels 
        of the input image. Usually, this image some type of edge aware 
        filtering, such as bilateral filtering, robust recursive envelopes etc.
    mid_tone: float [0,1]
        The mid point between the 'dark' and 'bright' tones. This is equivalent
        to a pixel value [0,255], but in the interval [0,1].
    tonal_width: float [0,1]
        The range of pixel values that will be affected by the correction. 
        Lower values will localize the enhancement only in a narrow range of 
        pixel values, whereas for higher values the enhancement will extend to 
        a greater range of pixel values. 
    areas_dark: float [0,1]
        Degree of enhencement in the dark image areas (0 = no enhencement)
    areas_bright: float [0,1]
        Degree of enhencement in the bright image areas (0 = no enhencement)
    preserve_tones: boolean
        Whether or not to preserve well-exposed tones around the middle of the 
        range. 
    verbose: boolean
        Display outputs.
    
    OUTPUT
    ------
    image_tonemapped: numpy array of WxH of float [0,1]
        Tonemapped grayscale image. 
        
    '''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_out = hsv_image[:, :, 2].astype(np.float32) / 255



    # defining parameters
    EPSILON = 1 / 256

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
            range_out=(EPSILON,1), 
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
    image_lower = image.copy()   
    image_lower[image_lower>=mid_tone] = 0
    alpha = (image_ph_mask ** 2) / tonal_width
    tone_continuation_factor = mid_tone / (mid_tone + EPSILON - image_ph_mask)
    alpha = alpha * tone_continuation_factor + areas_dark
    image_lower = (image_lower * (alpha + 1)) / (alpha + image_lower)

    # upper tones (above mid_tone level)
    image_upper = image.copy()
    image_upper[image_upper<mid_tone] = 0
    image_ph_mask_inv = 1 - image_ph_mask
    alpha = (image_ph_mask_inv ** 2) / tonal_width
    tone_continuation_factor = mid_tone / ((1 - mid_tone) - image_ph_mask_inv)
    alpha = alpha * tone_continuation_factor + areas_bright 
    image_upper = (image_upper * alpha) / (alpha + 1 - image_upper)

    image_tonemapped = image_lower + image_upper

    if preserve_tones is True:
        preservation_degree = np.abs(0.5 - image_ph_mask) / 0.5  # 0: near 0.5
#        preservation_degree = ((1 + 0.3) * preservation_degree) / (0.3 + preservation_degree)
        image_tonemapped = (preservation_degree * image_tonemapped + 
                           (1-preservation_degree) * image)

    return image_tonemapped


def apply_local_contrast_enhancement(image, degree=1.5, smoothing=0.2):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_out = hsv_image[:, :, 2].astype(np.float32) / 255

    image_ph_mask = get_photometric_mask(image_out, smoothing=smoothing)

    DARK_BOOST = 0.2
    THRESHOLD_DARK_TONES = 100 / 255
    detail_amplification_global = degree

    image_details = image_out - image_ph_mask  # image details

    # special treatment for dark regions
    detail_amplification_local = image_ph_mask / THRESHOLD_DARK_TONES
    detail_amplification_local[detail_amplification_local>1] = 1
    detail_amplification_local = ((1 - detail_amplification_local) * 
                                  DARK_BOOST) + 1  # [1, 1.2]

    # apply all detail adjustements
    image_details = (image_details * 
                     detail_amplification_global * 
                     detail_amplification_local)

    # add details back to the local neighborhood and stay within range
    hsv_image[:, :, 2] = np.clip(255 * (image_ph_mask + image_details), 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)