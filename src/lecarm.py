import numpy as np
from scipy.interpolate import interp1d
from scipy import sparse
from scipy.sparse.linalg import spsolve, LinearOperator
from scipy.sparse.linalg import cg
from scipy.ndimage import zoom


class CameraModel:
    def __init__(self, param=None):
        """
        Initialize the CameraModel class.

        Args:
            param (any, optional): Parameter to initialize the camera model. Defaults to None.
        """
        if param is not None:
            self.param = param
        self.name = self.__class__.__name__

    def set_param(self, param):
        """
        Set the parameter of the camera model.

        Args:
            param (any): Parameter to set.
        """
        self.param = param

    def btf(self, B0, k):
        """
        Apply the brightness transfer function.

        Args:
            B0 (numpy.ndarray): Input brightness values.
            k (float): Scaling factor.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        return self.crf(k * self.crf_inv(B0))

    def crf_inv(self, B):
        """
        Apply the inverse camera response function.

        Args:
            B (numpy.ndarray): Input brightness values.

        Returns:
            numpy.ndarray: Output exposure values.
        """
        idx = np.linspace(0, 1, 1024)
        e = idx
        b = self.crf(e)

        # Use interp1d for 1D interpolation
        f = interp1d(b, e, kind='cubic', fill_value="extrapolate")
        E = f(B.flatten())
        E = E.reshape(B.shape)

        return E

    # Abstract method crf should be implemented in child classes
    def crf(self, E):
        """
        Apply the camera response function.

        Args:
            E (numpy.ndarray): Input exposure values.

        Raises:
            NotImplementedError: This method should be implemented in child classes.
        """
        raise NotImplementedError("Subclass must implement abstract method")


class Sigmoid(CameraModel):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Sigmoid class.

        Args:
            *args: Variable number of arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.param = np.array([0.90, 0.60])  # n, sigma

    def crf(self, E):
        """
        Apply the sigmoid camera response function.

        Args:
            E (numpy.ndarray): Input exposure values.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        n = self.param[0]
        sigma = self.param[1]
        B = (1 + sigma) * (E ** n) / (E ** n + sigma)
        return B

    def btf(self, B0, k):
        """
        Apply the brightness transfer function for the sigmoid model.

        Args:
            B0 (numpy.ndarray): Input brightness values.
            k (float): Scaling factor.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        n = self.param[0]
        sigma = self.param[1]
        B1 = ((sigma + sigma ** 2) * k ** n * B0) / (k ** n * sigma * B0 + (1 + sigma - B0) * sigma)
        return B1

    def crf_inv(self, B):
        """
        Apply the inverse sigmoid camera response function.

        Args:
            B (numpy.ndarray): Input brightness values.

        Returns:
            numpy.ndarray: Output exposure values.
        """
        n = self.param[0]
        sigma = self.param[1]
        E = sigma * B / (1 + sigma - B)
        E = E ** (1 / n)
        return E


class Preferred(CameraModel):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Preferred class.

        Args:
            *args: Variable number of arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.param = np.array([4.35, 1.29, 0.14])  # parameters for the preferred model

    def crf(self, E):
        """
        Apply the preferred camera response function.

        Args:
            E (numpy.ndarray): Input exposure values.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        B = np.exp(self.param[1]) * np.power(E, self.param[0])
        B = B / (B + 1)
        B = np.power(B, self.param[2])
        return B

    def btf(self, B0, k):
        """
        Apply the brightness transfer function for the preferred model.

        Args:
            B0 (numpy.ndarray): Input brightness values.
            k (float): Scaling factor.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        cf = np.power(B0, 1 / self.param[2])
        ka = np.power(k, self.param[0])
        B1 = (cf * ka / (cf * (ka - 1) + 1)) ** self.param[2]
        return B1

    def crf_inv(self, B):
        """
        Apply the inverse preferred camera response function.

        Args:
            B (numpy.ndarray): Input brightness values.

        Returns:
            numpy.ndarray: Output exposure values.
        """
        cf = np.power(B, 1 / self.param[2])
        E = (cf / (1 - cf)) / np.exp(self.param[1])
        E = E ** (1 / self.param[0])
        return E


class Gamma(CameraModel):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Gamma class.

        Args:
            *args: Variable number of arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.param = 0.8  # gamma parameter

    def crf(self, E):
        """
        Apply the gamma camera response function.

        Args:
            E (numpy.ndarray): Input exposure values.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        B = np.exp(np.power(E, self.param))
        return B

    def btf(self, B0, k):
        """
        Apply the brightness transfer function for the gamma model.

        Args:
            B0 (numpy.ndarray): Input brightness values.
            k (float): Scaling factor.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        B1 = B0 * np.power(k, self.param)
        return B1

    def crf_inv(self, B):
        """
        Apply the inverse gamma camera response function.

        Args:
            B (numpy.ndarray): Input brightness values.

        Returns:
            numpy.ndarray: Output exposure values.
        """
        E = np.power(np.log(B), 1 / self.param)
        return E


class BetaGamma(CameraModel):
    def __init__(self, *args, **kwargs):
        """
        Initialize the BetaGamma class.

        Args:
            *args: Variable number of arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.param = np.array([-0.3293, 1.1258])  # dorf parameters

    def crf(self, E):
        """
        Apply the beta-gamma camera response function.

        Args:
            E (numpy.ndarray): Input exposure values.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        B = np.exp((1 - np.power(E, self.param[0])) * self.param[1])
        return B

    def btf(self, B0, k):
        """
        Apply the brightness transfer function for the beta-gamma model.

        Args:
            B0 (numpy.ndarray): Input brightness values.
            k (float): Scaling factor.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        beta = self.crf(k)
        gamma = np.power(k, self.param[0])
        B1 = np.power(B0, gamma) * beta
        return B1

    def crf_inv(self, B):
        """
        Apply the inverse beta-gamma camera response function.

        Args:
            B (numpy.ndarray): Input brightness values.

        Returns:
            numpy.ndarray: Output exposure values.
        """
        E = np.power((1 - np.log(B) / self.param[1]), 1 / self.param[0])
        return E


class Beta(CameraModel):
    def __init__(self, *args, **kwargs):
        """
        Initialize the Beta class.

        Args:
            *args: Variable number of arguments.
            **kwargs: Keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self.param = 0.8  # dorf: 0.4800

    def crf(self, E):
        """
        Apply the beta camera response function.

        Args:
            E (numpy.ndarray): Input exposure values.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        B = np.power(E, self.param)
        return B

    def btf(self, B0, k):
        """
        Apply the brightness transfer function for the beta model.

        Args:
            B0 (numpy.ndarray): Input brightness values.
            k (float): Scaling factor.

        Returns:
            numpy.ndarray: Output brightness values.
        """
        B1 = B0 * np.power(k, self.param)
        return B1

    def crf_inv(self, B):
        """
        Apply the inverse beta camera response function.

        Args:
            B (numpy.ndarray): Input brightness values.

        Returns:
            numpy.ndarray: Output exposure values.
        """
        E = np.power(B, 1 / self.param)
        return E

def limeEstimate(I, lambda_val=0.15, sigma=2, sharpness=0.001):
    """
    Estimate illumination T using the LIME method.

    Args:
        I (numpy.ndarray): Input image.
        lambda_val (float, optional): Lambda value. Defaults to 0.15.
        sigma (float, optional): Sigma value. Defaults to 2.
        sharpness (float, optional): Sharpness value. Defaults to 0.001.

    Returns:
        numpy.ndarray: Estimated illumination map.
    """
    I = I.astype(np.float32) / 255.0
    wx, wy = computeTextureWeights(I, sigma, sharpness)
    S = solveLinearEquation(I, wx, wy, lambda_val)
    return S

def computeTextureWeights(image, sigma, sharpness):
    """
    Compute texture weights.

    Args:
        fin (numpy.ndarray): Input image.
        sigma (float): Sigma value.
        sharpness (float): Sharpness value.

    Returns:
        tuple: Horizontal and vertical texture weights.
    """
    range_val = 5
    dt0_v = np.diff(image, axis=0)
    dt0_v = np.vstack((dt0_v, image[0, :] - image[-1, :]))
    dt0_h = np.diff(image, axis=1)
    dt0_h = np.hstack((dt0_h, image[:, 0] - image[:, -1]))

    mid = np.ceil(range_val / 2)
    temp = np.power(np.arange(range_val) - mid, 2)
    fil = np.exp(-temp / (2 * sigma ** 2))

    gauker_h = np.convolve(dt0_h.flatten(), fil, mode='same').reshape(dt0_h.shape)
    gauker_v = np.convolve(dt0_v.flatten(), fil, mode='same').reshape(dt0_v.shape)

    W_h = np.sum(fil) / (np.abs(gauker_h * dt0_h) + sharpness)
    W_v = np.sum(fil) / (np.abs(gauker_v * dt0_v) + sharpness)

    return W_h, W_v

def solveLinearEquation(IN, wx, wy, lambda_val):
    """
    Solve the linear equation.

    Args:
        IN (numpy.ndarray): Input image.
        wx (float): Horizontal weight.
        wy (float): Vertical weight.
        lambda_val (float): Lambda value.

    Returns:
        numpy.ndarray: Solution to the linear equation.
    """
    r, c, ch = IN.shape
    k = r * c
    dx = -lambda_val * wx
    dy = -lambda_val * wy

    tempx = np.hstack((wx[-1], wx[:-1]))
    tempy = np.vstack((wy[-1, :], wy[:-1, :]))
    dxa = -lambda_val * tempx
    dya = -lambda_val * tempy

    tempx = np.hstack((wx[-1], np.zeros(c - 1)))
    tempy = np.vstack((wy[-1, :], np.zeros((r - 1, c))))
    dxd1 = -lambda_val * tempx
    dyd1 = -lambda_val * tempy

    wx[-1] = 0
    wy[-1, :] = 0
    dxd2 = -lambda_val * wx
    dyd2 = -lambda_val * wy

    data = np.array([dxd1, dxd2, dyd1, dyd2, 1 - (dx + dy + dxa + dya)])
    diags = np.array([-k + r, -r, -r + 1, -1, 0])
    A = sparse.spdiags(data, diags, k, k)

    OUT = IN.copy()
    for ii in range(ch):
        tin = IN[:, :, ii].flatten()
        tout = spsolve(A, tin)
        OUT[:, :, ii] = tout.reshape((r, c))

    return OUT


def LECARM(in_image, camera_model=None):
    """
    Apply the Local Exposure Correction and Adaptive Rendering Model (LECARM) to an input image.

    Args:
        in_image (numpy.ndarray): Input image.
        camera_model (CameraModel, optional): Camera model to use. Defaults to Sigmoid.

    Returns:
        numpy.ndarray: Output image.
    """
    if camera_model is None:
        camera_model = Sigmoid()

    # Ensure input image is a float array
    if not in_image.dtype.kind == 'f':
        in_image = in_image.astype(np.float32) / 255.0

    # Define parameters
    ratio_max = 7.0

    # Define the LIME estimator function
    estimater = lambda t: limeEstimate(t, 0.15, 2.0)

    # Compute the maximum value across channels
    T = np.max(in_image, axis=2)

    # Apply the LIME estimator
    T_downsampled = zoom(T, 0.5, order=0)
    T_estimated = estimater(T_downsampled)
    T_upsampled = zoom(T_estimated, 2.0, order=0)

    # Compute the scaling factor K
    K = np.minimum(1.0 / T_upsampled, ratio_max)
    K = np.tile(K[:, :, np.newaxis], (1, 1, in_image.shape[2]))

    # Apply the brightness transfer function
    out_image = camera_model.btf(in_image, K)

    return out_image