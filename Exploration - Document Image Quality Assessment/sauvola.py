
import itertools
import math
import numpy as np
from scipy import ndimage as ndi
from collections import OrderedDict
from skimage.exposure import histogram
from skimage._shared.utils import check_nD, warn, deprecated
from skimage.transform import integral_image
from skimage.util import crop, dtype_limits

def threshold_sauvola(image, window_size=15, k=0.2, r=None):
    """Applies Sauvola local threshold to an array. Sauvola is a
    modification of Niblack technique.
    In the original method a threshold T is calculated for every pixel
    in the image using the following formula::
        T = m(x,y) * (1 + k * ((s(x,y) / R) - 1))
    where m(x,y) and s(x,y) are the mean and standard deviation of
    pixel (x,y) neighborhood defined by a rectangular window with size w
    times w centered around the pixel. k is a configurable parameter
    that weights the effect of standard deviation.
    R is the maximum standard deviation of a greyscale image.
    Parameters
    ----------
    image: (N, M) ndarray
        Grayscale input image.
    window_size : int, optional
        Odd size of pixel neighborhood window (e.g. 3, 5, 7...).
    k : float, optional
        Value of the positive parameter k.
    r : float, optional
        Value of R, the dynamic range of standard deviation.
        If None, set to the half of the image dtype range.
    Returns
    -------
    threshold : (N, M) ndarray
        Threshold mask. All pixels with an intensity higher than
        this value are assumed to be foreground.
    Notes
    -----
    This algorithm is originally designed for text recognition.
    References
    ----------
    .. [1] J. Sauvola and M. Pietikainen, "Adaptive document image
           binarization," Pattern Recognition 33(2),
           pp. 225-236, 2000.
           :DOI:`10.1016/S0031-3203(99)00055-2`
    Examples
    --------
    >>> from skimage import data
    >>> image = data.page()
    >>> t_sauvola = threshold_sauvola(image, window_size=15, k=0.2)
    >>> binary_image = image > t_sauvola
    """
    if r is None:
        imin, imax = dtype_limits(image, clip_negative=False)
        r = 0.5 * (imax - imin)
    m, s = _mean_std(image, window_size)
    return m * (1 + k * ((s / r) - 1))

def _mean_std(image, w):
    """Return local mean and standard deviation of each pixel using a
    neighborhood defined by a rectangular window with size w times w.
    The algorithm uses integral images to speedup computation. This is
    used by threshold_niblack and threshold_sauvola.
    Parameters
    ----------
    image : ndarray
        Input image.
    w : int
        Odd window size (e.g. 3, 5, 7, ..., 21, ...).
    Returns
    -------
    m : 2-D array of same size of image with local mean values.
    s : 2-D array of same size of image with local standard
        deviation values.
    References
    ----------
    .. [1] F. Shafait, D. Keysers, and T. M. Breuel, "Efficient
           implementation of local adaptive thresholding techniques
           using integral images." in Document Recognition and
           Retrieval XV, (San Jose, USA), Jan. 2008.
           :DOI:`10.1117/12.767755`
    """
    if w == 1 or w % 2 == 0:
        raise ValueError(
            "Window size w = %s must be odd and greater than 1." % w)

    left_pad = w // 2 + 1
    right_pad = w // 2
    padded = np.pad(image.astype('float'), (left_pad, right_pad),
                    mode='reflect')
    padded_sq = padded * padded

    integral = integral_image(padded)
    integral_sq = integral_image(padded_sq)

    kern = np.zeros((w + 1,) * image.ndim)
    for indices in itertools.product(*([[0, -1]] * image.ndim)):
        kern[indices] = (-1) ** (image.ndim % 2 != np.sum(indices) % 2)

    sum_full = ndi.correlate(integral, kern, mode='constant')
    m = crop(sum_full, (left_pad, right_pad)) / (w ** image.ndim)
    sum_sq_full = ndi.correlate(integral_sq, kern, mode='constant')
    g2 = crop(sum_sq_full, (left_pad, right_pad)) / (w ** image.ndim)
    # Note: we use np.clip because g2 is not guaranteed to be greater than
    # m*m when floating point error is considered
    s = np.sqrt(np.clip(g2 - m * m, 0, None))
    return m, s
