import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed # type: ignore


SOLVERS = [
    "Nelder-Mead",
    "Powell",
    "CG",
    "BFGS",
    "Newton-CG",
    "Anneal",
    "L-BFGS-B",
    "TNC",
    "COBYLA",
    "SLS-QP",
    "dogleg",
    "trust-ncg",
]


# Placeholder for the RSS function (to be minimized)
def rss(v, xdata, ydata, beta):
    """Function to be minimized in the Nelder Mead algorithm"""
    fitted_curve = xdata * v[0] * np.exp(v[1] * beta)
    return np.sum(np.abs(ydata - fitted_curve) ** 2)

# Cache for storing the beta values (to avoid recalculating them multiple times)
rss_cache = {}

# Function to compute the beta value, which depends on the input xdata
def get_beta(xdata):
    length = len(xdata)
    try:
        beta = rss_cache[length]
    except Exception:
        beta = 1j * (np.arange(length) - np.floor(length / 2.0))  # Frequency component
        rss_cache[length] = beta
    return beta

# Function to calculate the shift of the image (calculating the frequency domain)
def calc_img_shift(img_array_2d):
    """Calculates the image shift by Fourier Transforming the summed projections along axes"""
    xline = np.sum(img_array_2d, axis=0)  # Sum over rows (vertical projection)
    yline = np.sum(img_array_2d, axis=1)  # Sum over columns (horizontal projection)

    fx = np.fft.fftshift(np.fft.ifft(xline))  # Fourier transform and shift to the center
    fy = np.fft.fftshift(np.fft.ifft(yline))  # Fourier transform and shift to the center

    return fx, fy

def recon(gx, gy, dx=0.1, dy=0.1, pad=1, w=1.0, filter = True):
    """
    Reconstruct the final phase image
    Parameters
    ----------
    gx : 2-D numpy array
        phase gradient along x direction

    gy : 2-D numpy array
        phase gradient along y direction

    dx : float
        scanning step size in x direction (in micro-meter)

    dy : float
        scanning step size in y direction (in micro-meter)

    pad : float
        padding parameter
        default value, pad = 1 --> no padding
                    p p p
        pad = 3 --> p v p
                    p p p

    w : float
        weighting parameter for the phase gradient along x and y direction when
        constructing the final phase image

    Returns
    ----------
    phi : 2-D numpy array
        final phase image

    References
    ----------
    [1] Yan, Hanfei, Yong S. Chu, Jorg Maser, Evgeny Nazaretski, Jungdae Kim,
    Hyon Chol Kang, Jeffrey J. Lombardo, and Wilson KS Chiu, "Quantitative
    x-ray phase imaging at the nanoscale by multilayer Laue lenses," Scientific
    reports 3 (2013).

    """

    rows,cols = gx.shape

    gx_padding = np.zeros((pad * rows, pad * cols), dtype="d")
    gy_padding = np.zeros((pad * rows, pad * cols), dtype="d")

    gx_padding[(pad // 2) * rows : (pad // 2 + 1) * rows, (pad // 2) * cols : (pad // 2 + 1) * cols] = gx
    gy_padding[(pad // 2) * rows : (pad // 2 + 1) * rows, (pad // 2) * cols : (pad // 2 + 1) * cols] = gy

    tx = np.fft.fftshift(np.fft.fft2(gx_padding))
    ty = np.fft.fftshift(np.fft.fft2(gy_padding))

    c = np.zeros((pad * rows, pad * cols), dtype=complex)

    mid_col = pad * cols // 2 + 1
    mid_row = pad * rows // 2 + 1

    ax = 2 * np.pi * (np.arange(pad * cols) + 1 - mid_col) / (pad * cols * dx)
    ay = 2 * np.pi * (np.arange(pad * rows) + 1 - mid_row) / (pad * rows * dy)

    kappax, kappay = np.meshgrid(ax, ay)

    c = -1j * (kappax * tx + w * kappay * ty)

    c = np.ma.masked_values(c, 0)
    c /= kappax**2 + w * kappay**2
    c = np.ma.filled(c, 0)
    if filter:
        # use a high-pass filter to suppress amplified low-frequency signals, H.Y, 08/02/2022
        f = 1 - 0.9 * np.exp(-np.square(kappax * dx) - np.square(kappay * dy))
        c = f * c

    c = np.fft.ifftshift(c)
    phi_padding = np.fft.ifft2(c)
    phi_padding = -phi_padding.real

    phi = phi_padding[(pad // 2) * rows : (pad // 2 + 1) * rows, (pad // 2) * cols : (pad // 2 + 1) * cols]

    return phi

# Function to run DPC (Differential Phase Contrast) on a single image
def run_dpc(img, ref_fx, ref_fy, start_point, 
            max_iters, solver, reverse_x, reverse_y,
            roi_slice=None, mask=None):
    """
    Run DPC reconstruction on a single image with optional masking and cropping.

    Parameters
    ----------
    img : 2D numpy array or h5py slice
        Full image to process.
    roi_slice : tuple of slices
        Pixel slice (y, x) for cropping.
    mask : 2D array
        Binary mask to apply after cropping (same shape as cropped ROI).
    """
    # Apply ROI cropping
    if roi_slice is not None:
        img = img[roi_slice]

    # Apply mask
    if mask is not None:
        img = img * mask

    # Compute DPC shifts
    fx, fy = calc_img_shift(img)

    res_x = minimize(
        rss, start_point,
        args=(ref_fx, fx, get_beta(ref_fx)),
        method=solver,
        tol=1e-6,
        options=dict(maxiter=max_iters)
    )
    vx_x = res_x.x
    rx = res_x.fun
    a = vx_x[0]
    gx = reverse_x * vx_x[1]

    res_y = minimize(
        rss, start_point,
        args=(ref_fy, fy, get_beta(ref_fy)),
        method=solver,
        tol=1e-6,
        options=dict(maxiter=max_iters)
    )
    vy_y = res_y.x
    ry = res_y.fun
    gy = reverse_y * vy_y[1]

    return a, gx, gy, rx, ry


# Function to process a stack of images (DPC reconstruction)
def process_images_stack(det_images, ref_fx, ref_fy, start_point, 
                         max_iter, solver, reverse_x, reverse_y,
                         roi_slice=None, mask=None):
    # Detect and resolve lazy loader
    if callable(det_images):
        det_images = det_images()  # h5py.Dataset

    num_images = det_images.shape[0]

    # Initialize arrays to hold results
    a = np.zeros((num_images,))
    gx = np.zeros((num_images,))
    gy = np.zeros((num_images,))
    rx = np.zeros((num_images,))
    ry = np.zeros((num_images,))

    # Parallel processing
    results = Parallel(n_jobs=-1)(
        delayed(run_dpc)(
            det_images[i, :, :],
            ref_fx, ref_fy, start_point,
            max_iter, solver, reverse_x, reverse_y,
            roi_slice=roi_slice,
            mask=mask
        )
        for i in range(num_images)
    )

    for i, result in enumerate(results):
        a[i], gx[i], gy[i], rx[i], ry[i] = result

    return a, gx, gy, rx, ry


# Main function to reconstruct DPC from a stack of images
def recon_dpc_from_im_stack(det_images, ref_image_num=1, start_point=[1, 0], num_xy=[20, 20],
                            max_iter=1000, solver="Nelder-Mead", reverse_x=1, reverse_y=1,
                            energy=12, det_pixel=55, det_dist=2.05, dxy=[0.020, 0.020],
                            roi_slice=None, mask=None):
    """
    Reconstructs DPC phase image from a 3D (or lazy) image stack.

    Parameters
    ----------
    det_images : array, h5py.Dataset, or lazy loader
    roi_slice : tuple of slice
        Pixel region to crop per image before processing
    mask : 2D numpy array
        Binary mask applied to each cropped image
    """

    # Resolve lazy loader if needed
    if callable(det_images):
        dataset = det_images()
        lazy = True
    else:
        dataset = det_images
        lazy = False

    shape = dataset.shape
    if len(shape) == 4:
        ydim, xdim, yroi, xroi = shape
        dataset = dataset.reshape(-1, yroi, xroi)
    elif len(shape) == 3:
        _, yroi, xroi = shape
        ydim, xdim = num_xy
    else:
        raise ValueError(f"Invalid input shape: {shape}")

    # Get reference image
    ref_image = dataset[ref_image_num, :, :]
    if roi_slice is not None:
        ref_image = ref_image[roi_slice]
    if mask is not None:
        ref_image = ref_image * mask

    ref_fx, ref_fy = calc_img_shift(ref_image)

    # Compute DPC on full/lazy stack
    a, gx, gy, rx, ry = process_images_stack(
        dataset, ref_fx, ref_fy, start_point,
        max_iter, solver, reverse_x, reverse_y,
        roi_slice=roi_slice,
        mask=mask
    )

    # DPC scaling
    gx *= len(ref_fx) * det_pixel / (12.4e-4 / energy * det_dist * 1e6)
    gy *= len(ref_fy) * det_pixel / (12.4e-4 / energy * det_dist * 1e6)

    gx_ = gx.reshape(ydim, xdim)
    gy_ = gy.reshape(ydim, xdim)
    a_ = a.reshape(ydim, xdim)

    phi = recon(gx_, gy_, dxy[0], dxy[1])

    return a_, gx_, gy_, phi

