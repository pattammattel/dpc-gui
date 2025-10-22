#!/usr/bin/env python
"""
Created on May 23, 2013
@author: Cheng Chang (cheng.chang.ece@gmail.com)
         Computer Science Group, Computational Science Center
         Brookhaven National Laboratory

updated with new flyscan on Dec 2024(Ajith Pattammattel)

This code is for Differential Phase Contrast (DPC) imaging based on Fourier-shift fitting
implementation.

Reference: Yan, H. et al. Quantitative x-ray phase imaging at the nanoscale by multilayer
           Laue lenses. Sci. Rep. 3, 1307; DOI:10.1038/srep01307 (2013).

Test data is available at:
https://docs.google.com/file/d/0B3v6W1bQwN_AdjZwWmE3WTNqVnc/edit?usp=sharing
"""
from __future__ import print_function, division
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import minimize
import time
import h5py
import tqdm
import tifffile as tf
# from dpcmaps.db_config.db_config import db
# from hxntools.scan_info import get_scan_positions
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
from dpc_calc2 import *

hxn_detector_config = {'merlin1':{'pixel_size_um':55,
                                  'distance_m':0.500},
                       'merlin2':{'pixel_size_um':55,
                                  'distance_m':1.467},
                       'eiger1':{'pixel_size_um':75,
                                  'distance_m':2.050},
                        'eiger2_image':{'pixel_size_um':75,
                                  'distance_m':2.050}
                       }


rss_cache = {}
rss_iters = 0

def remove_hot_pixels(image, NSigma=8, fill_zero =False):
    """
    Removes hot pixels from an image by replacing them with the median of the surrounding pixels.

    Args:
        image (ndarray): 2D NumPy array representing the image.
        NSigma (int, optional): The number of standard deviations to use as a threshold for detecting hot pixels. Default is 3.

    Returns:
        ndarray: Image with hot pixels removed.
    """
    # Step 1: Flatten the image and calculate the mean and standard deviation
    flat_image = image.flatten()
    mean_val = np.mean(flat_image)
    std_val = np.std(flat_image)
    
    # Step 2: Identify hot pixels (pixels that are more than NSigma standard deviations away from the mean)
    hot_pixel_mask = np.abs(image - mean_val) > (NSigma * std_val)
    
    # Step 3: Replace hot pixels with the median of the surrounding pixels
    # Create a copy of the image to modify
    image_cleaned = image.copy()
    
    # Replace hot pixels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if hot_pixel_mask[i, j]:

                if fill_zero:
                    image_cleaned[i, j] = 0

                else:

                    # Take the median of the surrounding pixels (within a 3x3 window)
                    # Ensure the window stays within the bounds of the image
                    min_i, max_i = max(i-1, 0), min(i+2, image.shape[0])
                    min_j, max_j = max(j-1, 0), min(j+2, image.shape[1])
                    
                    # Get the neighborhood of the current pixel
                    neighborhood = image[min_i:max_i, min_j:max_j]
                    
                    # Replace the hot pixel with the median of the surrounding pixels
                    image_cleaned[i, j] = np.median(neighborhood)
    
    return image_cleaned




def load_metadata(scan_num:int, det_name:str):
    '''
    Get all metadata for the given scan number and detector name

    Parameters:
        - db:
            a Broker instance. For HXN experiments they are db1, db2, and db_old
        - scan_num: int
            the scan number
        - det_name: str
            the detector name

    Return:
        A dictionary that holds the metadata (except for those directly related to the image)
    '''
    header = db[int(scan_num)]
    start_doc = header.start
    scan_motors = start_doc['motors']
    items = [det_name, 'sclr1_ch3', 'sclr1_ch4'] + scan_motors
    #print(f"{items = }")
    bl = db.get_table(header, stream_name='baseline')
    df = db.get_table(header, fields=items, fill=False)
    #df = db.get_table(header)
    #images = db_old.get_images(db_old[sid], name=det_name)
    # get energy_kev
    energy_kev = bl.energy.iloc[0]
    
    if "dimensions" in start_doc:
        datashape = start_doc.dimensions
    elif "shape" in start_doc:
        datashape = start_doc.shape
    elif "num_points" in start_doc:
        datashape = [start_doc.num_points]

    if "per_points" in start_doc:
        x_step_size, y_step_size = start_doc.per_points[-1]
    elif "scan" in start_doc:
        scan_input = start_doc.scan.get('scan_input')
        x_step_size = abs(scan_input[1]-scan_input[0])/scan_input[2]
        y_step_size = abs(scan_input[4]-scan_input[3])/scan_input[5]
    
    if 'plan_args' in start_doc:
        plan_args = header.start['plan_args']
        scan_type = header.start['plan_name']

    elif 'scan' in start_doc:
        scan_type = start_doc.scan.get('type')
        scan_input = start_doc.scan.get('scan_input')
        mtr1,mtr2 = start_doc['motors']
        plan_args = {'motor1':mtr1 ,
                     'scan_start1':scan_input[0],
                     'scan_end1':scan_input[1] ,
                     'num1': scan_input[2],
                     'motor2':mtr2 ,
                     'scan_start2':scan_input[3],
                     'scan_end2':scan_input[4] ,
                     'num2': scan_input[5],
                     }
        
    print(f"{scan_type = }")

    # get scan_type, x_range, y_range, dr_x, dr_y
    if scan_type == 'FlyPlan2D' or '2D_FLY_PANDA':
        x_range = plan_args['scan_end1']-plan_args['scan_start1']
        y_range = plan_args['scan_end2']-plan_args['scan_start2']
        x_num = datashape[0]
        y_num = datashape[1]
        dr_x = x_step_size
        dr_y = y_step_size
        x_range = x_range - dr_x
        y_range = y_range - dr_y
    elif scan_type == 'rel_spiral_fermat' or scan_type == 'fermat':
        x_range = plan_args['x_range']
        y_range = plan_args['y_range']
        dr_x = plan_args['dr']
        dr_y = 0
    else:
        x_range = plan_args['args'][2]-plan_args['args'][1]
        y_range = plan_args['args'][6]-plan_args['args'][5]
        x_num = plan_args['args'][3]
        y_num = plan_args['args'][7]
        dr_x = 1.*x_range/x_num
        dr_y = 1.*y_range/y_num
        # x_range = x_range - dr_x
        # y_range = y_range - dr_y

    # get points
    points = np.array(get_scan_positions(header)[0])
    # get angle, ic
    if scan_type == '2D_FLY_PANDA' and scan_motors[1] == 'zpssy':
        angle = bl.zpsth[1]
        ic = np.array(list(header.data('sclr1_ch4')))

    elif scan_type == '2D_FLY_PANDA' and scan_motors[1] == 'dssy':
        angle = bl.dsth[1]
        ic = np.array(list(header.data('sclr1_ch4')))
    
    elif scan_motors[1] == 'ssy':
        angle = 0#bl.zpsth[1]
        ic = np.asfarray(df['sclr1_ch3'])

    elif scan_type == 'FlyPlan2D' and scan_motors[1] == 'zpssy':
        angle = bl.zpsth[1]
        ic = np.asfarray(df['sclr1_ch4'])
        
    elif scan_type == 'FlyPlan2D' and scan_motors[1] == 'dssy':
        angle = bl.dsth[1]
        ic = np.asfarray(df['sclr1_ch4'])

    data_len = int(x_num*y_num)
    # get diffamp dimensions (uncropped!)
    nz, = df[det_name].shape
    mds_table = df[det_name]

    # # get nx and ny by looking at the first image
    # img = db.reg.retrieve(mds_table.iat[0])[0]
    # #print(f"{img.shape = }")
    # if np.ndim(img) == 3:
    #     _,nx, ny = img.shape 
    # elif np.ndim(img) == 2:
    #     nx, ny = img.shape 


    # write everything we need to a dict
    metadata = dict()
    metadata['scan_id'] = start_doc['scan_id']
    metadata['scan_type'] = scan_type
    metadata['xray_energy_kev'] = energy_kev
    metadata['scan_type'] = scan_type
    metadata['dr_x'] = dr_x
    metadata['dr_y'] = dr_y
    metadata['x_range'] = x_range
    metadata['y_range'] = y_range
    metadata['datashape'] = datashape
    metadata['points'] = points
    metadata['angle'] = angle
    metadata['ic'] = np.squeeze(ic)
    metadata['ccd_pixel_um'] = hxn_detector_config[det_name]['pixel_size_um']
    metadata['det_distance_m'] = hxn_detector_config[det_name]['distance_m']
    # metadata['nz'] = nz
    # metadata['nx'] = nx
    # metadata['ny'] = ny
    metadata['data_len'] = data_len
    metadata['mds_table'] = mds_table

    return metadata

def make_hdf_dpc(start_scan_id = -1, end_scan_id = -1, det_name = 'eiger1',
                 crop_params = {'x_pos':0,'y_pos':0, 'x_len':100, 'y_len':100},
                 hot_pixels_list = [], wd = '.', plot_roi_img = False):
    #hdr.descriptors[0].data_keys.keys()
    sids = np.linspace(start_scan_id, end_scan_id,abs(end_scan_id-start_scan_id)+1)
    

    for sid_ in tqdm.tqdm(sids, desc="DPC:make h5"):
        h = db[int(sid_)]
        sid = h.start["scan_id"]
        start_doc = h.start

        if 'scan' in start_doc and det_name == 'eiger1':
            det_name = 'eiger2_image'


        if det_name in hxn_detector_config.keys():
            det_dist = hxn_detector_config[det_name]['distance_m']
            det_pixel = hxn_detector_config[det_name]['pixel_size_um']
        else:
            det_dist = 2.05
        if start_doc['plan_type'] == 'FlyPlan2D' or start_doc['scan']['type'] == '2D_FLY_PANDA':
            images = np.squeeze(list(h.data(det_name)))
            _, ysize,xsize = np.shape(images)
            xs = crop_params.get('x_pos', 0)
            xe = crop_params.get('x_len', xsize)
            ys = crop_params.get('y_pos', 0)
            ye = crop_params.get('y_len', ysize)
            cropped_imgs = images[:,ys:ys+ye,xs:xs+xe]
            if hot_pixels_list:
                for p in hot_pixels_list:
                    cropped_imgs[:,p[0],p[1]] = 0 #change to neighor average later
            #plan_args = db[sid].start['plan_args']
            bl = db.get_table(db[sid],stream_name='baseline')
            energy_kev = bl.energy.iloc[0]
            if "dimensions" in start_doc:
                datashape = start_doc.dimensions
            elif "shape" in start_doc:
                datashape = start_doc.shape
            elif "num_points" in start_doc:
                datashape = [start_doc.num_points]


            if "per_points" in start_doc:
                x_step_size, y_step_size = start_doc.per_points[-1]
            elif "scan" in start_doc:
                scan_input = start_doc.scan.get('scan_input')
                x_step_size = abs(scan_input[1]-scan_input[0])/scan_input[2]
                y_step_size = abs(scan_input[4]-scan_input[3])/scan_input[5]

            h5_name = os.path.join(wd,f'scan_{sid}_dpc.h5')

            with h5py.File(h5_name, 'w') as hf:
                dset1 = hf.create_dataset('data',
                                          data=cropped_imgs, 
                                          compression = 'gzip')
                dset2 = hf.create_dataset('scan_id',data=sid)
                dset3 = hf.create_dataset('datashape',data=datashape)
                dset4 = hf.create_dataset('energy',data=energy_kev)
                dset5 = hf.create_dataset('det_distance',data=det_dist)
                dset5 = hf.create_dataset('det_dist_pixel_size',data=[det_dist, det_pixel])
                dset6 = hf.create_dataset('xy_step_size',
                                          data=[x_step_size,y_step_size])
                
                print(f"{h5_name} is created")

                if plot_roi_img:
                    fig, ax = plt.subplots(0,2)
                    ax[0] = ax.imshow(images[0])
                    ax[1] = ax.imshow(images[0,ys:ys+ye,xs:xs+xe])

        else:
            print("Unknown Scan Type; Skipped")
            pass


def get_im_data_from_h5(h5_name):
    with h5py.File(h5_name, 'r') as hf:
        im_data = hf['data']
        return np.array(im_data)

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


def run_dpc(
    img,
    ref_img=None,
    ref_fx=None,
    ref_fy=None,
    start_point=[1, 0],
    max_iters=1000,
    solver="Nelder-Mead",
    reverse_x=1,
    reverse_y=1,
):
    """
    All units in micron
    """
    if ref_img is not None:
        ref_fx, ref_fy = calc_img_shift(ref_img)


    #print(f"{ref_fx.shape = },{ref_fx.shape = }")
    fx, fy = calc_img_shift(img)

    res = minimize(
        rss,
        start_point,
        args=(ref_fx, fx, get_beta(ref_fx)),
        method=solver,
        tol=1e-6,
        options=dict(maxiter=max_iters),
    )

    vx = res.x
    rx = res.fun
    a = vx[0]
    gx = reverse_x * vx[1]

    # vy = fmin(rss, start_point, args=(ref_fy, fy, get_beta(ref_fy)),
    #          maxiter=max_iters, maxfun=max_iters, disp=0)
    res = minimize(
        rss,
        start_point,
        args=(ref_fy, fy, get_beta(ref_fy)),
        method=solver,
        tol=1e-6,
        options=dict(maxiter=max_iters),
    )

    vy = res.x
    ry = res.fun
    gy = reverse_y * vy[1]

    # print(i, j, vx[0], vx[1], vy[1])
    return a, gx, gy, rx, ry



def main_from_h5(
    file_format,
    ref_image_num=1,
    savedir = '.',
    start_point=[1, 0],
    pool=None,
    first_image=1,
    solver="Nelder-Mead",
    display_fcn=None,
    invers=False,
):

    # print("DPC")
    # print("---")
    # print("\tFile format: %s" % file_format)
    # print("\tdx: %s" % dx)
    # print("\tdy: %s" % dy)
    # print("\trows: %s" % rows)
    # print("\tcols: %s" % cols)
    # print("\tstart point: %s" % start_point)
    # print("\tpixel size: %s" % pixel_size)
    # print("\tfocus to det: %s" % (focus_to_det / 1e6))
    # print("\tenergy: %s" % energy)
    # print("\tfirst image: %s" % first_image)
    # print("\treference image: %s" % ref_image)
    # print("\tsolver: %s" % solver)
    # print("\tROI: (%s, %s)-(%s, %s)" % (x1, y1, x2, y2))

    f = h5py.File(file_format, "r")
    sid = int(np.array(f['scan_id']))
    focus_to_det_m,  pixel_size = f['det_dist_pixel_size']
    focus_to_det  = focus_to_det_m*1e6
    det_images = np.array(f['data'])
    data_len = np.shape(det_images)[0]
    rows, cols = np.array(f['datashape'])
    pixel_size=np.array(f['det_dist_pixel_size'])
    dx=np.array(f['xy_step_size'])[0]
    dy=np.array(f['xy_step_size'])[1]
    energy=np.array(f['energy'])
    # Wavelength in micron
    lambda_ = 12.4e-4 / energy
    t0 = time.time()

    a = np.zeros((data_len))
    gx = np.zeros((data_len))
    gy = np.zeros((data_len))
    rx = np.zeros((data_len))
    ry = np.zeros((data_len))

    ref_fx, ref_fy = calc_img_shift(det_images[int(ref_image_num)])

    for i in tqdm.tqdm(range(data_len), desc= "DPC"):
        img = det_images[i]
        _a, _gx, _gy, _rx,_ry = run_dpc(img,
                                  ref_fx = ref_fx,
                                  ref_fy = ref_fy)
        a[i] = _a
        gx[i] = _gx
        gy[i] = _gy
        rx[i] = _rx
        ry[i] = _ry

    gx *= len(ref_fx) * pixel_size / (lambda_ * focus_to_det)
    gy *= len(ref_fy) * pixel_size / (lambda_ * focus_to_det)
    # ------------reconstruct the final phase image using gx and gy--------------------#
    #phi = recon(gx, gy, dx, dy)
    gx_ = gx.reshape(cols, rows)
    gy_ = gy.reshape(cols,rows)
    a_  = a.reshape(cols,rows)
    phi = recon(gx_,gy_,dx,dy)

    save_dir = os.path.join(savedir,f'dpc_results_{sid}')
    os.makedirs(save_dir, exist_ok=True)
    tf.imwrite(os.path.join(save_dir,'amp.tiff'), a_)
    tf.imwrite(os.path.join(save_dir,'gx.tiff'), gx_)
    tf.imwrite(os.path.join(save_dir,'gy.tiff'), gy_)
    tf.imwrite(os.path.join(save_dir,'phi.tiff'), phi)

    return a_, gx_, gy_, phi

def recon_dpc(
    sid,
    det_name,
    ref_image_num=1,
    norm = True,
    savedir = '.',
    start_point=[1, 0],
    max_iter = 1000,
    pool=None,
    solver="Nelder-Mead",
    reverse_x=1,
    reverse_y=1,
    display_fcn=None,
    invers=False,
    crop_params = {'x_pos':30,'y_pos':30, 
                   'x_len':300, 'y_len':300}):
    
    print(f"Loading data...{sid = }")
    mdata = load_metadata(sid, det_name=det_name)

    sid = int(mdata['scan_id'])
    pixel_size = mdata['ccd_pixel_um']
    focus_to_det  = mdata['det_distance_m']*1e6
    rows, cols = mdata['datashape']
    dx = mdata['dr_x']
    dy = mdata['dr_y']
    energy = mdata['xray_energy_kev']
    # Wavelength in micron
    lambda_ = 12.4e-4 / energy
    mds_table = mdata['mds_table']
    data_len = mdata['data_len']
    ic = mdata['ic']
    #print(f"{ic.shape = }")

    a = np.zeros((data_len))
    gx = np.zeros((data_len))
    gy = np.zeros((data_len))
    rx = np.zeros((data_len))
    ry = np.zeros((data_len))

    xs = crop_params.get('x_pos')
    xe = crop_params.get('x_len')
    ys = crop_params.get('y_pos')
    ye = crop_params.get('y_len')

    save_dir = os.path.join(savedir,f'dpc_results_{sid}')
    os.makedirs(save_dir, exist_ok=True)


    if mdata['scan_type'] == '2D_FLY_PANDA':
        det_images = db.reg.retrieve(mds_table.iat[0])[:,ys:ys+ye,xs:xs+xe]
        plt.figure()
        plt.imshow(det_images.mean(0))
        plt.title("det_roi")
        plt.show()
        plt.savefig(os.path.join(save_dir, 'det_roi.png'))
        if norm:
            det_images = det_images/ic[:, np.newaxis, np.newaxis]
        #print(f"{det_images.shape}")
        ref_img = det_images[int(ref_image_num)]
        ref_fx, ref_fy = calc_img_shift(ref_img)

        for i in tqdm.tqdm(range(len(det_images)), desc= f"DPC Recon: {sid = }"):
            img = det_images[i]
            _a, _gx, _gy, _rx,_ry = run_dpc(img,
                                    ref_fx = ref_fx,
                                    ref_fy = ref_fy,
                                    start_point=start_point,
                                    max_iters=max_iter,
                                    solver=solver,
                                    reverse_x=reverse_x,
                                    reverse_y=reverse_y)
            
            a[i] = _a
            gx[i] = _gx
            gy[i] = _gy
            rx[i] = _rx
            ry[i] = _ry
            
            #if i%500 == 0:
                #print(f"{_a = }, {_gx=}, {_gy=}")
    else:
        ref_img = np.squeeze(db.reg.retrieve(mds_table.iat[int(ref_image_num)]))
        if norm:
            ref_img = ref_img/ic[int(ref_image_num)]
        ref_fx, ref_fy = calc_img_shift(ref_img[ys:ys+ye,xs:xs+xe])

        plt.figure()
        plt.imshow(remove_hot_pixels(ref_img[ys:ys+ye,xs:xs+xe]), extent = [ys,ys+ye,xs,xs+xe])
        plt.title("det_roi")
        plt.show()
        plt.savefig(os.path.join(save_dir, 'det_roi.png'))

        for i in tqdm.tqdm(range(data_len), desc= "DPC"):
            img = np.squeeze(db.reg.retrieve(mds_table.iat[int(i)]))


            if norm: img = img/ic[i]

            img = remove_hot_pixels(img, NSigma=3)
            _a, _gx, _gy, _rx,_ry = run_dpc(img[ys:ys+ye,xs:xs+xe],
                                    ref_fx = ref_fx,
                                    ref_fy = ref_fy,
                                    start_point=start_point,
                                    max_iters=max_iter,
                                    solver=solver,
                                    reverse_x=reverse_x,
                                    reverse_y=reverse_y)
            
            #if i%1000 == 0:
                #print(f"{_a = }, {_gx=}, {_gy=}")

            a[i] = _a
            gx[i] = _gx
            gy[i] = _gy
            rx[i] = _rx
            ry[i] = _ry

    gx *= len(ref_fx) * pixel_size / (lambda_ * focus_to_det)
    gy *= len(ref_fy) * pixel_size / (lambda_ * focus_to_det)
    # ------------reconstruct the final phase image using gx and gy--------------------#
    #phi = recon(gx, gy, dx, dy)
    gx_ = gx.reshape(cols, rows)
    gy_ = gy.reshape(cols,rows)
    a_  = a.reshape(cols,rows)
    phi = recon(gx_,gy_,dx,dy)

    titles = ['Gradient_x','Gradient_y',
              'Amplitude','Phase']
    # Create a figure and a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    axs = axs.ravel()
    fig.suptitle(f"{sid}")

    for ax,title,im_data in zip(axs,titles,[gx_,gy_,a_,phi]):
        ax.axis('off')
        im = ax.imshow(im_data, cmap='viridis')
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="13%", pad=0.1)
        fig.colorbar(im, ax=ax, cax=cax)

    plt.tight_layout()
    plt.show(block=False)

    
    
    tf.imwrite(os.path.join(save_dir,'amp.tiff'), a_)
    tf.imwrite(os.path.join(save_dir,'gx.tiff'), gx_)
    tf.imwrite(os.path.join(save_dir,'gy.tiff'), gy_)
    tf.imwrite(os.path.join(save_dir,'phi.tiff'), phi)
    fig.savefig(os.path.join(save_dir, 'images.png'))

    return a_, gx_, gy_, phi


def parse_scan_range(str_scan_range):
    "strings like '123-345,789,348,456-890' "

    scanNumbers = []
    slist = str_scan_range.split(",")
    #print(slist)
    for item in slist:
        if "-" in item:
            slist_s, slist_e = item.split("-")
            print(slist_s, slist_e)
            scanNumbers.extend(list(np.linspace(int(slist_s.strip()), 
                                            int(slist_e.strip()), 
                                            int(slist_e.strip())-int(slist_s.strip())+1)))
        else:
            scanNumbers.append(int(item.strip()))
    
    return np.int_(sorted(scanNumbers))

def dpc_recon_with_db(sid_list, det_name = 'eiger1',
                 crop_params = {'x_pos':0,'y_pos':0, 'x_len':100, 'y_len':100},
                 hot_pixels_list = [], wd = '.'):
    
    for sid_ in (sid_list):
        make_hdf_dpc(start_scan_id = int(sid_), 
                     end_scan_id = int(sid_), 
                     det_name = det_name,
                     crop_params = crop_params,
                     hot_pixels_list = hot_pixels_list, 
                     wd = wd, 
                     plot_roi_img = False)
        
        a, gx, gy, phi = main_from_h5(os.path.join(wd,f'scan_{int(sid_)}_dpc.h5'),
                                       savedir=wd)