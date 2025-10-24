import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
from skimage.registration import phase_cross_correlation
from scipy.ndimage import center_of_mass
from tqdm import tqdm
import time
from skimage import io
#from save_data_h5_click import save_data

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QCheckBox, QMessageBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


from matplotlib.widgets import RectangleSelector


# from databroker import Broker
# db = Broker.named('hxn')
from hxntools.CompositeBroker import db
from hxntools.scan_info import get_scan_positions

def onselect(eclick, erelease):
    """
    Callback for rectangle selector.
    eclick and erelease are matplotlib events at press and release.
    """
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    xmin, xmax = sorted([x1, x2])
    ymin, ymax = sorted([y1, y2])

    #print(f"Selected window: X [{xmin}:{xmax}], Y [{ymin}:{ymax}]")
    roi_coords['xmin'] = xmin
    roi_coords['xmax'] = xmax
    roi_coords['ymin'] = ymin
    roi_coords['ymax'] = ymax

    # Example: show the cropped region
    roi = tmp[ymin:ymax, xmin:xmax]
    plt.figure()
    plt.imshow(roi)
    plt.title('Selected ROI')
    plt.xlabel(f"Selected window: X [{xmin}:{xmax}], Y [{ymin}:{ymax}]")
    plt.show()

def rm_pixel(data,ix,iy):
    data[ix,iy] = np.median(data[ix-1:ix+1,iy-1:iy+1])
    return data
def load_one_frame(scan_num,det_name,frame_n):
    sid = int(scan_num)
    h = db[sid]
    h5_handle = db.reg.get_spec_handler(h.table()[det_name][1][:-2])._handle
    return h5_handle['entry/data/data'][frame_n]



def load_data(scan_num,det_name,mesh_flag=True,fly_flag=True,check_flag=False):
    sid = int(scan_num)

    bl = db[sid].table('baseline')
    df = db.get_table(db[sid],fill=False)
    #images = db.get_images(db[sid],name=det_name)
    h = db[sid]

    global roi_coords, tmp
    roi_coords = {}



    
    tmp = np.log(np.flipud(load_one_frame(sid,det_name,0)).T)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(tmp)
    ax.set_title("Drag to select ROI")
    selector = RectangleSelector(
        ax,
        onselect,
        #drawtype='box',
        useblit=True,
        button=[1],  # left mouse button
        minspanx=5, minspany=5,
        spancoords='pixels',
        interactive=True
    )
    print("select ROI")
    plt.show()
    plt.pause(10)
    plt.close()
    plt.close()
    print('loading data ...')
    
    images = list(h.data(det_name))
    images = np.array(np.squeeze(images))    
    #print(np.shape(images))

    plan_args = h.start
    distance = plan_args['scan']['detector_distance']
    try:
        angle = bl.zpsth[1]
    except:
        angle = 0
    dcm_th = bl.dcm_th[1]
    energy_kev = 12.39842 / (2.*3.1355893 * np.sin(dcm_th * np.pi / 180.))

    num_frame, x_frame, y_frame = np.shape(images)

    data = []

    if mesh_flag:
        if fly_flag:
            x_range = plan_args['scan']['scan_input'][1] - plan_args['scan']['scan_input'][0]
            y_range = plan_args['scan']['scan_input'][4] - plan_args['scan']['scan_input'][3]
            x_num = plan_args['scan']['scan_input'][2]
            y_num = plan_args['scan']['scan_input'][5]
            #x_range = plan_args['scan_end1']-plan_args['scan_start1']
            #y_range = plan_args['scan_end2']-plan_args['scan_start2']
            #x_num = plan_args['num1']
            #y_num = plan_args['num2']
        else:
            x_range = plan_args['args'][2]-plan_args['args'][1]
            y_range = plan_args['args'][6]-plan_args['args'][5]
            x_num = plan_args['args'][3]
            y_num = plan_args['args'][7]
        dr_x = 1.*x_range/x_num
        dr_y = 1.*y_range/y_num
        x_range = x_range - dr_x
        y_range = y_range - dr_y
    else:
        x_range = plan_args['x_range']
        y_range = plan_args['y_range']
        dr_x = plan_args['dr']
        dr_y = 0

    motors = h.start['motors']
    #print(motors)
    # x = np.array(df[motors[0]])
    # y = np.array(df[motors[1]])
    x,y = get_scan_positions(h)
    
    points = np.zeros((2,num_frame))
    points[0,:] = x#[:500]
    points[1,:] = y#[:500]

    ic = np.squeeze(np.array(list(h.data('sclr1_ch4'))))
    #ic = np.squeeze(np.array(list(h.data('sclr1_ch3'))))
    if ic[0] == 0:
        ic[0] = ic[1]
    
    print('processing data ...')
    for i in tqdm(range(num_frame)):

        tt = (np.flipud(images[i,:,:]).T)
        nx,ny = np.shape(tt)

        t = tt
        if check_flag:
            t = tt
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(t)
            ax.set_title("Drag to select ROI")
            selector = RectangleSelector(
                ax,
                onselect,
                #drawtype='box',
                useblit=True,
                button=[1],  # left mouse button
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True
            )
            print("select ROI")
            plt.show()
            
        t = t * ic[0] / ic[i]
        if i == 0:
            cx = (roi_coords['ymax'] + roi_coords['ymin'])//2
            cy = (roi_coords['xmax'] + roi_coords['xmin'])//2
            n = roi_coords['ymax'] - roi_coords['ymin']
            nn = roi_coords['xmax'] - roi_coords['xmin']
            n += np.mod(n,2)
            nn += np.mod(nn,2)
            nmin = np.min((n,nn))
            n = nmin
            nn = nmin
            nx,ny = np.shape(t)
            data = np.zeros((num_frame,n,nn))

        #t = rm_pixel(t,500,296)    
        
        tmptmp = t[cx-n//2:cx+n//2,cy-nn//2:cy+nn//2]

        data[i,:,:] = np.fft.fftshift(tmptmp)

    threshold = 1.
    data = data - threshold
    data[data < 0.] = 0.
    data = np.sqrt(data)
    return data, angle,x_range, y_range, dr_x, dr_y, points, energy_kev, distance, n, nn #, Ni_xrf, Au_xrf


def save_data(scan_num, data_dir, mesh_flag=True, fly_flag=True, check_flag=False):
    scan_num = int(scan_num)
    mesh_flag = int(mesh_flag)
    fly_flag = int(fly_flag)
    try:
        if 'merlin1' in db[scan_num].start.scan['detectors']:
            det_name = 'merlin1'
            det_pixel_um = 55.
        if 'eiger2' in db[scan_num].start.scan['detectors']:
            det_name = 'eiger2_image'
            det_pixel_um = 75.
        print(det_name)
    except:
        if 'merlin1' in db[scan_num].start['detectors']:
            det_name = 'merlin1'
            det_pixel_um = 55.
        if 'eiger2' in db[scan_num].start['detectors']:
            det_name = 'eiger2_image'
            det_pixel_um = 75.
        print(det_name)
    #det_name = 'eiger2_image'
    #det_pixel_um = 75.

    data, angle, x_range, y_range, dr_x, dr_y, points, energy_kev, det_distance_m, nx, ny = load_data(
        scan_num, det_name, mesh_flag, fly_flag, check_flag
    )
    #print(np.shape(data), 'angle: ', angle)
    #print('energy:', energy_kev)
    lambda_nm = 1.2398 / energy_kev
    pixel_size = lambda_nm * 1.e-9 * det_distance_m / (nx * det_pixel_um * 1e-6)
    depth_of_field = lambda_nm * 1.e-9 / (nx / 2 * det_pixel_um * 1.e-6 / det_distance_m) ** 2
    #print('pixel num, pixel size, depth of field: ', nx, pixel_size, depth_of_field)

    print('saving data ...')
    # Ensure the directory path ends with a slash
    if not data_dir.endswith('/'):
        data_dir += '/'

    import os
    os.makedirs(data_dir, exist_ok=True)

    h5_path = f"{data_dir}scan_{scan_num}.h5"
    with h5py.File(h5_path, 'w') as hf:
        dset = hf.create_dataset('diffamp', data=data)
        dset = hf.create_dataset('points', data=points)
        dset = hf.create_dataset('x_range', data=x_range)
        dset = hf.create_dataset('y_range', data=y_range)
        dset = hf.create_dataset('dr_x', data=dr_x)
        dset = hf.create_dataset('dr_y', data=dr_y)
        dset = hf.create_dataset('z_m', data=det_distance_m)
        dset = hf.create_dataset('lambda_nm', data=lambda_nm)
        dset = hf.create_dataset('ccd_pixel_um', data=det_pixel_um)
        dset = hf.create_dataset('angle', data=angle)
        #dset = hf.create_dataset('Ni_xrf',data=Ni_xrf)
        #dset = hf.create_dataset('Au_xrf',data=Au_xrf)

# ----------------------------
# DPC helper functions
# ----------------------------
def pad_gradient(g):
    nx,ny = np.shape(g)
    gg = np.zeros((2*nx,2*ny))
    gg[:nx,:ny] = g
    gg[nx:,:ny] = np.flipud(g)
    gg[:nx,ny:] = np.fliplr(g)
    gg[nx:,ny:] = np.flipud(np.fliplr(g))
    return gg

def phase_uw(gx,gy,step_nm, sign=1):

    nx,ny = np.shape(gx)
    w = 1 # Weighting parameter

    gx_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gx))) / np.sqrt(np.size(gx))
    gy_fft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(gy))) / np.sqrt(np.size(gy))


    dqx_1d = 2*np.pi/(nx*step_nm*1e-9) * (np.arange(nx) - nx//2) 
    dqy_1d = 2*np.pi/(ny*step_nm*1e-9) * (np.arange(ny) - ny//2)
    
    dqx,dqy = np.meshgrid(dqy_1d,dqx_1d)

    #print(np.shape(gx),np.shape(dqx))
    tmp = (gx_fft * dqx + w * gy_fft * dqy) / (dqx**2 + dqy**2 + 1e-12)
    
    phi = sign*1j * np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(tmp))) * np.sqrt(np.size(gx))
    
    return phi

def rm_linear_x_bg(img,ln):
    nx, ny = np.shape(img)
    line_bg = np.mean(img[:,0:ln],axis=1)
    for i in range(ny):
        img[:,i] -= line_bg

    return img

def rm_linear_y_bg(img,ln):
    nx, ny = np.shape(img)
    line_bg = np.mean(img[0:ln,:],axis=0)
    for i in range(nx):
        img[i,:] -= line_bg

    return img

def cal_dpc(fn,fnp,sign=1,lnx=20,lny=20,disp_flag=False, rm_linear_x_flag=True, rm_linear_y_flag=True, quad_num=1):
    stime = time.perf_counter()
    #fnp = '/data/home/home/xjhuang/GPU_ptycho/h5_data/'
    f = h5py.File(fnp+'scan_'+str(fn)+'.h5','r')

    data = np.asarray(f['diffamp'])
    nz,nnx,nny = np.shape(data)
    z_m = np.asarray(f['z_m'])
    det_px_um = np.array(f['ccd_pixel_um'])
    lambda_nm = np.array(f['lambda_nm'])
    step_nm = np.array(f['dr_x']) * 1e3
    p = np.array(f['points'])
    x_dr = np.array(f['dr_x'])
    x_range = np.array(f['x_range'])
    y_dr = np.array(f['dr_y'])
    y_range = np.array(f['y_range'])
    f.close()

    ny = int(np.round(x_range/x_dr)) + 1
    nx = int(np.round(y_range/y_dr)) + 1
    count = 0
    
    ref = np.fft.fftshift(np.mean(data,axis=0))

    # cal dpc
    dpc = np.zeros((nx,ny,4))

    # Ensure det_px_um and z_m are scalars before the loop
    det_px_um_scalar = float(np.array(det_px_um).flatten()[0])
    z_m_scalar = float(np.array(z_m).flatten()[0])
    
    for i in tqdm(range(nx)):
        #if np.mod((i+1),50) == 0:
        #    print('row ', i+1, '/', nx)
        for j in range(ny):
            tmp = np.fft.fftshift(data[count,:,:])
            (cx, cy) = center_of_mass(tmp**2)
            cx = float(np.array(cx).flatten()[0])
            cy = float(np.array(cy).flatten()[0])
            dpc[i,j,0] = np.sum(tmp**2)
            if count == 0:
                cx_ref = cx
                cy_ref = cy
            dpc[i,j,1] = cx * det_px_um_scalar * 1e-6 / z_m_scalar
            dpc[i,j,2] = cy * det_px_um_scalar * 1e-6 / z_m_scalar
            count += 1
        
    dpc[:,:,0] /= np.max(dpc[:,:,0])
    dpc[:,:,1] -= cx_ref
    dpc[:,:,2] -= cy_ref
    dpc_h_rad = pad_gradient(dpc[:,:,1]) * 2 * np.pi / (lambda_nm*1e-9) 
    dpc_v_rad = pad_gradient(dpc[:,:,2]) * 2 * np.pi / (lambda_nm*1e-9) 

    stxm = dpc[:,:,0]
    pha = phase_uw(dpc_h_rad, dpc_v_rad, step_nm, sign)

    if quad_num == 1:
        tmp = pha[:nx,:ny].real - pha[:nx,:ny].real.max()
    elif quad_num == 2:
        tmp = np.flipud(pha[nx:,:ny].real - pha[nx:,:ny].real.max())
    elif quad_num == 3:
        tmp = np.fliplr(pha[:nx,ny:].real - pha[:nx,ny:].real.max())
    elif quad_num == 4:
        tmp = np.flipud(np.fliplr(pha[nx:,ny:].real - pha[nx:,ny:].real.max()))

    dpc[:,:,3] = tmp    
    if rm_linear_x_flag:
        #print('rm x line')
        tmp = rm_linear_x_bg(tmp,lnx)
        dpc[:,:,3] = tmp - tmp.max()
    if rm_linear_y_flag:
        #print('rm y line')
        tmp = rm_linear_y_bg(tmp,lny)
        dpc[:,:,3] = tmp - tmp.max()
        
    
    if disp_flag:
        plt.close('all')

        plt.figure(figsize=(12,10))
        plt.subplot(221)
        plt.imshow(dpc[:,:,1])
        plt.title('dpc h')
        plt.colorbar()

        plt.subplot(222)
        plt.imshow(dpc[:,:,2])
        plt.title('dpc v')
        plt.colorbar()

        plt.subplot(223)
        plt.imshow(dpc[:,:,0])
        plt.title('stxm')
        plt.colorbar()
        
        plt.subplot(224)
        plt.imshow(dpc[:,:,3])
        #plt.imshow(pha.real)
        plt.title('phase')
        plt.colorbar()


        plt.show()
        
    etime = time.perf_counter()
    print('spend ', etime-stime, 's')
    return dpc

# ----------------------------
# Main GUI
# ----------------------------
class DPCApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DPC Calculator (PyQt5)")

        # Main widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # --- Left panel: inputs ---
        input_layout = QVBoxLayout()

        # Add databroker load button at the top
        load_db_btn = QPushButton("Load from Databroker")
        load_db_btn.clicked.connect(self.load_from_databroker)
        input_layout.addWidget(load_db_btn)

        # Directory selector
        input_layout.addWidget(QLabel("Data directory:"))
        self.dir_input = QLineEdit()
        self.dir_input.setText("/data/home/home/xjhuang/GPU_ptycho/h5_data/")  # default
        input_layout.addWidget(self.dir_input)
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.select_directory)
        input_layout.addWidget(browse_btn)

        self.scan_input = self.make_input(input_layout, "Scan number:", "294677")
        #self.file_path_input = self.make_input(input_layout, "Path:", "")
        self.sign_input = self.make_input(input_layout, "Sign:", "1")
        self.quad_input = self.make_input(input_layout, "Quadrant:", "1")
        #self.x_line_correct_input   = self.make_input(input_layout, "x line correction (opt):", "0")
        #self.y_line_correct_input   = self.make_input(input_layout, "y line correction (opt):", "0")


        # Create checkboxes
        self.x_line_correction_checkbox = QCheckBox("x line correction")
        self.x_line_correction_checkbox.setChecked(False)  # Set default to unchecked
        input_layout.addWidget(self.x_line_correction_checkbox)


        self.y_line_correction_checkbox = QCheckBox("y line correction")
        self.y_line_correction_checkbox.setChecked(False)  # Set default to unchecked
        input_layout.addWidget(self.y_line_correction_checkbox)

        if self.y_line_correction_checkbox.isChecked():
            self.y_line_correct_input = 1
        else:
            self.y_line_correct_input = 0

        if self.x_line_correction_checkbox.isChecked():
            self.x_line_correct_input = 1
        else:
            self.x_line_correct_input = 0
            

        self.x_line_correction_checkbox.stateChanged.connect(self.update_label)
        self.y_line_correction_checkbox.stateChanged.connect(self.update_label)

        
        self.dpc = {}
        
        calc_btn = QPushButton("Load && Calculate")
        calc_btn.clicked.connect(self.run_dpc)
        input_layout.addWidget(calc_btn)

        save_btn = QPushButton("Save Result")
        save_btn.clicked.connect(self.save_dpc)
        input_layout.addWidget(save_btn)

        input_layout.addStretch()  # push widgets to top

        main_layout.addLayout(input_layout, stretch=0)

        # --- Right panel: Figure ---
        right_panel = QVBoxLayout()  # Create a vertical layout for the right panel

        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvas(self.fig)
        right_panel.addWidget(self.canvas)

        self.toolbar = NavigationToolbar(self.canvas, self)
        right_panel.addWidget(self.toolbar)  # Add toolbar below the canvas

        main_layout.addLayout(right_panel, stretch=1)

    def update_label(self):
        if self.y_line_correction_checkbox.isChecked():
            self.y_line_correct_input = 1
        else:
            self.y_line_correct_input = 0

        if self.x_line_correction_checkbox.isChecked():
            self.x_line_correct_input = 1
        else:
            self.x_line_correct_input = 0
            
        
    def save_dpc(self):
        if self.dir_input.text().strip()[-1] != '/':
            #np.save(self.dir_input.text().strip()+'/dpc_'+str(self.scan_input.text().strip())+'.npy',self.dpc)
            io.imsave(self.dir_input.text().strip()+'/dpc_'+str(self.scan_input.text().strip())+'.tif',self.dpc.astype(np.float32))
        else:
            #np.save(self.dir_input.text().strip()+'dpc_'+str(self.scan_input.text().strip())+'.npy',self.dpc)
            io.imsave(self.dir_input.text().strip()+'dpc_'+str(self.scan_input.text().strip())+'.tif',self.dpc.astype(np.float32))
        
        
        
    def make_input(self, layout, label_text, default):
        layout.addWidget(QLabel(label_text))
        line_edit = QLineEdit()
        line_edit.setText(default)
        layout.addWidget(line_edit)
        return line_edit

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Data Directory")
        if dir_path:
            self.dir_input.setText(dir_path)
    
    def load_from_databroker(self):
        from PyQt5.QtWidgets import QInputDialog
        scan_num, ok = QInputDialog.getInt(self, "Load from Databroker", "Enter scan number:")
        if ok:
            try:
                data_dir = self.dir_input.text().strip()
                save_data(scan_num, data_dir)
                QMessageBox.information(self, "Success", f"Scan {scan_num} loaded and saved to {data_dir}.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load scan {scan_num}: {e}")

    def run_dpc(self):
        try:
            scan_num = int(self.scan_input.text())
            sign     = int(self.sign_input.text())
            path = self.dir_input.text().strip()
            if path[-1] != '/':
                path += '/'
            quad_num = int(self.quad_input.text())
            x_line_correct   = bool(self.x_line_correct_input)
            y_line_correct   = bool(self.y_line_correct_input)
        except ValueError:
            QMessageBox.critical(self, "Input Error", "Please check your input values.")
            return

        try:
            self.dpc = cal_dpc(scan_num, fnp=path, sign=sign, quad_num=quad_num, rm_linear_x_flag=x_line_correct, rm_linear_y_flag=y_line_correct)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.fig.clear()
        axes = self.fig.subplots(2, 2)
        im0 = axes[0,0].imshow(self.dpc[:,:,1], cmap='gray'); axes[0,0].set_title("DPC H")
        im1 = axes[0,1].imshow(self.dpc[:,:,2], cmap='gray'); axes[0,1].set_title("DPC V")
        im2 = axes[1,0].imshow(self.dpc[:,:,0], cmap='gray'); axes[1,0].set_title("STXM")
        im3 = axes[1,1].imshow(self.dpc[:,:,3], cmap='gray'); axes[1,1].set_title("Phase")

        for ax, im in zip(axes.flat, (im0,im1,im2,im3)):
            self.fig.colorbar(im, ax=ax)
        self.fig.tight_layout()
        self.canvas.draw()

# ----------------------------
# Run the application
# ----------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DPCApp()
    window.resize(1000, 700)
    window.show()
    sys.exit(app.exec_())

