
import sys
import os
import warnings
import re
from difflib import get_close_matches
import numpy as np
import pyqtgraph as pg
import tifffile as tf
from scipy.ndimage import center_of_mass
from pyqtgraph import functions as fn
from functools import wraps
# from PyQt6 import QtWidgets, uic, QtCore, QtGui, QtTest
# from PyQt6.QtWidgets import QMessageBox
# from PyQt6.QtCore import QObject, pyqtSignal
from PyQt5 import QtWidgets, uic, QtCore, QtGui, QtTest
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QObject, pyqtSignal
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
ui_path = os.path.dirname(os.path.abspath(__file__))
warnings.filterwarnings('ignore', category=RuntimeWarning)
from h5_data_io import *
from dpc_kernel2 import *
from image_utils import *

#beamline specific
detector_list = ["eiger2_image","merlin1","merlin2", "eiger1"]
scalars_list = ["None", "sclr1_ch1","sclr1_ch2","sclr1_ch3","sclr1_ch4","sclr1_ch5"]

def load_stylesheet(path):
    with open(path, "r") as file:
        stylesheet = file.read()
    return stylesheet


def remove_nan_inf(im):
    im = np.array(im)
    im[np.isnan(im)] = 0
    im[np.isinf(im)] = 0
    return im


def remove_hot_pixels(image_array, NSigma=3):
    image_array = remove_nan_inf(image_array)
    image_array[abs(image_array) > np.std(image_array) * NSigma] = 0
    return image_array


def show_error_message_box(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(None, "Error", error_message)
            pass
    return wrapper

def extract_detector_name(filename, detector_list, fuzzy_cutoff=0.6):
    """
    Robustly extract a detector name from a filename by:
    1) Checking for exact matches
    2) Using fuzzy matching for any tokens (and position) in the filename
    
    Returns the exact detector name (correct case) or None.
    """
    # 1) Get the base name (no dirs, no extension)
    base = os.path.splitext(os.path.basename(filename))[0]
    
    # 2) Tokenize the base name on non-alphanumeric characters
    tokens = re.findall(r"[A-Za-z0-9]+", base)
    if not tokens:
        return None
    
    # Precompute lowercase mapping for exact matching
    det_lower_map = {d.lower(): d for d in detector_list}
    det_lowers = list(det_lower_map.keys())
    
    # 3) Exact match for the detector names anywhere in the filename
    for tok in tokens:
        tok_l = tok.lower()
        if tok_l in det_lower_map:
            return det_lower_map[tok_l]
    
    # 4) Fuzzy match any token in the filename against the known detector names
    for tok in tokens:
        m = get_close_matches(tok.lower(), det_lowers, n=1, cutoff=fuzzy_cutoff)
        if m:
            return det_lower_map[m[0]]
    
    # 5) Check if any detector name is a substring of the base (in any order)
    base_l = base.lower()
    for det in detector_list:
        if det.lower() in base_l:
            return det
    
    return None
    
class EmittingStream(QObject):

    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))


class DiffViewWindow(QtWidgets.QMainWindow):

    def __init__(self):
        super(DiffViewWindow, self).__init__()
        uic.loadUi(os.path.join(ui_path,'dpc_view.ui'), self)
        print("ui loaded")
        
        #sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        #sys.stderr = EmittingStream(textWritten=self.errorOutputWritten)

        self.prev_config = {} # TODO, record the workflow later
        self.wd = None
        self.diff_img = None
        self.single_diff = None
        self.diff_stack = None
        self.roi = None
        self.cropped_stack = None

        #beamline specific paramaters
        self.cb_norm_scalars.addItems(scalars_list)
        self.cb_det_list.addItems(detector_list)
        self.cb_det_list.setCurrentIndex(0)
        self.cb_norm_scalars.setCurrentIndex(4)

        num_comma_validator = QtGui.QRegularExpressionValidator(QtCore.QRegularExpression("[0-9,]*"))
        self.cb_solvers.addItems(SOLVERS)

        
        #self.display_diff_img_from_h5() #testing only
        #connections
        self.pb_select_wd.clicked.connect(self.choose_wd)
        self.pb_load_from_h5.clicked.connect(lambda:self.load_and_display_diff_data(
            self.sb_ref_img_num.value()))
        self.pb_load_data_from_db.clicked.connect(lambda:self.load_and_display_diff_data(
            self.sb_ref_img_num.value(), from_h5=False))
        self.diff_im_view.scene().sigMouseClicked.connect(self.on_mouse_doubleclick)
        self.pb_plot_mask.clicked.connect(self.plot_mask)
        self.pb_apply_mask.clicked.connect(self.apply_mask)
        self.pb_apply_roi.clicked.connect(self.get_masked_cropped_data)
        self.pb_recon_dpc.clicked.connect(self._recon_dpc)
    
    def __del__(self):
        import sys
        # Restore sys.stdout
        sys.stdout = sys.__stdout__


    def normalOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.pte_status.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.pte_status.setTextCursor(cursor)
        self.pte_status.ensureCursorVisible()


    def errorOutputWritten(self, text):
        """Append text to the QTextEdit."""
        # Maybe QTextEdit.append() works as well, but this is how I do it:
        cursor = self.pte_status.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.pte_status.setTextCursor(cursor)
        self.pte_status.ensureCursorVisible()
    
    def choose_wd(self):
        """updates the line edit for working directory"""
        self.wd = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder')
        self.le_wd.setText((str(self.wd)))

    def create_load_params(self):

        self.load_params = {"wd":self.le_wd.text(),
                            "sid":int(self.le_sid.text()), 
                            "threshold":(self.sb_low_threshold.value(),self.sb_high_threshold.value()),
                            "mon":self.cb_norm_scalars.currentText(),
                            "det":self.cb_det_list.currentText(),
                            "roi":None,
                            "mask":None,
                            }
        
        if self.load_params['mon'] == 'None':
            self.load_params['mon'] = None

    def load_im_stack_from_db(self):
        self.create_load_params()
        self.det = self.load_params["det"]
        # export_single_detector_h5 now takes `det=` (not `dets=`) and returns a flat dict
        self.all_data_dict = export_single_detector_h5(
            self.load_params["sid"],
            det=self.det,
            wd=self.load_params["wd"],
            mon=self.load_params["mon"],
            compression=None,
            save_and_return=True
        )[0]


    def load_im_stack_from_h5(self):
        self.create_load_params()
        sid = self.load_params["sid"]

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select HDF5 File",
            self.load_params["wd"],
            "HDF5 Files (*.h5 *.hdf5);;All Files (*)"
        )

        # decide which detector to use
        if getattr(self, "det", None) is None:
            self.det = self.load_params["det"]
        else:
            self.load_params["det"] = self.det
            self.cb_det_list.setCurrentText(self.det)
        print(f"{self.det=}")

        if filename:
            print(f"Loading {os.path.basename(filename)}, please wait…")
            # unpack_single_detector_h5 uses `det_name=` now
            self.all_data_dict = unpack_single_detector_h5(
                filename,
                det_name=self.det
            )
        else:
            raise FileNotFoundError(
                f"An HDF5 for scan {sid} not found; "
                f"expected scan_{sid}_{self.load_params['det']}.h5"
            )


    def get_diff_data(self):
        # flat keys now: "det" and "Io", no nested diff_data dict
        self.diff_stack = self.all_data_dict["det"]
        self.Io = self.all_data_dict["Io"]

        # shape is (dim1, dim2, roi_y, roi_x)
        self.im_y, self.im_x, self.roi_y, self.roi_x = self.diff_stack.shape

        # flatten back into (n_steps, roi_y, roi_x)
        self.diff_stack = self.diff_stack.reshape(-1, self.roi_y, self.roi_x)

    
    def _inject_dict(self, d: dict, prefix: str = ""):
        """
        Recursively turn nested dict into flat attributes:
          {'scan':{'detector_distance':2.0}, 'energy':7.1}
        → self.scan_detector_distance, self.energy
        """
        for key, val in d.items():
            attr = f"{prefix}_{key}" if prefix else key
            # if isinstance(val, dict):
            #     self._inject_dict(val, prefix=attr)
            # else:
            setattr(self, attr, val)


    def get_and_fill_scan_params(self):
        # pull from the new top‐level "scan_params" key
        scan_params = self.all_data_dict["scan_params"]
        # inject only the flat keys—nested dicts (like 'scan') stay as dicts
        self._inject_dict(scan_params)

        # energy is now self.energy
        self.dsb_energy.setValue(self.energy)
        # detector distance lives under the nested scan dict
        self.dsb_det_dist.setValue(self.scan["detector_distance"])

        # scan_input also lives under self.scan
        x_num, y_num = self.scan["scan_input"][2], self.scan["scan_input"][5]
        x0, x1 = self.scan["scan_input"][0], self.scan["scan_input"][1]
        y0, y1 = self.scan["scan_input"][3], self.scan["scan_input"][4]

        x_step = round((x1 - x0) / x_num, 2)
        y_step = round((y1 - y0) / y_num, 2)

        self.dsb_x_step.setValue(x_step)
        self.dsb_y_step.setValue(y_step)
        self.sb_x_num.setValue(int(x_num))
        self.sb_y_num.setValue(int(y_num))

    def display_diff_data(self, im_index = 0):
        #self.load_im_stack_from_h5()
        #GUI widegt limits
        self.sb_ref_img_num.setMaximum(int(self.diff_stack.shape[0]))
        
        # Show axes: these are auto-labeled as pixel numbers
        # self.diff_im_view.invertY(True)
        self.diff_im_view.setLabel('left', 'Y Pixels')
        self.diff_im_view.setLabel('bottom', 'X Pixels')
        
        self.display_data = self.diff_stack[im_index,:,:]
        self.img_item = pg.ImageItem()
        self.img_item.setImage(self.display_data)
        lut = pg.colormap.get('viridis')  # You can also use: 'inferno', 'plasma', 'cividis', etc.
        self.img_item.setColorMap(lut)
        self.diff_im_view.addItem(self.img_item)
        
        #Add ROI
        self.create_roi()
        if not self.roi in self.diff_im_view.items():
            self.diff_im_view.addItem(self.roi)
        # Optional: connect to get ROI updates
        self.roi.sigRegionChangeFinished.connect(self.get_roi_info)


        #masking pixels
        self.mask = np.ones_like(self.display_data, dtype=bool)
        # Transparent overlay for mask
        self.mask_overlay = pg.ImageItem()
        self.mask_overlay.setZValue(10)
        self.mask_overlay.setOpts(opacity=0.4, lut=self._make_mask_lut())
        self.diff_im_view.addItem(self.mask_overlay)
        self.update_mask_overlay()

    def load_and_display_diff_data(self, im_index = 0, from_h5 = True):
        if from_h5:
            self.load_im_stack_from_h5()
        else:
            self.load_im_stack_from_db()
        self.get_diff_data()
        self.display_diff_data(im_index)
        self.get_and_fill_scan_params()


    def create_roi(self):

        if self.display_data is None:
            print("No image loaded.")
            return

        height, width = self.display_data.shape
        roi_width = width / 2
        roi_height = height / 2

        total = np.sum(self.display_data)
        if total == 0:
            print("Image is all zeros — defaulting to image center.")
            cx, cy = width / 2, height / 2
        else:
            cy, cx = center_of_mass(self.display_data)
            if not (0 <= cx < width and 0 <= cy < height):
                print("Center of mass out of bounds — using image center.")
                cx, cy = width / 2, height / 2

        self.roi = pg.RectROI([cx - roi_width / 2, cy - roi_height / 2],
                    [roi_width, roi_height],
                    pen='r',
                    maxBounds = QtCore.QRectF(0, 0, width, height))
        
        
    def _make_mask_lut(self):
        """LUT: 1 → red; 0 → transparent."""
        lut = np.zeros((2, 4), dtype=np.ubyte)
        lut[1] = [0, 0, 0, 0]        # Transparent
        lut[0] = [255, 0, 0, 255]    # Red with alpha
        return lut

    def update_mask_overlay(self):
        self.mask_overlay.setImage(self.mask, autoLevels=False)

    def on_mouse_doubleclick(self, event):
        if  event.double():
            pos = event.scenePos()
            mouse_point = self.diff_im_view.plotItem.vb.mapSceneToView(pos)
            x, y = int(mouse_point.x()), int(mouse_point.y())

            if 0 <= x < self.display_data.shape[1] and 0 <= y < self.display_data.shape[0]:
                # Toggle pixel: 1 ↔ 0
                self.mask[y, x] = 1 - self.mask[y, x]
                print(f"{'Masked' if self.mask[y,x] == 0 else 'Unmasked'} pixel: ({x}, {y})")
                self.update_mask_overlay()

    def apply_mask(self):
        print("plotting mask applied img")
        masked = self.display_data * self.mask
        # Open a new window to show masked result
        self.win_masked = pg.ImageView()
        self.win_masked.setImage(masked)
        self.win_masked.getView().invertY(False)
        self.win_masked.setWindowTitle("Mask")
        self.win_masked.setPredefinedGradient("viridis")
        self.win_masked.show()

    def plot_mask(self):
        # Open a new window to show masked result
        print("plotting mask")
        self.win_mask = pg.ImageView()
        self.win_mask.setImage(self.mask)
        self.win_mask.getView().invertY(False)
        self.win_mask.setWindowTitle("Mask")
        self.win_mask.setPredefinedGradient("bipolar")
        self.win_mask.show()

    def get_roi_info(self):
        pos = self.roi.pos()
        size = self.roi.size()
        print(f"ROI Position: {pos}, Size: {size}")
        return pos,size
    
    def get_masked_cropped_data(self):
        
        masked = self.diff_stack*self.mask[np.newaxis, :,:]
        self.cropped_stack = self.roi.getArrayRegion(
            masked,
            self.img_item,
            axes=(1, 2),
            returnMappedCoords=False,
            order=0)
        # print(cropped.max()), print(cropped.dtype)
        # print(self.diff_stack.max()), print(self.diff_stack.dtype)
        self.win_cropped = pg.ImageView()
        self.win_cropped.setImage(self.cropped_stack[0:100])
        self.win_cropped.setWindowTitle("Croppped and masked, first 100")
        self.win_cropped.getView().invertY(False)
        self.win_cropped.setPredefinedGradient("viridis")
        self.win_cropped.show()


        #self.diff_stack_clean = self.diff_stack*self.mask[:, np.newaxis, np.newaxis]

    def clear_all_masked_pixels(self):
        pass

    def find_and_mask_hot_pixels(self):
        pass

    def _recon_dpc(self):

        ref_img = self.sb_ref_img_num.value()
        max_iter = self.sb_max_iter.value()
        solver = self.cb_solvers.currentText()
        reverse_gy = 1
        if self.cb_reverse_gy.isChecked():
            reverse_gy = -1

        reverse_gx = 1
        if self.cb_reverse_gx.isChecked():
            reverse_gx = -1

        energy = self.dsb_energy.value()
        det_pixel = self.dsb_det_pixel_size.value()
        det_dist = self.dsb_det_dist.value()
        dxy = [self.dsb_x_step.value(),self.dsb_y_step.value()]
        num_xy = [self.sb_y_num.value(),self.sb_x_num.value()]
        
        a_, gx_, gy_, phi = recon_dpc_from_im_stack(self.cropped_stack, 
                                                    ref_image_num=ref_img, 
                                                    start_point=[1, 0], 
                                                    max_iter=max_iter, 
                                                    solver=solver, 
                                                    reverse_x=reverse_gx, 
                                                    reverse_y=reverse_gy,
                                                    energy = energy, 
                                                    det_pixel = det_pixel, 
                                                    det_dist = det_dist,
                                                    dxy = dxy,
                                                    num_xy =num_xy)
        
        self.gx_im_view.setImage(gx_)
        self.gx_im_view.view.register("Gradient_x")
        self.gy_im_view.setImage(gy_)
        self.gy_im_view.setWindowTitle("Gradient_y")
        self.amp_im_view.setImage(a_)
        self.amp_im_view.setWindowTitle("Gradient_Amplitude")
        self.phase_im_view.setImage(phi)
        self.phase_im_view.setWindowTitle("Phase")

    '''
    def load_and_save_from_db(self):
        
        self.create_load_params()
        real_sid = db[int(self.load_params['sid'])].start["scan_id"]
        self.load_params['sid'] = real_sid
        print(self.load_params)
        print(f"Loading {self.load_params['sid']} please wait...this may take a while...")
        
        
        QtTest.QTest.qWait(1000)
        #saves data to a default folder with sid name      
        export_single_detector_h5(int(self.load_params['sid']),
                        det=self.load_params['det'],
                        wd= self.load_params['wd'],
                        mon = self.load_params['mon']
                        )   
        self.load_from_local_and_display() #looks for the filename matching with sid
        #TODO add assertions and exceptions, thread it


    def load_from_db(self):
        
        self.create_load_params()
        real_sid = db[int(self.load_params['sid'])].start["scan_id"]
        self.load_params['sid'] = real_sid
        print(self.load_params)
        print(f"Loading {self.load_params['sid']} please wait...this may take a while...")
        export_single_detector_h5(int(self.load_params['sid']),
                               self.load_params['det'],
                               )
        # diff_array = return_diff_array(int(self.load_params['sid']), 
        #                              det=self.load_params['det'], 
        #                              mon=self.load_params['mon'], 
        #                              threshold=self.load_params['threshold'])
        
        self.display_diff_sum_img()
        QtTest.QTest.qWait(1000)
        self.display_xrf_img()

    def load_from_local_and_display(self):
        #self.create_load_params()
        # self.display_param["diff_wd"] = os.path.join(os.path.join(self.load_params["wd"],f"{self.load_params['sid']}_diff_data"),
        #                                         f"{self.load_params['sid']}_diff_{self.load_params['det']}.tiff")

        self.display_param["diff_wd"] = os.path.join(self.load_params["wd"],
                                                     f"scan_{self.load_params['sid']}_{self.load_params['det']}.h5")
        
        self.display_diff_sum_img()
        QtTest.QTest.qWait(1000)
        self.display_xrf_img()

    def choose_diff_file(self):

        """updates the line edit for working directory"""

        filename_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select DiffFile')
        if filename_[0]:
            self.diff_file = filename_[0]
            self.display_param["diff_wd"] = self.diff_file
            print(f"Loading {filename_[0]} please wait...\n this may take a while...")
            self.display_diff_sum_img()
            QtTest.QTest.qWait(1000)
            self.display_xrf_img()
            print("Done")
        else:
            pass

    def choose_xrf_file(self):

        filename_ = QtWidgets.QFileDialog.getOpenFileName(self, 'Select XRF File')
        if filename_[0]:
            self.xrf_file = filename_[0]
            self.display_param["xrf_wd"] = self.xrf_file
            self.display_xrf_img()
            
        else:
            pass


    def create_pointer(self):
        # Use ScatterPlotItem to draw points
        self.scatterItem = pg.ScatterPlotItem(
            size=10, 
            pen=pg.mkPen(None), 
            brush=pg.mkBrush(255, 0, 0),
            hoverable=True,
            hoverBrush=pg.mkBrush(0, 255, 255)
        )
        self.scatterItem.setZValue(2) # Ensure scatterPlotItem is always at top
            

    def display_xrf_img(self, num=-1):

        try:
            self.xrf_plot_canvas.clear()
            self.create_pointer()
        except:
            pass

        im_array = self.xrf_img
        z, ysize,xsize = np.shape(self.xrf_stack)
        self.statusbar.showMessage(f"Image Shape = {np.shape(self.xrf_stack)}")
        # A plot area (ViewBox + axes) for displaying the image

            

        self.p1_xrf = self.xrf_plot_canvas.addPlot(title= "xrf_Image")
        #self.p1_xrf.setAspectLocked(True)
        self.p1_xrf.getViewBox().invertY(True)
        self.p1_xrf.getViewBox().setLimits(xMin = 0,
                                    xMax = xsize,
                                    yMin = 0,
                                    yMax = ysize
                                    )
            
        self.p1_xrf.addItem(self.scatterItem)

        # Item for displaying image data
        #self.img_item_xrf = pg.ImageItem(axisOrder = 'row-major')
        self.img_item_xrf = pg.ImageItem()
        if self.display_param["xrf_img_settings"]["display_log"]:
            self.xrf_stack = np.nan_to_num(np.log10(self.xrf_stack), nan=np.nan, posinf=np.nan, neginf=np.nan)
        self.img_item_xrf.setImage(self.xrf_stack[int(num)], opacity=1)
        self.p1_xrf.addItem(self.img_item_xrf)
    
        self.hist_xrf = pg.HistogramLUTItem()
        color_map_xrf = pg.colormap.get(self.display_param["xrf_img_settings"]["lut"])
        self.hist_xrf.gradient.setColorMap(color_map_xrf)
        self.hist_xrf.setImageItem(self.img_item_xrf)
        if self.display_param["xrf_img_settings"]["hist_lim"][0] == None or self.display_param["xrf_img_settings"]["hist_lim"][1] == None:
            self.hist_xrf.autoHistogramRange = False
        else:
            self.hist_xrf.autoHistogramRange = False
            self.hist_xrf.setLevels(min=self.display_param["xrf_img_settings"]["hist_lim"][0], 
                                max=self.display_param["xrf_img_settings"]["hist_lim"][1])
        self.xrf_plot_canvas.addItem(self.hist_xrf)
        # self.img_item.hoverEvent = self.imageHoverEvent
        self.img_item_xrf.mousePressEvent = self.MouseClickEvent_xrf
        self.img_item_xrf.hoverEvent = self.imageHoverEvent_xrf
        self.roi_exists = False
        # self.roi_state = None

    def display_diff_sum_img(self):
        if not self.display_param["diff_wd"] == None:
            #TODO, may have memory issues
            tiffs = (".tiff", ".tif")
            if self.display_param["diff_wd"].endswith(tiffs):
                self.diff_stack = tf.imread(self.display_param["diff_wd"])

            else:
                (
                self.diff_stack,
                self.Io,
                self.scan_pos,
                self.xrf_stack,
                self.xrf_elem_list
                ) = unpack_single_detector_h5(
                self.display_param["diff_wd"],
                self.load_params["det"]
                )


            name_ = os.path.basename(self.display_param["diff_wd"])

            self.cb_xrf_elem_list.addItems(self.xrf_elem_list)

            if self.diff_stack.ndim != 4:
                raise ValueError(f"{np.shape(self.diff_stack)}; only works for data shape with (im1,im2,det1,det2) structure")
            # print(np.shape(self.diff_stack))

            self.diff_sum_img = np.nansum(self.diff_stack, axis = (-2,-1))#memory efficient?
            if not self.roi is None:
                self.diff_im_view.removeItem(self.roi)
                self.roi = None
            try:
                self.diff_sum_plot_canvas.clear()
                self.diff_im_view.clear()
                self.create_pointer()
                
            except:
                pass
            
            
            im_array = self.diff_sum_img 
            ysize,xsize = np.shape(im_array)
            self.statusbar.showMessage(f"Image Shape = {np.shape(im_array)}")
            # A plot area (ViewBox + axes) for displaying the image

            self.p1_diff_sum = self.diff_sum_plot_canvas.addPlot(title= "Diff_Sum_Image")
            #self.p1_diff_sum.setAspectLocked(True)
            self.p1_diff_sum.getViewBox().invertY(True)
            self.p1_diff_sum.getViewBox().setLimits(xMin = 0,
                                        xMax = xsize,
                                        yMin = 0,
                                        yMax = ysize
                                        )
            
            self.p1_diff_sum.addItem(self.scatterItem)

            # Item for displaying image data
            #self.img_item_diff_sum = pg.ImageItem(axisOrder = 'row-major')
            self.img_item_diff_sum = pg.ImageItem()
            if self.display_param["diff_sum_img_settings"]["display_log"]:
                im_array = np.nan_to_num(np.log10(im_array), nan=np.nan, posinf=np.nan, neginf=np.nan)
            self.img_item_diff_sum.setImage(im_array, opacity=1)
            self.p1_diff_sum.addItem(self.img_item_diff_sum)
        
            self.hist_diff_sum = pg.HistogramLUTItem()
            color_map_diff_sum = pg.colormap.get(self.display_param["diff_sum_img_settings"]["lut"])
            self.hist_diff_sum.gradient.setColorMap(color_map_diff_sum)
            self.hist_diff_sum.setImageItem(self.img_item_diff_sum)
            if self.display_param["diff_sum_img_settings"]["hist_lim"][0] == None or self.display_param["diff_sum_img_settings"]["hist_lim"][1] == None:
                self.hist_diff_sum.autoHistogramRange = False
            else:
                self.hist_diff_sum.autoHistogramRange = False
                self.hist_diff_sum.setLevels(min=self.display_param["diff_sum_img_settings"]["hist_lim"][0], 
                                    max=self.display_param["diff_sum_img_settings"]["hist_lim"][1])
            self.diff_sum_plot_canvas.addItem(self.hist_diff_sum)
            self.diff_im_view.hoverEvent = self.imageHoverEvent_diff
            self.img_item_diff_sum.mousePressEvent = self.MouseClickEvent_diff_sum
            self.img_item_diff_sum.hoverEvent = self.imageHoverEvent_diff_sum
            self.roi_exists = False

    def MouseClickEvent_xrf(self, event = QtCore.QEvent):

        if event.type() == QtCore.QEvent.GraphicsSceneMouseDoubleClick:
            if event.button() == QtCore.Qt.LeftButton:
                self.points = []
                pos = self.img_item_xrf.mapToParent(event.pos())
                i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
                self.points.append([i, j])
                self.scatterItem.setData(pos=self.points)
                self.single_diff = self.diff_stack[j,i, :,:]
                self.display_param["diff_img_settings"]["display_log"] = self.cb_diff_log_scale.isChecked()

                if self.display_param["diff_img_settings"]["display_log"]:
                    self.single_diff = np.nan_to_num(np.log10(self.single_diff), nan=np.nan, posinf=np.nan, neginf=np.nan)
                
                if self.display_param["diff_img_settings"]["hist_lim"] == (None,None):
                    self.diff_im_view.setImage(self.single_diff)
                else:
                    self.diff_im_view.setImage(self.single_diff,autoLevels = False,autoHistogramRange=True)
                    levels = self.display_param["diff_img_settings"]["hist_lim"]
                    self.diff_im_view.setLevels(levels[0], levels[1])
                if self.roi == None:
                    self.create_roi(self.single_diff.shape)
                self.diff_im_view.addItem(self.roi)
                self.diff_im_view.setPredefinedGradient(self.display_param["diff_img_settings"]["lut"] )


            else: event.ignore()
        else: event.ignore()

    def MouseClickEvent_diff_sum(self, event = QtCore.QEvent):

        if event.type() == QtCore.QEvent.GraphicsSceneMouseDoubleClick:
            if event.button() == QtCore.Qt.LeftButton:
                self.points = []
                pos = self.img_item_diff_sum.mapToParent(event.pos())
                i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
                self.points.append([i, j])
                self.scatterItem.setData(pos=self.points)
                self.single_diff = self.diff_stack[j,i, :,:]
                self.display_param["diff_img_settings"]["display_log"] = self.cb_diff_log_scale.isChecked()

                if self.display_param["diff_img_settings"]["display_log"]:
                    self.single_diff = np.nan_to_num(np.log10(self.single_diff), nan=np.nan, posinf=np.nan, neginf=np.nan)

                
                if self.display_param["diff_img_settings"]["hist_lim"] == (None,None):
                    self.diff_im_view.setImage(self.single_diff)
                else:
                    self.diff_im_view.setImage(self.single_diff,autoLevels = False,autoHistogramRange=True)
                    levels = self.display_param["diff_img_settings"]["hist_lim"]
                    self.diff_im_view.setLevels(levels[0], levels[1])
                if self.roi == None:
                    self.create_roi(self.single_diff.shape)
                self.diff_im_view.addItem(self.roi)
                self.diff_im_view.setPredefinedGradient(self.display_param["diff_img_settings"]["lut"] )

            else: event.ignore()
        else: event.ignore()

    def imageHoverEvent_xrf(self, event = QtCore.QEvent):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isEnter():
            pos = self.img_item_xrf.mapToParent(event.pos())
            i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
            # Set bounds for clipping

            #TODO does not work the hover event si,np.clip to avoid hovering outside
            val = self.xrf_img[j, i]
            #print(val)
            self.statusbar.showMessage(f'pixel: {i, j} , {val = }')

    def imageHoverEvent_diff(self, event = QtCore.QEvent):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isEnter():
            pos = self.img_item.mapToParent(event.pos())
            i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
            #TODO does not work the hover event si
            val = self.single_diff[j, i]
            #print(val)
            self.statusbar.showMessage(f'pixel: {i, j} , {val = }')

    def imageHoverEvent_diff_sum(self, event = QtCore.QEvent):
        """Show the position, pixel, and value under the mouse cursor.
        """
        if event.isEnter():
            pos = self.img_item_diff_sum.mapToParent(event.pos())
            i, j = int(np.floor(pos.x())), int(np.floor(pos.y()))
            #TODO does not work the hover event si
            val = self.diff_sum_img[j, i]
            #print(val)
            self.statusbar.showMessage(f'pixel: {i, j} , {val = }')
    
    def toggle_hist_scale_diff(self, auto = False):        
        
        if auto:
            self.display_param["diff_img_settings"]["hist_lim"] = (None,None)

        else:
            hist_min, hist_max = self.diff_im_view.getLevels()
            self.display_param["diff_img_settings"]["hist_lim"] = (hist_min, hist_max)
            self.statusbar.showMessage(f"Histogram level set to [{hist_min :.4f}, {hist_max :.4f}]")
    
    def create_roi(self, im_array_dim):
        # if self.roi_state !=None:
        #     self.roi.setState(self.roi_state)   
        sz = np.ceil(im_array_dim[0]*0.2)
        roi_x = im_array_dim[1] 
        roi_y = im_array_dim[0] 
        self.roi =pg.PolyLineROI(
                                [[0, 0], [0, sz], [sz, sz], [sz, 0]],
                                pos=(int(roi_x // 2), int(roi_y // 2)),
                                maxBounds=QtCore.QRectF(0, 0, im_array_dim[1], im_array_dim[0]),
                                pen=pg.mkPen("r", width=1), 
                                hoverPen=pg.mkPen("w", width=1),
                                handlePen = pg.mkPen("m", width=3, ),
                                closed=True,
                                removable=True,
                                snapSize = 1,
                                translateSnap = True
                                )
        self.roi.setZValue(10)
        

    def get_mask_from_roi(self):

        # get the roi region:QPaintPathObject
        roiShape = self.roi.mapToItem(self.diff_im_view.getImageItem(), self.roi.shape())
        
        grid_shape = np.shape(self.single_diff)

        # get data in the scatter plot
        scatterData = np.meshgrid(np.arange(grid_shape[1]), np.arange(grid_shape[0]))
        scatterData = np.reshape(scatterData,(2, grid_shape[0]*grid_shape[1]))

        #xprint(f"{np.shape(scatterData) = }")

        # generate a binary mask for points inside or outside the roishape
        selected = [roiShape.contains(QtCore.QPointF(pt[0], pt[1])) for pt in scatterData.T]

        #print(f"{np.shape(selected) = }")

        # # reshape the mask to image dimensions
        self.mask2D = np.reshape(selected, (self.single_diff.shape))

        # # get masked image1
        # self.maskedImage = self.mask2D * self.single_diff
        print(f"{np.shape(self.single_diff) = }")
        print(f"{self.mask2D.shape}")

        plot1 = pg.image(self.mask2D)
        plot1.setPredefinedGradient("bipolar")
        # plot2 = pg.image(self.single_diff*self.mask2D)
        # plot2.setPredefinedGradient("bipolar")

        masked_diff_sum, masked_diff_img = self.apply_mask_to_diff_stack(self.diff_stack,self.mask2D)
        plot3 = pg.image(masked_diff_sum)
        plot3.setPredefinedGradient("viridis")
        
        plot4 = pg.image(masked_diff_img)
        plot4.setPredefinedGradient("viridis")

    def apply_mask_to_diff_stack(self,diff_data_4d, mask):

        masked_stack = diff_data_4d*mask[np.newaxis,np.newaxis,:,:]

        self.masked_diff_sum, self.masked_diff_img = np.sum(masked_stack,axis = (-1,-2)), np.sum(masked_stack,axis = (0,1))
        return self.masked_diff_sum,self.masked_diff_img
    
    def save_mask_data(self):
        """updates the line edit for working directory"""
        self.save_folder = QtWidgets.QFileDialog.getExistingDirectory(self, 'Select Folder', self.wd)
        tf.imwrite(os.path.join(self.save_folder,"_masked_diff_sum.tiff"),  self.masked_diff_img)
        tf.imwrite(os.path.join(self.save_folder,"_mask.tiff"),  self.mask2D)

    '''

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet(os.path.join(ui_path,"style_sheet.css")))
    font = QtGui.QFont("Arial", 10)
    app.setFont(font)   
    w = DiffViewWindow()
    w.show()
    sys.exit(app.exec())