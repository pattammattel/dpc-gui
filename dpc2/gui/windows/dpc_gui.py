
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
# from qtpy import QtWidgets, uic, QtCore, QtGui, QtTest
# from PyQt6.QtWidgets import QMessageBox
# from PyQt6.QtCore import QObject, pyqtSignal
from qtpy import QtWidgets, uic, QtCore, QtGui, QtTest
from qtpy.QtWidgets import QMessageBox, QProgressDialog
from qtpy.QtCore import QObject, QThread, pyqtSignal,Qt
pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
warnings.filterwarnings('ignore', category=RuntimeWarning)
from dpc2.utils.dpc_fileio import *
from dpc2.utils.dpc_kernel2 import *
from dpc2.utils.image_utils import *
from dpc2.gui import UI_DIR, DETECTOR_DATA_KEY_MAP
from dpc2.utils.helpers import with_busy_popup

#beamline specific
detector_list = ["eiger2","merlin1","merlin2", "eiger1"]
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
        uic.loadUi(os.path.join(UI_DIR,'dpc_view.ui'), self)
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
                            "det":DETECTOR_DATA_KEY_MAP[self.cb_det_list.currentText()],
                            "roi":None,
                            "mask":None,
                            }
        
        if self.load_params['mon'] == 'None':
            self.load_params['mon'] = None
    
    @with_busy_popup(
    title="Loading",
    message="Data loading in progress. Please wait...",
    success_msg="Data loaded successfully!",
    error_msg="Data loading failed.")

    def load_im_stack_from_db(self):
        self.create_load_params()
        self.det = self.load_params["det"]
        # export_single_detector_h5 now takes `det=` (not `dets=`) and returns a flat dict
        self.all_data_dict = export_diff_data_as_h5_single(
            self.load_params["sid"],
            det=self.det,
            wd=self.load_params["wd"],
            mon=self.load_params["mon"],
            compression=None,
            save_and_return=True
        )


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
        self.diff_stack = self.all_data_dict.get("det_images")
        self.Io = self.all_data_dict.get("Io")

        if self.diff_stack is None:
            raise ValueError("Missing 'det_images' in loaded data.")

    
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

    def display_diff_data(self, im_index=0):
        # If it's a lazy loader (function), call it and get the dataset object (not full array)
        if callable(self.diff_stack):
            dataset = self.diff_stack()  # e.g., h5py.Dataset
            self.sb_ref_img_num.setMaximum(int(dataset.shape[0]) - 1)
            self.display_data = dataset[im_index, :, :]  # Load only one frame
        else:
            # Eager-loaded NumPy array
            self.sb_ref_img_num.setMaximum(int(self.diff_stack.shape[0]) - 1)
            self.display_data = self.diff_stack[im_index, :, :]

        # Display the image
        self.img_item = pg.ImageItem()
        self.img_item.setImage(self.display_data)
        lut = pg.colormap.get('viridis')
        self.img_item.setColorMap(lut)
        self.diff_im_view.addItem(self.img_item)

        # Remove existing ROI if present
        if hasattr(self, "roi") and self.roi in self.diff_im_view.items():
            self.diff_im_view.removeItem(self.roi)

        # Add ROI
        self.create_roi()
        if not self.roi in self.diff_im_view.items():
            self.diff_im_view.addItem(self.roi)
        self.roi.sigRegionChangeFinished.connect(self.get_roi_info)

        # Mask overlay
        self.mask = np.ones_like(self.display_data, dtype=bool)
        self.mask_overlay = pg.ImageItem()
        self.mask_overlay.setZValue(10)
        self.mask_overlay.setOpts(opacity=0.4, lut=self._make_mask_lut())
        self.diff_im_view.addItem(self.mask_overlay)
        self.update_mask_overlay()



    def load_and_display_diff_data(self, im_index = 0, from_h5 = False):
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
    
    def get_masked_cropped_data_(self):
        # Apply mask to the currently displayed image
        masked_image = self.display_data * self.mask

        # Use the ROI to extract the region from the masked image
        cropped = self.roi.getArrayRegion(
            masked_image,
            self.img_item,
            returnMappedCoords=False,
            order=0
        )

        # Display the cropped image
        self.win_cropped = pg.ImageView()
        self.win_cropped.setImage(cropped)
        self.win_cropped.setWindowTitle("Cropped and Masked Image")
        self.win_cropped.getView().invertY(False)
        self.win_cropped.setPredefinedGradient("viridis")
        self.win_cropped.show()

        # Optionally store for reuse
        self.cropped_stack = cropped


    def get_masked_cropped_data(self, plot_after = True):
        """Extracts ROI slice and cropped mask for use in memory-efficient lazy DPC, and displays a preview."""

        # Get the slice from the ROI (Y, X slices)
        roi_slice, _ = self.roi.getArraySlice(self.display_data, self.img_item, returnSlice=True)
        self.roi_slice = roi_slice  # Store for use in DPC lazy processing

        # Crop the mask to the ROI region
        self.mask_roi = self.mask[roi_slice]  # Mask shape should match ROI'd image

        # Apply mask to the current displayed image (single frame)
        masked_image = self.display_data * self.mask

        # Crop the current image using the same slice
        self.cropped_stack = masked_image[roi_slice]
        # Store cropped frame if needed (preview only)

        if plot_after:

            # Display the cropped image
            self.win_cropped = pg.ImageView()
            self.win_cropped.setImage(self.cropped_stack)
            self.win_cropped.setWindowTitle("Cropped and Masked Image")
            self.win_cropped.getView().invertY(False)
            self.win_cropped.setPredefinedGradient("viridis")
            self.win_cropped.show()




    def clear_all_masked_pixels(self):
        pass

    def find_and_mask_hot_pixels(self):
        pass

    def _recon_dpc_(self):
        # GUI parameters
        ref_img = self.sb_ref_img_num.value()
        max_iter = self.sb_max_iter.value()
        solver = self.cb_solvers.currentText()
        reverse_gy = -1 if self.cb_reverse_gy.isChecked() else 1
        reverse_gx = -1 if self.cb_reverse_gx.isChecked() else 1
        energy = self.dsb_energy.value()
        det_pixel = self.dsb_det_pixel_size.value()
        det_dist = self.dsb_det_dist.value()
        dxy = [self.dsb_x_step.value(), self.dsb_y_step.value()]
        num_xy = [self.sb_y_num.value(), self.sb_x_num.value()]

        # Determine if using lazy-loaded data
        is_lazy = callable(self.diff_stack)
        dataset = self.diff_stack if is_lazy else self.cropped_stack

        # Ensure ROI and mask are available if needed
        
        if not hasattr(self, "roi_slice") or not hasattr(self, "mask_roi"):
            print("[INFO] ROI or mask not initialized, calling get_masked_cropped_data()")
            self.get_masked_cropped_data(plot_after=False)
            roi_slice = self.roi_slice
            mask = self.mask_roi

        # Run reconstruction
        a_, gx_, gy_, phi = recon_dpc_from_im_stack(
            dataset,
            ref_image_num=ref_img,
            start_point=[1, 0],
            max_iter=max_iter,
            solver=solver,
            reverse_x=reverse_gx,
            reverse_y=reverse_gy,
            energy=energy,
            det_pixel=det_pixel,
            det_dist=det_dist,
            dxy=dxy,
            num_xy=num_xy,
            roi_slice=roi_slice,
            mask=mask
        )

        # Display results
        self.gx_im_view.setImage(gx_)
        self.gx_im_view.view.setWindowTitle("Gradient_x")
        self.gy_im_view.setImage(gy_)
        self.gy_im_view.setWindowTitle("Gradient_y")
        self.amp_im_view.setImage(a_)
        self.amp_im_view.setWindowTitle("Gradient_Amplitude")
        self.phase_im_view.setImage(phi)
        self.phase_im_view.setWindowTitle("Phase")

    def _recon_dpc(self):
        # Extract GUI params
        ref_img = self.sb_ref_img_num.value()
        max_iter = self.sb_max_iter.value()
        solver = self.cb_solvers.currentText()
        reverse_gy = -1 if self.cb_reverse_gy.isChecked() else 1
        reverse_gx = -1 if self.cb_reverse_gx.isChecked() else 1
        energy = self.dsb_energy.value()
        det_pixel = self.dsb_det_pixel_size.value()
        det_dist = self.dsb_det_dist.value()
        dxy = [self.dsb_x_step.value(), self.dsb_y_step.value()]
        num_xy = [self.sb_y_num.value(), self.sb_x_num.value()]

        is_lazy = callable(self.diff_stack)
        dataset = self.diff_stack if is_lazy else self.cropped_stack

        if not hasattr(self, "roi_slice") or not hasattr(self, "mask_roi"):
            print("[INFO] ROI or mask not initialized, calling get_masked_cropped_data()")
            self.get_masked_cropped_data(plot_after=False)

        roi_slice = getattr(self, "roi_slice", None)
        mask = getattr(self, "mask_roi", None)

        # Create and run thread
        self.dpc_thread = QThread()
        self.dpc_worker = DPCWorker(
            dataset, ref_img, [1, 0], max_iter, solver,
            reverse_gx, reverse_gy, energy, det_pixel, det_dist,
            dxy, num_xy, roi_slice, mask
        )
        self.dpc_worker.moveToThread(self.dpc_thread)

        # Connect signals
        self.dpc_thread.started.connect(self.dpc_worker.run)
        self.dpc_worker.finished.connect(self._handle_dpc_result)
        self.dpc_worker.error.connect(self._handle_dpc_error)
        self.dpc_worker.finished.connect(self.dpc_thread.quit)
        self.dpc_worker.finished.connect(self.dpc_worker.deleteLater)
        self.dpc_thread.finished.connect(self.dpc_thread.deleteLater)

        self.dpc_progress = QProgressDialog("Reconstructing DPC...", None, 0, 0, self)
        self.dpc_progress.setWindowTitle("DPC In Progress")
        self.dpc_progress.setWindowModality(Qt.ApplicationModal)
        self.dpc_progress.setCancelButton(None)
        self.dpc_progress.setMinimumDuration(0)  # show immediately
        self.dpc_progress.setValue(0)
        self.dpc_progress.show()


        self.dpc_thread.start()

    def _handle_dpc_result(self, a_, gx_, gy_, phi):
        if hasattr(self, "dpc_progress"):
            self.dpc_progress.close()

        self.gx_im_view.setImage(gx_)
        self.gx_im_view.view.setWindowTitle("Gradient_x")
        self.gy_im_view.setImage(gy_)
        self.gy_im_view.setWindowTitle("Gradient_y")
        self.amp_im_view.setImage(a_)
        self.amp_im_view.setWindowTitle("Gradient_Amplitude")
        self.phase_im_view.setImage(phi)
        self.phase_im_view.setWindowTitle("Phase")

    def _handle_dpc_error(self, e):
        if hasattr(self, "dpc_progress"):
            self.dpc_progress.close()

        QMessageBox.critical(self, "DPC Reconstruction Error", str(e))





class DPCWorker(QObject):
    finished = pyqtSignal(object, object, object, object)
    error = pyqtSignal(Exception)

    def __init__(self, dataset, ref_img, start_point, max_iter, solver,
                 reverse_x, reverse_y, energy, det_pixel, det_dist,
                 dxy, num_xy, roi_slice, mask):
        super().__init__()
        self.dataset = dataset
        self.ref_img = ref_img
        self.start_point = start_point
        self.max_iter = max_iter
        self.solver = solver
        self.reverse_x = reverse_x
        self.reverse_y = reverse_y
        self.energy = energy
        self.det_pixel = det_pixel
        self.det_dist = det_dist
        self.dxy = dxy
        self.num_xy = num_xy
        self.roi_slice = roi_slice
        self.mask = mask

    def run(self):
        try:
            a_, gx_, gy_, phi = recon_dpc_from_im_stack(
                self.dataset,
                ref_image_num=self.ref_img,
                start_point=self.start_point,
                max_iter=self.max_iter,
                solver=self.solver,
                reverse_x=self.reverse_x,
                reverse_y=self.reverse_y,
                energy=self.energy,
                det_pixel=self.det_pixel,
                det_dist=self.det_dist,
                dxy=self.dxy,
                num_xy=self.num_xy,
                roi_slice=self.roi_slice,
                mask=self.mask
            )
            self.finished.emit(a_, gx_, gy_, phi)
        except Exception as e:
            self.error.emit(e)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(load_stylesheet(os.path.join(ui_path,"style_sheet.css")))
    font = QtGui.QFont("Arial", 10)
    app.setFont(font)   
    w = DiffViewWindow()
    w.show()
    sys.exit(app.exec())