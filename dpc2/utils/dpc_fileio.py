
import os
import warnings
import h5py
import pandas as pd
import datetime
import warnings
import sys
import numpy as np
import shutil
import tifffile as tf
from tqdm import tqdm
try:    
    from hxntools.CompositeBroker import db
    from hxntools.scan_info import get_scan_positions
except:
    print("Offline analysis; No BL data available")
    pass
import csv
import getpass
from typing import List, Optional, Union

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

if  os.getlogin().startswith("xf03") or os.getlogin().startswith("pattam"):

    #sys.path.insert(0,'/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
    from hxntools.CompositeBroker import db
    from hxntools.scan_info import get_scan_positions

else: 
    db = None
    print("Offline analysis; No BL data available") 



det_params = {'merlin1':55, "merlin2":55, "eiger2_images":75}

def parse_scan_range(str_scan_range):
    """
    Parse a string representing a list or range of scan numbers.

    Example:
        "100-103, 105, 108-110" → [100, 101, 102, 103, 105, 108, 109, 110]

    Parameters
    ----------
    str_scan_range : str
        Comma-separated scan numbers and ranges (e.g., "100-102,105,110-112")

    Returns
    -------
    np.ndarray
        Sorted array of unique scan numbers as integers.
    """
    scan_numbers = set()
    for item in str_scan_range.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            try:
                start, end = map(int, item.split("-"))
                scan_numbers.update(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid range format: '{item}'")
        else:
            try:
                scan_numbers.add(int(item))
            except ValueError:
                raise ValueError(f"Invalid scan number: '{item}'")

    return np.array(sorted(scan_numbers), dtype=int)


def convert_old_fly2d_start_doc(old):
    """
    Convert a legacy FlyPlan2D start_doc into the new 2D_FLY_PANDA format.
    """
    # Map top‐level keys across
    new = {
        "time":             old["time"],
        "uid":              old["uid"],
        "beamline_id":      old.get("beamline_id", ""),
        "scan_id":          old["scan_id"],
        "sample":           old.get("sample", ""),
        "x_scale_factor":   old.get("x_scale_factor"),
        "z_scale_factor":   old.get("z_scale_factor"),
        "versions":         old.get("versions", {}),
        "PI":               old.get("PI", ""),
        "experimenters":    old.get("experimenters", ""),
        "config":           old.get("config", {}),
        "group":            old.get("group", ""),
        "scan_name":        old.get("scan_name", ""),
        # new‐style plan metadata
        "plan_type":        "generator",
        "plan_name":        "fly2dpd",
        # build the nested "scan" block
        "scan": {
            "type":             "2D_FLY_PANDA",
            "scan_input": [
                old["scan_start1"], old["scan_end1"], old["num1"],
                old["scan_start2"], old["scan_end2"], old["num2"],
            ],
            "sample_name":      "",
            "detectors":        old.get("detectors", []),
            "detector_distance": old.get("detector_distance"),  # if you have it
            "dwell":            old.get("exposure_time"),
            "fast_axis":        {"motor_name": old.get("motor1"), "units": "um"},
            "slow_axis":        {"motor_name": old.get("motor2"), "units": "um"},
            "shape":            list(old.get("shape", [])),
        },
        # leave these top‐level for compatibility
        "motors":           old.get("motors", []),
        "motor1":           old.get("motor1"),
        "motor2":           old.get("motor2"),
        "shape":            list(old.get("shape", [])),
        "fly_type":         old.get("fly_type", ""),
    }
    return new


def get_path(scan_id, key_name='merlin1'):
    """Return file path with given scan id and keyname.
    """
    
    h = db[int(scan_id)]
    e = list(db.get_events(h, fields=[key_name]))
    #id_list = [v.data[key_name] for v in e]
    id_list = [v['data'][key_name] for v in e]
    rootpath = db.reg.resource_given_datum_id(id_list[0])['root']
    flist = [db.reg.resource_given_datum_id(idv)['resource_path'] for idv in id_list]
    flist = set(flist)
    fpath = [os.path.join(rootpath, file_path) for file_path in flist]
    return fpath

def get_flyscan_dimensions(hdr):
    start_doc = hdr.start
    # 2D_FLY_PANDA: prefer 'dimensions', fallback to 'shape'
    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        if 'dimensions' in start_doc:
            return start_doc['dimensions']
        elif 'shape' in start_doc:
            return start_doc['shape']
        else:
            raise ValueError("No dimensions or shape found for 2D_FLY_PANDA scan")
    # rel_scan: use 'shape' or 'num_points'
    elif start_doc.get('plan_name') == 'rel_scan':
        if 'shape' in start_doc:
            return start_doc['shape']
        elif 'num_points' in start_doc:
            return [start_doc['num_points']]
        else:
            raise ValueError("No shape or num_points found for rel_scan")
    else:
        raise ValueError("Unknown scan type for get_flyscan_dimensions")

def get_all_scalar_data(hdr):

    keys = list(hdr.table().keys())
    scalar_keys = [k for k in keys if k.startswith('sclr1') ]
    #print(f"{scalar_keys = }")
    print(f"[DATA] fetching scalar data")
    scan_dim = get_flyscan_dimensions(hdr)
    scalar_stack_list = []

    for sclr in sorted(scalar_keys):
        
        scalar = np.array(list(hdr.data(sclr))).squeeze()
        sclr_img = scalar.reshape(scan_dim)
        scalar_stack_list.append(sclr_img)

    # Stack all the 2D images along a new axis (axis=0).
    scalar_stack = np.stack(scalar_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)

    return  scalar_stack, sorted(scalar_keys)

def get_all_xrf_roi_data(hdr):


    channels = [1, 2, 3]
    keys = list(hdr.table().keys())
    roi_keys = [k for k in keys if k.startswith('Det')]
    det1_keys = [k for k in keys if k.startswith('Det1')]
    elem_list = [k.replace("Det1_", "") for k in det1_keys]

    #print(f"{elem_list = }")
    print(f"[DATA] fetching XRF ROIs")
    scan_dim = get_flyscan_dimensions(hdr)
    xrf_stack_list = []

    for elem in sorted(elem_list):
        roi_keys = [f'Det{chan}_{elem}' for chan in channels]
        spectrum = np.sum([np.array(list(hdr.data(roi)), dtype=np.float32).squeeze() for roi in roi_keys], axis=0)
        xrf_img = spectrum.reshape(scan_dim)
        xrf_stack_list.append(xrf_img)

    # Stack all the 2D images along a new axis (axis=0).
    xrf_stack = np.stack(xrf_stack_list, axis=0)

    #print("3D Stack shape:", xrf_stack.shape)
    return xrf_stack, sorted(elem_list)

def get_scan_details(hdr):
    start_doc = hdr.start
    param_dict = {"scan_id": start_doc.get("scan_id")}
    # 2D_FLY_PANDA logic (original)
    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        df = db.get_table(hdr,stream_name = "baseline")
        mots = start_doc['motors']
        # Create a datetime object from the Unix time.
        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc["motors"]
        if "detectors" in start_doc.keys():
            param_dict["detectors"] = start_doc["detectors"]
            param_dict["scan_start1"] = start_doc["scan_start1"]
            param_dict["num1"] = start_doc["num1"]
            param_dict["scan_end1"] = start_doc["scan_end1"]
            if len(mots)==2:
                param_dict["scan_start2"] = start_doc["scan_start2"]
                param_dict["scan_end2"] = start_doc["scan_end2"]
                param_dict["num2"] = start_doc["num2"]
            param_dict["exposure_time"] = start_doc["exposure_time"]
        elif "scan" in start_doc.keys():
            param_dict["scan"] = start_doc["scan"]
        param_dict["zp_theta"] = np.round(df.zpsth.iloc[0],3)
        param_dict["mll_theta"] = np.round(df.dsth.iloc[0],3)
        param_dict["energy"] = np.round(df.energy.iloc[0],3)
        return param_dict
    # rel_scan logic
    elif start_doc.get('plan_name') == 'rel_scan':
        # Basic info
        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc.get("motors", [])
        param_dict["detectors"] = start_doc.get("detectors", [])
        param_dict["num_points"] = start_doc.get("num_points", None)
        param_dict["num_intervals"] = start_doc.get("num_intervals", None)
        param_dict["plan_args"] = start_doc.get("plan_args", {})
        param_dict["plan_type"] = start_doc.get("plan_type", None)
        param_dict["plan_name"] = start_doc.get("plan_name", None)
        param_dict["scan_name"] = start_doc.get("scan_name", None)
        param_dict["sample"] = start_doc.get("sample", None)
        param_dict["PI"] = start_doc.get("PI", None)
        param_dict["experimenters"] = start_doc.get("experimenters", None)
        param_dict["shape"] = start_doc.get("shape", None)
        # Add any other relevant keys as needed
        return param_dict
    else:
        print("[SCAN META] not all metadata was not exported; fallback option used")
        datetime_object = datetime.datetime.fromtimestamp(start_doc["time"])
        formatted_time = datetime_object.strftime('%Y-%m-%d %H:%M:%S')
        param_dict["time"] = formatted_time
        param_dict["motors"] = start_doc.get("motors", [])
        param_dict["detectors"] = start_doc.get("detectors", [])
        return param_dict

def get_scan_metadata(hdr):
    
    output = db.get_table(hdr,stream_name = "baseline")
    df_dictionary = pd.DataFrame([get_scan_details(hdr)])
    output = pd.concat([output, df_dictionary], ignore_index=True)
    return output


def save_dict_to_h5(group, dictionary):
    """Recursively store a dictionary into HDF5 format, handling unicode arrays and lists of strings. Skips None values."""
    for key, value in dictionary.items():
        if value is None:
            continue  # Skip None values
        if isinstance(value, dict):  # If it's a nested dictionary, create a subgroup
            subgroup = group.create_group(key)
            save_dict_to_h5(subgroup, value)
        else:
            # Handle numpy unicode arrays
            if isinstance(value, np.ndarray) and value.dtype.kind == 'U':
                value = value.astype('S')
            # Handle lists of unicode strings
            if isinstance(value, list) and value and isinstance(value[0], str):
                value = np.array(value, dtype='S')
            group.create_dataset(key, data=value)
            

def read_dict_from_h5(group):
    """
    Recursively read a dictionary from HDF5 format, decoding any byte-strings
    into Python str.
    """
    result = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = read_dict_from_h5(item)
        else:
            raw = item[()]
            result[key] = _decode_bytes(raw)
    return result

           
def _load_scan_common(hdr, mon, data_type='float32'):
    """
    Load everything *except* detector stacks (i.e. scan positions, xrf, scalar, scan params).
    Automatically converts old FlyPlan2D start docs to the new 2D_FLY_PANDA style.
    """

    # Pull and possibly convert the start doc
    sd = hdr.start
    if sd.get("plan_name") == "FlyPlan2D":
        print("[SCAN TYPE] old flyscan; converting scan startdoc")
        sd = convert_old_fly2d_start_doc(sd)
        # Overwrite hdr.start so downstream helpers see the new layout
        hdr.start = sd

    # Now all variants live under sd["scan"]["type"]
    plan = sd.get("scan", {}).get("type")
    if plan != "2D_FLY_PANDA":
        raise ValueError(f"[SCAN COMMON] Unsupported plan: {plan!r}")

    # Dimensions are in the nested scan dict
    dim1, dim2 = sd["scan"]["shape"]
    Io = None
    if mon:
        # reshape the monitor array into (dim1, dim2)
        raw_Io = list(hdr.data(str(mon)))
        Io = np.array(raw_Io, dtype=data_type).squeeze().reshape(dim1, dim2)

    # 2) Scan positions
    xy = list(get_scan_positions(hdr))

    # 3) XRF & scalar
    xrf_stack, xrf_names     = get_all_xrf_roi_data(hdr)
    scalar_stack, scalar_names = get_all_scalar_data(hdr)

    # 4) Scan parameters & metadata table
    sid = sd["scan_id"]
    scan_params = get_scan_details(hdr)
    scan_table  = get_scan_metadata(hdr)

    return {
        "Io": Io,
        "dim1": dim1,
        "dim2": dim2,
        "scan_positions": np.array(xy),
        "xrf_stack": xrf_stack,
        "xrf_names": xrf_names,
        "scalar_stack": scalar_stack,
        "scalar_names": scalar_names,
        "scan_params": scan_params,
        "scan_table": scan_table,
    }


def _load_detector_stack(hdr, det, data_type='float32'):
    """
    Load & reshape detector data more efficiently.
    """

    print("loading diff data")
    data_name = '/entry/instrument/detector/data'
    files = get_path(hdr.start["scan_id"], det)
    files = sorted(files, key=os.path.getctime)

    # Use memory mapping if possible for large datasets
    def read_file(fn):
        with h5py.File(fn, 'r') as f:
            return np.array(f[data_name], dtype=data_type)  # np.array slightly faster than np.asarray for HDF5

    # Option 1: Multithreading for I/O-bound task (if on fast shared FS)
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(4, len(files))) as executor:
        arrays = list(executor.map(read_file, files))

    # Concatenate if needed
    data = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=0)
    return np.flip(data, axis=1)


def _read_group_as_dict(group):
    out = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            out[key] = _read_group_as_dict(item)
        else:
            val = item[()]
            if isinstance(val, (bytes, bytearray)):
                val = val.decode('utf-8')
            elif isinstance(val, np.ndarray) and val.dtype.kind in ('S', 'O', 'a'):
                val = [v.decode('utf-8') if isinstance(v, (bytes, bytearray)) else v for v in val]
            out[key] = val
    return out

def strip_and_rename_entry_data(h5_path, det="merlin1", compression="gzip"):
    """
    Strip all HDF5 groups except /entry/data/data and move it to /diff_data/{det}/det_images
    """
    with h5py.File(h5_path, 'r+') as f:
        # Step 1: Reference original dataset without loading
        if "/entry/data/data" not in f:
            raise KeyError("'/entry/data/data' not found in the file")
        dset_ref = f["/entry/data/data"]

        # Step 2: Create target group and copy dataset (fast internal copy)
        grp = f.require_group(f"/diff_data/{det}")
        f.copy(dset_ref, grp, name="det_images")

        # Step 3: Delete everything *except* /diff_data/{det}
        to_delete = [k for k in f.keys() if k == "entry"]
        for k in to_delete:
            del f[k]

        # Optional: delete extra groups inside /diff_data if needed
        for k in list(f["diff_data"].keys()):
            if k != det:
                del f["diff_data"][k]

def _ensure_h5_compatible_array(arr):
    """Convert numpy unicode arrays to byte string arrays for HDF5 compatibility."""
    if isinstance(arr, np.ndarray) and arr.dtype.kind == 'U':
        return arr.astype('S')
    return arr

def export_fly2d_as_h5_single(
    hdr,
    det="merlin1",
    wd=".",
    mon="sclr1_ch4",
    compression="gzip",
    save_to_disk=True,
    copy_if_possible=True,
    save_and_return=False
):
    
        
    start_doc = hdr.start
    if 'scan' not in start_doc:
        print("[SCAN TYPE] old flyscan")
        start_doc = convert_old_fly2d_start_doc(start_doc)
    sid = start_doc["scan_id"]
    stop_doc = getattr(hdr, 'stop', None)
    scan_type = start_doc['scan']['type'] if 'scan' in start_doc and 'type' in start_doc['scan'] else ''
    detectors = ','.join(start_doc.get('detectors', []))
    exit_status = stop_doc.get('exit_status', '') if stop_doc else ''
    os_user = os.getlogin() if hasattr(os, 'getlogin') else getpass.getuser()
    raw_files = get_path(sid, det)
    raw_data_path = raw_files[0] if raw_files else ''
    if exit_status != '' and exit_status != 'success':
        print(f"[EXPORT] Scan {sid} exit_status is {exit_status}, skipping export")
        return {"scan_id": sid, "scan_type": scan_type, "detectors": detectors, "exit_status": exit_status, "status": "skipped_failed", "raw_data_path": raw_data_path, "os_user": os_user}
    if det not in start_doc["scan"].get("detectors", []):
        raise ValueError(f"[DETECTOR] Scan {sid} does not use detector {det}")
    print(f"[SCAN TYPE] Scan {sid} uses plan 2D_FLY_PANDA")
    common = _load_scan_common(hdr, mon)
    if not common:
        raise ValueError(f"[DATA] Scan {sid} cannot fetch scan data")
    out_fn = os.path.join(wd, f"scan_{sid}_{det}.h5")
    copied = False
    if copy_if_possible and len(raw_files) == 1:
        print(f"[DATA] Found the raw files; copying to h5")
        shutil.copy2(raw_files[0], out_fn)
        strip_and_rename_entry_data(out_fn, det=det)
        copied = True
    if save_to_disk:
        mode = "a" if copied else "w"
        with h5py.File(out_fn, mode) as f:
            grp = f.require_group(f"/diff_data/{det}")
            if not copied:
                raw = _load_detector_stack(hdr, det)  # shape (n_steps, ry, rx)
                grp.create_dataset(
                    "det_images",
                    data=raw,
                    compression=compression
                )
            if common["Io"] is not None:
                grp.create_dataset("Io", data=common["Io"])
            print("[SCAN] fetching scan positions")
            sp = f.require_group("scan_positions")
            sp.create_dataset("positions", data=common["scan_positions"])
            pp = f.require_group("scan_params")
            save_dict_to_h5(pp, common["scan_params"])
            xg = f.require_group("xrf_roi_data")
            xg.create_dataset("xrf_array",  data=_ensure_h5_compatible_array(common["xrf_stack"]))
            xg.create_dataset("xrf_elem_names", data=np.array(common["xrf_names"], dtype='S'))
            sg2 = f.require_group("scalar_data")
            sg2.create_dataset("scalar_array",       data=_ensure_h5_compatible_array(common["scalar_stack"]))
            sg2.create_dataset("scalar_array_names", data=np.array(common["scalar_names"], dtype='S'))
            scan_table = common.get("scan_table", None)
            if scan_table is not None:
                csv_fn = out_fn.replace('.h5', '.csv')
                scan_table.to_csv(csv_fn, index=False)
            if scan_table is not None and save_to_disk:
                print(f"[EXPORT] Scan {sid} has scan_table, saving diff_det_config")
                diff_cols = scan_table.columns[scan_table.columns.str.contains("diff", case=False)]
                if len(diff_cols) > 0:
                    #print(f"[EXPORT] Scan {sid} diff columns: {diff_cols}")
                    diff_config_grp = f.require_group("diff_det_config")
                    # Collect keys and values
                    keys = list(diff_cols)
                    # Stack values as 2D array (rows: columns, cols: values per column)
                    values = []
                    for col in keys:
                        val = scan_table[col].values
                        # If all NaN, skip this column
                        if np.all(pd.isna(val)):
                            print(f"[EXPORT WARNING] Scan {sid} diff column {col} is all NaN, skipping")
                            continue
                        # If single value and NaN, skip
                        if len(val) == 1 and pd.isna(val[0]):
                            print(f"[EXPORT WARNING] Scan {sid} diff column {col} is NaN, skipping")
                            continue
                        values.append(val)
                    # Only keep keys/values that were not skipped
                    valid_keys = [k for k, v in zip(keys, values)]
                    # Convert to arrays
                    if values:
                        # Pad values to same length if needed
                        diff_config_grp.create_dataset("names", data=np.array(valid_keys, dtype='S'))
                        diff_config_grp.create_dataset("values", data=values)
                    print(f"[EXPORT] Scan {sid} has diff columns, saving diff_det_config")
                else:
                    print(f"[EXPORT ERROR] Scan {sid} has no diff columns, skipping diff_det_config")
    if save_and_return:

        def load_det():
            with h5py.File(out_fn, "r") as f:
                print(f"[RETURN DATA] diff data is flipped along y axis when returned")
                return np.flip(f[f"/diff_data/{det}/det_images"][()], 1)
            

        """common;
            "Io": Io,
            "dim1": dim1,
            "dim2": dim2,
            "scan_positions": np.array(xy),
            "xrf_stack": xrf_stack,
            "xrf_names": xrf_names,
            "scalar_stack": scalar_stack,
            "scalar_names": scalar_names,
            "scan_params": scan_params,
            "scan_table": scan_table,
        """

        return {
            "det_images": load_det(),
            "Io": common.get("Io"),
            "scan_positions": common.get("scan_positions"),
            "scan_params": common.get("scan_params"),
            "xrf_array": common.get("xrf_stack"),
            "xrf_names": common.get("xrf_names"),
            "scalar_array": common.get("scalar_stack"),
            "scalar_names": common.get("scalar_names"),
        }

    return {
        "scan_id": sid,
        "scan_type": scan_type,
        "detectors": detectors,
        "exit_status": exit_status or 'success',
        "status": "exported",
        "raw_data_path": raw_data_path,
        "os_user": os_user
    }


    #     return {"scan_id": sid, "scan_type": '2D_FLY_PANDA', "detectors": '', "exit_status": '', "status": f"skipped_error: {e}", "raw_data_path": '', "os_user": os_user}

def export_relscan_as_h5_single(
    hdr,
    det="merlin1",
    wd=".",
    mon="sclr1_ch4",
    compression="gzip",
    save_to_disk=True,
    copy_if_possible=True,
    save_and_return=False
):
    
    start_doc   = hdr.start
    sid = start_doc["scan_id"]
    stop_doc = getattr(hdr, 'stop', None)
    scan_type = start_doc.get('plan_name', '')
    detectors = ','.join(start_doc.get('detectors', []))
    exit_status = stop_doc.get('exit_status', '') if stop_doc else ''
    os_user = os.getlogin() if hasattr(os, 'getlogin') else getpass.getuser()
    raw_files = get_path(sid, det)
    raw_data_path = raw_files[0] if raw_files else ''
    if exit_status != '' and exit_status != 'success':
        print(f"[EXPORT] Scan {sid} exit_status is {exit_status}, skipping export")
        return {"scan_id": sid, "scan_type": scan_type, "detectors": detectors, "exit_status": exit_status, "status": "skipped_failed", "raw_data_path": raw_data_path, "os_user": os_user}
    print(f"[SCAN TYPE] Scan {sid} uses plan rel_scan")
    motors = start_doc.get('motors', [])
    shape = start_doc.get('shape', None)
    if shape is None:
        if 'num_points' in start_doc:
            shape = [start_doc['num_points']]
        else:
            raise ValueError(f"Cannot determine scan shape for rel_scan in scan {sid}")
    try:
        xy = list(get_scan_positions(hdr))
        scan_positions = np.array(xy)
    except Exception:
        df = hdr.table()
        if len(motors) == 2:
            scan_positions = np.stack([df[motors[0]], df[motors[1]]], axis=1)
        elif len(motors) == 1:
            scan_positions = np.array(df[motors[0]])[:, None]
        else:
            scan_positions = None
    Io = None
    if mon and mon in hdr.table().columns:
        Io = np.array(list(hdr.data(str(mon))), dtype='float32').squeeze()
        if len(shape) == 2:
            Io = Io.reshape(shape)
    try:
        xrf_stack, xrf_names = get_all_xrf_roi_data(hdr)
    except Exception as e:
        print(f"[XRF DATA ERROR IGNORED] {e}")
        xrf_stack, xrf_names = None, []
    try:
        scalar_stack, scalar_names = get_all_scalar_data(hdr)
    except Exception as e:
        print(f"[SCALAR DATA ERROR IGNORED] {e}")
        scalar_stack, scalar_names = None, []
    scan_params = get_scan_details(hdr)
    scan_table  = get_scan_metadata(hdr)
    out_fn = os.path.join(wd, f"scan_{sid}_{det}.h5")
    copied = False
    raw_files = get_path(sid, det)
    if raw_files:
        raw_data_path = raw_files[0]
    if copy_if_possible and len(raw_files) == 1:
        shutil.copy2(raw_files[0], out_fn)
        strip_and_rename_entry_data(out_fn, det=det)
        copied = True
    if save_to_disk:
        mode = "a" if copied else "w"
        with h5py.File(out_fn, mode) as f:
            grp = f.require_group(f"/diff_data/{det}")
            if not copied:
                raw = _load_detector_stack(hdr, det)
                if len(shape) == 2:
                    raw = raw.reshape(shape[0], shape[1], *raw.shape[1:])
                elif len(shape) == 1:
                    raw = raw.reshape(shape[0], *raw.shape[1:])
                grp.create_dataset(
                    "det_images",
                    data=raw,
                    compression=compression
                )
            if Io is not None:
                grp.create_dataset("Io", data=Io)
            sp = f.require_group("scan_positions")
            sp.create_dataset("positions", data=scan_positions)
            pp = f.require_group("scan_params")
            save_dict_to_h5(pp, scan_params)
            xg = f.require_group("xrf_roi_data")
            if xrf_stack is not None:
                xg.create_dataset("xrf_array",  data=_ensure_h5_compatible_array(xrf_stack))
            if xrf_names is not None:
                xg.create_dataset("xrf_elem_names", data=np.array(xrf_names, dtype='S'))
            sg2 = f.require_group("scalar_data")
            if scalar_stack is not None:
                sg2.create_dataset("scalar_array",       data=_ensure_h5_compatible_array(scalar_stack))
            if scalar_names is not None:
                sg2.create_dataset("scalar_array_names", data=np.array(scalar_names, dtype='S'))
            scan_table = common.get("scan_table", None)
            if scan_table is not None:
                csv_fn = out_fn.replace('.h5', '.csv')
                scan_table.to_csv(csv_fn, index=False)
            if scan_table is not None and save_to_disk:
                diff_cols = scan_table.columns[scan_table.columns.str.contains("diff", case=False)]
                if len(diff_cols) > 0:
                    diff_config_grp = f.require_group("diff_det_config")
                    # Collect keys and values
                    keys = list(diff_cols)
                    values = []
                    for col in keys:
                        val = scan_table[col].values
                        # If all NaN, skip this column
                        if np.all(pd.isna(val)):
                            print(f"[EXPORT WARNING] Scan {sid} diff column {col} is all NaN, skipping")
                            continue
                        # If single value and NaN, skip
                        if len(val) == 1 and pd.isna(val[0]):
                            print(f"[EXPORT WARNING] Scan {sid} diff column {col} is NaN, skipping")
                            continue
                        values.append(val)
                    # Only keep keys/values that were not skipped
                    valid_keys = [k for k, v in zip(keys, values)]
                    if values:
                        diff_config_grp.create_dataset("names", data=np.array(valid_keys, dtype='S'))
                        diff_config_grp.create_dataset("values", data=values)
                    print(f"[EXPORT] Scan {sid} has diff columns, saving diff_det_config")
                else:
                    print(f"[EXPORT ERROR] Scan {sid} has no diff columns, skipping diff_det_config")
    if save_and_return:
        return {
            "det_images": raw,
            "Io": Io,
            "scan_positions": scan_positions,
            "scan_params": {"motors": motors, "shape": shape},
            "xrf_array": xrf_array,
            "xrf_names": xrf_names,
            "scalar_array": scalar_array,
            "scalar_names": scalar_names,
        }
        
    return {"scan_id": sid, "scan_type": scan_type, "detectors": detectors, "exit_status": exit_status or 'success', "status": "exported", "raw_data_path": raw_data_path, "os_user": os_user}
    

def export_diff_data_as_h5_single(
    sid,
    det="merlin1",
    wd=".",
    mon="sclr1_ch4",
    compression="gzip",
    save_to_disk=True,
    copy_if_possible=True,
    save_and_return=False
):
    """
    Dispatches to the correct export function based on scan type.
    Returns a dict with logbook info: scan_id, scan_type, detectors, exit_status, status, raw_data_path, os_user.
    """

    hdr = db[int(sid)]
    start_doc = hdr.start
    if start_doc.get("plan_name") == "FlyPlan2D":
        print("[SCAN TYPE] old fly2d; converting the startdoc")
        start_doc = convert_old_fly2d_start_doc(start_doc)
    sid = start_doc["scan_id"]
    if 'scan' in start_doc and start_doc['scan'].get('type') == '2D_FLY_PANDA':
        return export_fly2d_as_h5_single(hdr, det, wd, mon, compression, save_to_disk, copy_if_possible, save_and_return)
    elif start_doc.get('plan_name') == 'rel_scan':
        return export_relscan_as_h5_single(hdr, det, wd, mon, compression, save_to_disk, copy_if_possible, save_and_return)
    else:
        raise ValueError(f"[SCAN TYPE] Scan {sid} is of unknown type")



def export_diff_data_as_h5_batch(
    sid_list,
    det="merlin1",
    wd=".",
    mon="sclr1_ch4",
    compression="gzip",
    copy_if_possible=True,
    overwrite=False
):
    """
    Batch‐export one detector from each scan in sid_list:
      • Calls export_diff_data_as_h5_single(...) with save_to_disk=True,
        copy_if_possible as given, and save_and_return=False.
      • If overwrite is False, skips scans whose HDF5 file already exists.
      • Always skips scans whose exit_status is not 'success'.
      • Prints a warning if a scan is skipped.
      • Writes a batch_export_log.csv file with columns: scan_id, scan_type, detectors, exit_status, status, raw_data_path, os_user.
      • Returns None.
    """
    # normalize to list
    if isinstance(sid_list, (int, float)):
        sid_list = [int(sid_list)]

    first_sid = sid_list[0]
    last_sid = sid_list[-1]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"batch_export_log_{first_sid}_to_{last_sid}_{timestamp}.csv"

    log_path = os.path.join(wd, log_filename)
    log_exists = os.path.exists(log_path)
    log_fields = [  "scan_id", 
                    "scan_type",
                    "detectors",
                    "exit_status", 
                    "status", 
                    "raw_data_path", 
                    "os_user", 
                    "error"]
    log_rows = []

    for sid in tqdm(sid_list, desc="Batch exporting scans"):
        
        out_fn = os.path.join(wd, f"scan_{sid}_{det}.h5")
        if not overwrite and os.path.exists(out_fn):
            print(f"Skipping scan {sid!r}: {out_fn} already exists (overwrite=False)")
            
            os_user = os.getlogin() if hasattr(os, 'getlogin') else getpass.getuser()
            log_rows.append({   "scan_id": sid, 
                                "scan_type": '', 
                                "detectors": '', 
                                "exit_status": '', 
                                "status": "skipped_exists", 
                                "raw_data_path": '', 
                                "os_user": os_user})
            continue
        try:
            log_info = export_diff_data_as_h5_single(
                sid,
                det=det,
                wd=wd,
                mon=mon,
                compression=compression,
                save_to_disk=True,
                copy_if_possible=copy_if_possible,
                save_and_return=False
            )
        
        except Exception as e:
            log_info = {
                "scan_id": sid,
                "scan_type": '',
                "detectors": '',
                "exit_status": '',
                "status": "error",
                "raw_data_path": '',
                "os_user": os_user,
                "error": str(e)
            }
        log_rows.append(log_info)
    # Write log
    write_header = not log_exists
    with open(log_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        if write_header:
            writer.writeheader()
        for row in log_rows:
            writer.writerow(row)


def unpack_diff_h5(filename, det="merlin1"):
    """
    Unpack a single‐detector HDF5 file with this structure:

      diff_data/<det>/{Io, det_images}
      scalar_data/{scalar_array, scalar_array_names}
      scan_positions/positions
      scan_params/...         (possibly nested)
      xrf_roi_data/{xrf_array, xrf_elem_names}

    Returns a dict with keys:
      det_images, Io,
      scalar_array, scalar_names,
      scan_positions,
      scan_params,
      xrf_array, xrf_names
    """
    def _decode_list(arr):
        return [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else x
            for x in arr
        ]

    result = {}
    with h5py.File(filename, "r") as f:
        # 1) diff_data
        dd = f["diff_data"][det]
        result["det_images"] = dd["det_images"][()]
        result["Io"] = dd["Io"][()] if "Io" in dd else None

        # 2) scalar_data
        sd = f["scalar_data"]
        result["scalar_array"] = sd["scalar_array"][()]
        raw_sn = sd["scalar_array_names"][()]
        result["scalar_names"] = _decode_list(raw_sn.tolist())

        # 3) scan_positions
        sp = f["scan_positions"]
        result["scan_positions"] = sp["positions"][()]

        # 4) scan_params (recursive)
        result["scan_params"] = _read_group_as_dict(f["scan_params"])

        # 5) xrf_roi_data
        if "xrf_roi_data" in f:
            xg = f["xrf_roi_data"]
            result["xrf_array"] = xg["xrf_array"][()]
            raw_xn = xg["xrf_elem_names"][()]
            result["xrf_names"] = _decode_list(raw_xn.tolist())
        else:
            result["xrf_array"] = None
            result["xrf_names"] = []

    return result


def export_selected_scan_details_to_csv(
    scan_ids: List[int],
    fields_of_interest: List[str],
    csv_path: str = "selected_scan_details.csv",
    error_log_path: Optional[str] = None,
    return_dataframe: bool = False
) -> Optional[pd.DataFrame]:
    """
    Export only selected fields from get_scan_details for each scan in scan_ids.
    - scan_ids: list of scan IDs (ints)
    - fields_of_interest: list of column names to export (besides 'scan_id')
    - csv_path: output CSV file path
    - error_log_path: optional path to save error logs
    - return_dataframe: if True, returns the DataFrame of results

    Returns: DataFrame if return_dataframe is True, else None
    """
    all_rows = []
    errors = []
    all_fields = set(['scan_id'] + fields_of_interest)

    for sid in tqdm(scan_ids, desc="Exporting scans"):
        try:
            hdr = db[int(sid)]
            details = get_scan_details(hdr)
            row = {"scan_id": details.get("scan_id", sid)}
            for field in fields_of_interest:
                row[field] = details.get(field, None)
            all_rows.append(row)
            all_fields.update(row.keys())
        except Exception as e:
            error_msg = f"Skipping scan {sid} due to error: {e}"
            print(error_msg)
            errors.append({"scan_id": sid, "error": str(e)})

    # Ensure all rows have all fields
    all_fields = list(all_fields)
    for row in all_rows:
        for field in all_fields:
            if field not in row:
                row[field] = None

    # Write to CSV
    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=all_fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"Export complete. Data saved to: {csv_path}")

    # Write error log if requested
    if error_log_path and errors:
        with open(error_log_path, mode='w', newline='') as errfile:
            err_writer = csv.DictWriter(errfile, fieldnames=["scan_id", "error"])
            err_writer.writeheader()
            for err in errors:
                err_writer.writerow(err)
        print(f"Errors logged to: {error_log_path}")

    if return_dataframe:
        return pd.DataFrame(all_rows)
    return None

if __name__ == "__main__" or "get_ipython" in globals():
    print("\n✅ Diffraction data I/O module loaded.")
    
    print("\n#####📘 For export only use this ######:\n") 
    print("▶ export_diff_data_as_h5_batch(sid_list, det, wd, mon, compression, save_to_disk, copy_if_possible)")
    print("   → Fast bulk exporter. If only 1 raw HDF5 file exists, it will copy instead of re-saving.")
    
    print("\n#####📘 For export and return/visualize data ######:\n") 
    print("▶ export_diff_data_as_h5_single(sid_list, det, wd, mon, compression, save_to_disk, return_data)")
    print("   → Saves or returns data for one or more scan IDs.")
    
    print("\n#####📘 To read the h5 saved using export_diff_data_as_h5 function ######:\n") 
    print("▶ unpack_diff_h5(filename, det)")
    print("   → Reads saved HDF5 back into a dictionary (diff, scan, XRF, scalar).")
    print("----------------------------------------------------------")
    print("\n#####📘 To export selected scan metadata fields to CSV (beamline log) ######:\n")
    print("▶ export_selected_scan_details_to_csv(scan_ids, fields_of_interest, csv_path='selected_scan_details.csv', error_log_path=None, return_dataframe=False)")
    print("   → Exports only the specified fields from get_scan_details for each scan in scan_ids to a CSV file.")
    print("   → Arguments:")
    print("      - scan_ids: list of scan IDs (ints)")
    print("      - fields_of_interest: list of column names to export (besides 'scan_id')")
    print("      - csv_path: output CSV file path (default: 'selected_scan_details.csv')")
    print("      - error_log_path: optional path to save error logs (default: None)")
    print("      - return_dataframe: if True, returns the DataFrame of results (default: False)")
    print("   → Example usage:")
    print("      fields = ['energy', 'dcm_energy', 'ugap', 'm1', 'm2', 'time']")
    print("      scan_ids = [200001, 200002, 200003]")
    print("      export_selected_scan_details_to_csv(scan_ids, fields, csv_path='beamline_log.csv', error_log_path='beamline_log_errors.csv')")
    print("----------------------------------------------------------")