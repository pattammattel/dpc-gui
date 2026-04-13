import os
import h5py
import numpy as np
from pathlib import Path

try:
    from hxntools.CompositeBroker import db
except ImportError:
    import sys
    sys.path.insert(0, '/nsls2/data2/hxn/shared/config/bluesky_overlay/2023-1.0-py310-tiled/lib/python3.10/site-packages')
    try:
        from hxntools.CompositeBroker import db
    except ImportError:
        db = None
        print("Offline analysis; hxntools not found")
        


def lazy_return_data(scan_id, key_name='merlin1', dataset_path='/entry/instrument/detector/data', return_file=False):
    """
    Return lazy-loaded HDF5 dataset for given scan_id without loading into memory.
    h5py already does lazy loading - just keep the file open!
    
    Parameters
    ----------
    scan_id : int
        Scan ID number
    key_name : str, optional
        Detector key name (default: 'merlin1')
    dataset_path : str, optional
        Path to dataset in HDF5 file
    return_file : bool, optional
        If True, returns file handle instead of dataset (default: False)
    
    Returns
    -------
    h5py.Dataset or h5py.File
        Open HDF5 dataset (or file if return_file=True).
        Data is lazy-loaded (not in memory until you slice it).
        
        WARNING: The underlying file stays open. Either:
        1. Keep the reference alive, or
        2. The file will close when object is garbage collected
    
    Examples
    --------
    # Direct dataset access (recommended):
    data = lazy_return_data(349821)
    print(data.shape)      # (1600, 499, 464) - no data loaded yet
    first_frame = data[0]  # Only loads 1 frame
    subset = data[0:10, :, :]  # Only loads 10 frames
    
    # If you need the file handle:
    f = lazy_return_data(349821, return_file=True)
    data = f['/entry/instrument/detector/data']
    other_data = f['/some/other/path']
    f.close()
    """
    
    h = db[int(scan_id)]
    e = list(db.get_events(h, fields=[key_name]))
    id_list = [v['data'][key_name] for v in e]
    rootpath = db.reg.resource_given_datum_id(id_list[0])['root']

    # Convert to Path object
    root = Path(rootpath)

    # Check if it exists and resolve if it's a symlink
    if root.exists():
        try:
            root = root.resolve(strict=False)
        except Exception as e:
            print(f"Warning: Failed to resolve path {root}: {e}")
    else:
        print(f"Warning: root path {root} does not exist.")

    rootpath = str(root)

    if str(rootpath).startswith("/data"):
        rootpath = '/nsls2/data2/hxn/legacy'

    print(f"Data path: {rootpath}")

    flist = [db.reg.resource_given_datum_id(idv)['resource_path'] for idv in id_list]
    flist = set(flist)
    fpath = [os.path.join(rootpath, file_path) for file_path in flist]
    
    # Open file and get dataset
    file_handle = h5py.File(fpath[0], 'r')
    if dataset_path in file_handle:
        dataset = file_handle[dataset_path]
        print(f"Dataset shape: {dataset.shape}, dtype: {dataset.dtype}")
        
        if return_file:
            return file_handle
        else:
            # Return the dataset. File stays open as long as dataset exists.
            return dataset
    else:
        print(f"Warning: Dataset path '{dataset_path}' not found in file.")
        print(f"Available keys: {list(file_handle.keys())}")
        return file_handle if return_file else None