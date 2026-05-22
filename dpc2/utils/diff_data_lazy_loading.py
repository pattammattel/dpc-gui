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


def sum_frames_chunked(dataset, chunk_size=100, axes=(1, 2)):
    """
    Memory-efficient sum over spatial dimensions (x, y) for (num_frames, x, y) data.
    Processes data in chunks to avoid loading entire dataset into memory.
    
    Parameters
    ----------
    dataset : h5py.Dataset or np.ndarray
        Data with shape (num_frames, x, y)
    chunk_size : int, optional
        Number of frames to load at once (default: 100)
    axes : tuple, optional
        Axes to sum over (default: (1, 2) for x and y)
    
    Returns
    -------
    np.ndarray
        1D array of shape (num_frames,) with sum of each frame
    
    Examples
    --------
    # Memory-efficient sum:
    data = lazy_return_data(349821)
    frame_sums = sum_frames_chunked(data, chunk_size=50)
    
    # Or with direct file access:
    with h5py.File('data.h5', 'r') as f:
        data = f['/entry/instrument/detector/data']
        sums = sum_frames_chunked(data)
    """
    num_frames = dataset.shape[0]
    result = np.zeros(num_frames, dtype=np.float64)
    
    # Process in chunks
    for i in range(0, num_frames, chunk_size):
        end_idx = min(i + chunk_size, num_frames)
        # Load only chunk_size frames at a time
        chunk = dataset[i:end_idx]
        # Sum over x,y dimensions for this chunk
        result[i:end_idx] = np.sum(chunk, axis=axes)
        
    return result


def sum_frames_direct(dataset, axes=(1, 2)):
    """
    Direct sum over spatial dimensions - loads all data into memory.
    Fast but memory-intensive. Use only if dataset fits in RAM.
    
    Parameters
    ----------
    dataset : h5py.Dataset or np.ndarray
        Data with shape (num_frames, x, y)
    axes : tuple, optional
        Axes to sum over (default: (1, 2) for x and y)
    
    Returns
    -------
    np.ndarray
        1D array of shape (num_frames,) with sum of each frame
    
    Examples
    --------
    data = lazy_return_data(349821)
    frame_sums = sum_frames_direct(data)  # Loads all into memory
    """
    return np.sum(dataset[()], axis=axes)