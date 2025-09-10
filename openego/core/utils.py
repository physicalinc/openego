from typing import Union, Mapping, Any, Optional, List
from pathlib import Path
import numpy as np
import h5py
import json
import cv2

def load_json(file_path: Union[str, Path]) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)

def get_video_frames(video_path: Path, frame_slice: Optional[slice] = None) -> np.ndarray:
    """Load video frames using OpenCV."""
    video_capture = cv2.VideoCapture(str(video_path))
    num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = range(num_frames) if frame_slice is None else range(*frame_slice.indices(num_frames))
    frames: List = []
    for frame_index in frame_indices:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video_capture.read()
        if not success: 
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    video_capture.release()
    return np.stack(frames) if frames else np.empty((0, 0, 0, 3), dtype=np.uint8)

def get_video_info(video_path: Union[str, Path]) -> dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")
    
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    info = {
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": int(round(raw_fps)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["num_frames"] / raw_fps if raw_fps > 0 else 0.0
    cap.release()
    return info

def get_sorted_paths(dir_path: Path, pattern: str) -> List[Path]:
    """Get sorted file paths matching pattern."""
    from natsort import natsorted
    return natsorted(list(dir_path.rglob(pattern)))

def get_hdf5_data(file_path: Path, key: Optional[str] = None) -> Any:
    """Load data from HDF5 file."""
    with h5py.File(file_path, "r") as f:
        if key is not None:
            return f[key][()]
        return {key: f[key][()] for key in f.keys()}