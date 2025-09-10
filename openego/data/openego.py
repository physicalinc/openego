from ..core.utils import get_sorted_paths, get_video_info, get_video_frames, load_json, get_hdf5_data
from ..core.constants import MANO_JOINT_NAMES, EGODEX_JOINT_NAMES
from typing import List, Mapping, Optional, Any
from pathlib import Path
import numpy as np
import h5py


class OpenEgoDataProvider:
    def __init__(
        self,
        data_dir: Path,
        data_types: List[str] = ["joint", "rgb", "annotation", "metadata"],
        # benchmarks: List = [], # Leave empty to include all
    ):  
        self.data_dir = data_dir
        assert data_dir.exists(), f"Data directory does not exist: {data_dir}"
        self.data_types = data_types
        self.video_paths = get_sorted_paths(self.data_dir, "*.mp4")
        self._video_benchmarks = [get_benchmark_name(video_path) for video_path in self.video_paths]
        self.benchmarks = sorted(list(set(self._video_benchmarks)))
        self._video_infos = [get_video_info(video_path) for video_path in self.video_paths]

    def __len__(self):
        return len(self.video_paths)
    
    @property
    def num_demos(self) -> int:
        return self.__len__()

    @property
    def num_frames(self) -> int:
        return sum([info['num_frames'] for info in self._video_infos])

    @property
    def duration(self) -> float: # In seconds
        return sum([info['duration'] for info in self._video_infos])

    def __getitem__(self, index: int, demo_slice: Optional[slice] = None) -> Mapping[str, np.ndarray]:
        video_path = self.video_paths[index]
        benchmark_name = self._video_benchmarks[index]
        video_info = self._video_infos[index]

        data = {}
        if "joint" in self.data_types:
            data['joint'] = self._load_joint(video_path, benchmark_name, demo_slice)
        if "annotation" in self.data_types:
            data['annotation'] = self._load_annotation(video_path, benchmark_name, demo_slice)
        if "metadata" in self.data_types:
            data['metadata'] = self._load_metadata(video_path, benchmark_name, video_info)
            data['metadata']['video_path'] = self.video_paths[index]
            data['metadata']['benchmark'] = self._video_benchmarks[index]
        if "rgb" in self.data_types:
            data['rgb'] = self._load_rgb(video_path, demo_slice)
        
        return data

    def _load_rgb(self, video_path: Path, demo_slice: Optional[slice] = None):
        return get_video_frames(video_path, demo_slice)

    def _load_joint(self, video_path: Path, benchmark_name: str, demo_slice: Optional[slice] = None):
        if demo_slice is not None:
            raise NotImplementedError("Slicing for joint data loading is not yet implemented.")

        if benchmark_name == "egodex":
            return get_egodex_joints(video_path)
        elif benchmark_name in self.benchmarks:
            return get_hdf5_data(video_path.parent/"joints.hdf5")
        else:
            raise RuntimeError(f"Unknown benchmark or missing joint data for video: {video_path}")
        
    def _load_annotation(self, video_path: Path, benchmark_name: str, demo_slice: Optional[slice] = None):
        if benchmark_name == 'egodex':
            raise NotImplementedError("Need to implement annotation loading for egodex.")
        elif benchmark_name in self.benchmarks:
            return load_json(video_path.parent/"annotation.json")
        else: 
            raise RuntimeError(f"Unknown benchmark or missing annotation data for video: {video_path}")

    def _load_metadata(self, video_path: Path, benchmark_name: str, video_info: Mapping[str, Any]):
        if benchmark_name == 'egodex':
            return { 
                "intrinsic": get_egodex_intrinsic(video_path),
                "original_metadata" : { "task": video_path.parent.name.replace("_", " ") },
                **video_info
            }
        elif benchmark_name in self.benchmarks:
            demo_path = video_path.parent
            return { "original_metadata": get_hdf5_data(demo_path/"original_metadata.hdf5"), 
                    **get_hdf5_data(demo_path/"metadata.hdf5"), }
        else:
            raise RuntimeError(f"Unknown benchmark or missing metadata for video: {video_path}")


def get_benchmark_name(video_path: Path) -> str:
    if "demo" in video_path.parent.name:
        return video_path.parent.parent.name.lower().strip()
    elif "part" in video_path.parent.parent.name:
        return video_path.parent.parent.parent.name.lower().strip()
    else:
        raise ValueError(f"Cannot determine benchmark name from path: {video_path}")

def get_egodex_joints(video_path: Path, visibility_confidence_threshold: float = 0.5) -> Mapping[str, np.ndarray]:
    with h5py.File(video_path.with_suffix(".hdf5"), "r") as f:
        data = {}
        for key in f.keys():
            item = f[key]
            if isinstance(item, h5py.Dataset):
                data[key] = item[()]
            elif isinstance(item, h5py.Group):
                for subkey in item.keys():
                    if isinstance(item[subkey], h5py.Dataset):
                        data_key = EGODEX_JOINT_NAMES.get(subkey, subkey) + ("_confidence" if key == "confidences" else "")
                        data[data_key] = item[subkey][()]

    hand_dict = {}
    for hand_key in ["left_hand", "right_hand"]:
        hand_name = hand_key.replace("_hand", "")
        hand_dict[hand_key] = np.stack([data[f"{hand_name}_{key}"][:, :3, 3] for key in MANO_JOINT_NAMES], axis=1)
        confidence = np.stack([data.get(f"{hand_name}_{key}_confidence", np.ones_like(hand_dict[hand_key])[..., 0]) for key in MANO_JOINT_NAMES], axis=1)
        hand_dict[f"{hand_key}_visibility"] = (confidence>visibility_confidence_threshold).astype(np.int32)

    hand_dict['intrinsics'] = data['intrinsic']
    hand_dict['joint_names'] = np.array(MANO_JOINT_NAMES)
    return hand_dict

def get_egodex_intrinsic(video_path: Path) -> np.ndarray:
    with h5py.File(video_path.with_suffix(".hdf5"), "r") as f:
        intrinsic = f["camera"]["intrinsic"][()]
    return intrinsic