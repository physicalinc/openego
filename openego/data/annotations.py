from ..core.projection import convert_points_to_trajetory_coordinates
from ..core.utils import get_video_frames
from typing import List, Optional, Mapping, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class Action:
    start_timestamp: float
    end_timestamp: float
    objects: List[str]
    actors: List[str]
    label: str
    fps: int
    video_joints: Optional[Mapping[str, np.ndarray]] = None
    video_path: Optional[Path] = None

    @property
    def dict(self) -> Mapping[str, Any]:
        d = self.__dict__.copy()
        d.pop('video_path')
        d.pop('video_joints')
        d['joints'] = self.joints
        return d

    @property
    def duration(self) -> float:
        return self.end_timestamp - self.start_timestamp
    
    @property
    def start_frame(self) -> int:
        return int(round(self.start_timestamp * self.fps))

    @property
    def end_frame(self) -> int:
        return int(round(self.end_timestamp * self.fps))
    
    @property
    def frames(self) -> np.ndarray:
        if self.video_path is None:
            raise ValueError("video_path is not set for this ActionAnnotation.")
        
        return get_video_frames(self.video_path, slice(self.start_frame, self.end_frame))
    
    @property
    def joints(self) -> Mapping[str, np.ndarray]:
        return dict(left_hand=self.left_hand_joints, left_hand_visibility=self.left_hand_visibility,
                    right_hand=self.right_hand_joints, right_hand_visibility=self.right_hand_visibility,
                    intrinsic=self.intrinsic)
    
    @property
    def joints_pixel(self) -> Mapping[str, np.ndarray]:
        return dict(left_hand=self.left_hand_pixel_joints, left_hand_visibility=self.left_hand_visibility,
                    right_hand=self.right_hand_pixel_joints, right_hand_visibility=self.right_hand_visibility,
                    intrinsic=self.intrinsic)

    @property
    def intrinsic(self) -> Optional[np.ndarray]:
        return self.video_joints.get('intrinsics', None)

    @property
    def left_hand_joints(self) -> np.ndarray:
        return self.video_joints['left_hand'][self.start_frame: self.end_frame]
    
    @property
    def right_hand_joints(self) -> np.ndarray:
        return self.video_joints['right_hand'][self.start_frame: self.end_frame]

    @property
    def left_hand_visibility(self) -> np.ndarray:
        vis = self.video_joints['left_hand_visibility'][self.start_frame: self.end_frame]
        # Handle both 1D and 2D visibility arrays
        if len(vis.shape) == 1:
            # Broadcast 1D array to match joint shape
            vis = vis[:, np.newaxis].repeat(21, axis=1)
        return vis
    
    @property
    def right_hand_visibility(self) -> np.ndarray:
        vis = self.video_joints['right_hand_visibility'][self.start_frame: self.end_frame]
        # Handle both 1D and 2D visibility arrays
        if len(vis.shape) == 1:
            # Broadcast 1D array to match joint shape
            vis = vis[:, np.newaxis].repeat(21, axis=1)
        return vis
    
    @property
    def left_hand_pixel_joints(self) -> np.ndarray:
        return convert_points_to_trajetory_coordinates(self.left_hand_joints, self.intrinsic)

    @property
    def right_hand_pixel_joints(self) -> np.ndarray:
        return convert_points_to_trajetory_coordinates(self.right_hand_joints, self.intrinsic)

    def visualize(self):
        if self.video_path is None:
            raise ValueError("video_path is not set for this ActionAnnotation, cannot visualize frames.")
        import imageio
        import tempfile
        import sys

        frames = self.frames
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpfile:
            imageio.mimsave(tmpfile.name, frames, fps=float(self.fps))
            print(f"Label: '{self.label}'")
            try:
                from IPython.display import display, Video
                # Check if we're in a notebook by looking for 'ipykernel' in sys.modules
                if 'ipykernel' in sys.modules:
                    display(Video(tmpfile.name, embed=True))
                else:
                    print(f"Video saved to: {tmpfile.name}")
            except ImportError:
                print(f"Video saved to: {tmpfile.name}")