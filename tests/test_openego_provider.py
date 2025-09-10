"""Tests for OpenEgoDataProvider class."""

import pytest
import numpy as np
from pathlib import Path
from openego import OpenEgoDataProvider, Action


class TestOpenEgoDataProvider:
    """Test suite for OpenEgoDataProvider."""
    
    @pytest.fixture
    def test_data_dir(self):
        """Return path to test data directory."""
        return Path(__file__).parent / "test-data"
    
    @pytest.fixture
    def provider(self, test_data_dir):
        """Create OpenEgoDataProvider instance with test data."""
        return OpenEgoDataProvider(
            data_dir=test_data_dir,
            data_types=["joint", "rgb", "annotation", "metadata"]
        )
    
    def test_init_with_valid_directory(self, test_data_dir):
        """Test initialization with valid data directory."""
        provider = OpenEgoDataProvider(data_dir=test_data_dir)
        assert provider.data_dir == test_data_dir
        assert provider.data_types == ["joint", "rgb", "annotation", "metadata"]
        assert len(provider.video_paths) > 0
    
    def test_init_with_invalid_directory(self):
        """Test initialization with non-existent directory."""
        with pytest.raises(AssertionError, match="Data directory does not exist"):
            OpenEgoDataProvider(data_dir=Path("/non/existent/path"))
    
    def test_len(self, provider):
        """Test __len__ returns number of videos."""
        assert len(provider) == 1  # We have 1 demo video
        assert provider.__len__() == 1
    
    def test_num_demos_property(self, provider):
        """Test num_demos property."""
        assert provider.num_demos == 1
        assert provider.num_demos == len(provider)
    
    def test_num_frames_property(self, provider):
        """Test num_frames property returns total frames."""
        assert provider.num_frames == 676  # From annotation.json
        assert isinstance(provider.num_frames, int)
    
    def test_duration_property(self, provider):
        """Test duration property returns total duration in seconds."""
        expected_duration = 22.533333333333335  # From annotation.json
        assert abs(provider.duration - expected_duration) < 0.1
        assert isinstance(provider.duration, float)
    
    def test_benchmarks_property(self, provider):
        """Test benchmarks property."""
        assert "ho-cap" in provider.benchmarks
        assert isinstance(provider.benchmarks, list)
        assert len(provider.benchmarks) > 0
    
    def test_getitem_all_data_types(self, provider):
        """Test __getitem__ returns all requested data types."""
        data = provider[0]
        
        # Check all data types are present
        assert "joint" in data
        assert "rgb" in data
        assert "annotation" in data
        assert "metadata" in data
        
        # Check data shapes and types
        assert isinstance(data["joint"], dict)
        assert "left_hand" in data["joint"]
        assert "right_hand" in data["joint"]
        assert isinstance(data["joint"]["left_hand"], np.ndarray)
        
        assert isinstance(data["rgb"], np.ndarray)
        assert len(data["rgb"].shape) == 4  # (frames, height, width, channels)
        
        assert isinstance(data["annotation"], dict)
        assert "actions" in data["annotation"]
        
        assert isinstance(data["metadata"], dict)
        assert "video_path" in data["metadata"]
        assert "benchmark" in data["metadata"]
    
    def test_getitem_subset_data_types(self, test_data_dir):
        """Test __getitem__ with subset of data types."""
        provider = OpenEgoDataProvider(
            data_dir=test_data_dir,
            data_types=["annotation", "metadata"]
        )
        data = provider[0]
        
        # Only requested data types should be present
        assert "annotation" in data
        assert "metadata" in data
        assert "joint" not in data
        assert "rgb" not in data
    
    def test_getitem_invalid_index(self, provider):
        """Test __getitem__ with invalid index."""
        with pytest.raises(IndexError):
            _ = provider[99]
    
    def test_load_rgb(self, provider):
        """Test RGB frame loading."""
        data = provider[0]
        rgb = data["rgb"]
        
        assert isinstance(rgb, np.ndarray)
        assert rgb.dtype == np.uint8
        assert len(rgb.shape) == 4
        assert rgb.shape[0] == 676  # num frames
        assert rgb.shape[1] == 720  # height
        assert rgb.shape[2] == 1280  # width
        assert rgb.shape[3] == 3  # RGB channels
    
    def test_load_joint(self, provider):
        """Test joint data loading."""
        data = provider[0]
        joints = data["joint"]
        
        assert isinstance(joints, dict)
        assert "left_hand" in joints
        assert "right_hand" in joints
        assert "left_hand_visibility" in joints
        assert "right_hand_visibility" in joints
        assert "intrinsics" in joints
        assert "joint_names" in joints
        
        # Check shapes
        assert joints["left_hand"].shape == (676, 21, 3)  # frames, joints, xyz
        assert joints["right_hand"].shape == (676, 21, 3)
        assert joints["left_hand_visibility"].shape == (676,)  # frames only
        assert joints["right_hand_visibility"].shape == (676,)
        assert joints["intrinsics"].shape == (3, 3)
        assert len(joints["joint_names"]) == 21
    
    def test_load_annotation(self, provider):
        """Test annotation loading."""
        data = provider[0]
        annotation = data["annotation"]
        
        assert isinstance(annotation, dict)
        assert "task" in annotation
        assert "actions" in annotation
        assert "video_info" in annotation
        
        assert annotation["task"] == "Picking up and moving various objects on a table."
        assert len(annotation["actions"]) == 4
        
        # Check first action
        first_action = annotation["actions"][0]
        assert first_action["start_timestamp"] == 1.0
        assert first_action["end_timestamp"] == 3.6
        assert first_action["objects"] == ["Tazo green tea carton"]
        assert first_action["actors"] == ["left_hand"]
    
    def test_load_metadata(self, provider):
        """Test metadata loading."""
        data = provider[0]
        metadata = data["metadata"]
        
        assert isinstance(metadata, dict)
        assert "video_path" in metadata
        assert "benchmark" in metadata
        assert "original_metadata" in metadata
        
        assert metadata["benchmark"] == "ho-cap"
        assert str(metadata["video_path"]).endswith("demo_0000/video.mp4")
    
    def test_get_benchmark_name(self):
        """Test benchmark name extraction from paths."""
        from openego.data.openego import get_benchmark_name
        
        # Test standard demo path
        path = Path("/data/HO-Cap/demo_0000/video.mp4")
        assert get_benchmark_name(path) == "ho-cap"
        
        # Test with different case
        path = Path("/data/HOI4D/demo_0001/video.mp4")
        assert get_benchmark_name(path) == "hoi4d"
    
    def test_action_dataclass(self, provider):
        """Test Action dataclass functionality."""
        data = provider[0]
        action_dict = data["annotation"]["actions"][0]
        
        # Add required fields for Action
        action_dict["fps"] = 30
        action_dict["video_joints"] = data["joint"]
        action_dict["video_path"] = provider.video_paths[0]
        
        action = Action(**action_dict)
        
        # Test properties
        assert action.start_frame == 30  # 1.0s * 30fps
        assert action.end_frame == 108  # 3.6s * 30fps
        assert action.label == "left hand grasps, lifts, and then places the Tazo green tea carton back on the table"
        
        # Test joint access
        assert action.left_hand_joints.shape == (78, 21, 3)  # 108-30 frames
        assert action.right_hand_joints.shape == (78, 21, 3)
        
        # Test dict property
        action_data = action.dict
        assert "joints" in action_data
        assert "start_timestamp" in action_data
        assert "video_path" not in action_data  # Should be excluded
    
    def test_action_frames_property(self, provider):
        """Test Action.frames property loads video frames."""
        data = provider[0]
        action_dict = data["annotation"]["actions"][0]
        action_dict["fps"] = 30
        action_dict["video_joints"] = data["joint"]
        action_dict["video_path"] = provider.video_paths[0]
        
        action = Action(**action_dict)
        frames = action.frames
        
        assert isinstance(frames, np.ndarray)
        assert frames.shape[0] == 78  # 108-30 frames
        assert frames.shape[1:] == (720, 1280, 3)  # height, width, channels
    
    def test_action_pixel_joints(self, provider):
        """Test Action pixel coordinate conversion."""
        data = provider[0]
        action_dict = data["annotation"]["actions"][0]
        action_dict["fps"] = 30
        action_dict["video_joints"] = data["joint"]
        action_dict["video_path"] = provider.video_paths[0]
        
        action = Action(**action_dict)
        
        # Test pixel joint properties
        left_pixel = action.left_hand_pixel_joints
        right_pixel = action.right_hand_pixel_joints
        
        assert isinstance(left_pixel, np.ndarray)
        assert isinstance(right_pixel, np.ndarray)
        assert left_pixel.shape == (78, 21, 2)  # frames, joints, xy
        assert right_pixel.shape == (78, 21, 2)
        assert left_pixel.dtype == np.int32
    
    def test_video_slice(self, test_data_dir):
        """Test loading video with frame slice."""
        # Create provider with only RGB data to test slicing
        provider = OpenEgoDataProvider(
            data_dir=test_data_dir,
            data_types=["rgb"]  # Only RGB, no joints
        )
        
        # Test with slice
        data = provider.__getitem__(0, demo_slice=slice(10, 20))
        rgb = data["rgb"]
        
        assert rgb.shape[0] == 10  # Should have 10 frames
        assert rgb.shape[1:] == (720, 1280, 3)
        
        # Also test that joint slicing raises error
        provider_with_joints = OpenEgoDataProvider(
            data_dir=test_data_dir,
            data_types=["joint"]
        )
        with pytest.raises(NotImplementedError, match="Slicing for joint data loading"):
            _ = provider_with_joints.__getitem__(0, demo_slice=slice(10, 20))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])