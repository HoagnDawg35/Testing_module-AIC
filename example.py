import cv2
import numpy as np
from multi_camera_tracker import MultiCameraTracker
from utils import calibrate_camera, visualize_3d_tracking

def main():
    # Example camera calibration data
    # In a real application, you would calibrate each camera using the calibrate_camera function
    camera_calibrations = {
        0: {
            'camera_matrix': np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0,    0,   1]
            ]),
            'dist_coeffs': np.zeros(5),
            'rvec': np.array([0, 0, 0]),
            'tvec': np.array([0, 0, 0])
        },
        1: {
            'camera_matrix': np.array([
                [1000, 0, 320],
                [0, 1000, 240],
                [0,    0,   1]
            ]),
            'dist_coeffs': np.zeros(5),
            'rvec': np.array([0, np.pi/4, 0]),
            'tvec': np.array([5, 0, 0])
        }
    }
    
    # Initialize the tracker
    tracker = MultiCameraTracker(camera_calibrations)
    
    # Example camera positions in 3D space
    camera_positions = [
        np.array([0, 0, 0]),
        np.array([5, 0, 0])
    ]
    
    # Example detection data from two cameras
    # In a real application, you would get these from your object detection system
    detections_by_camera = {
        0: [
            {
                'bbox': [100, 100, 200, 200],
                'center': [150, 150],
                'confidence': 0.9,
                'dimensions': [1.0, 1.0, 1.0],
                'rotation': [0, 0, 0]
            }
        ],
        1: [
            {
                'bbox': [150, 150, 250, 250],
                'center': [200, 200],
                'confidence': 0.85,
                'dimensions': [1.0, 1.0, 1.0],
                'rotation': [0, 0, 0]
            }
        ]
    }
    
    # Update the tracker with new detections
    tracked_objects = tracker.update(detections_by_camera)
    
    # Visualize the tracking results
    visualize_3d_tracking(tracked_objects, camera_positions)

if __name__ == '__main__':
    main() 