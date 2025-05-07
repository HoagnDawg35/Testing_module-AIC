# Multi-Camera Multi-Object 3D Tracking System

This project implements a real-time multi-camera tracking system that can track multiple objects in 3D space using synchronized camera feeds. The system uses Kalman filtering for state estimation and the Hungarian algorithm for data association across multiple camera views.

## Features

- Multi-camera support with synchronized tracking
- 3D object tracking with position, dimensions, and rotation
- Kalman filtering for smooth tracking and prediction
- Data association using the Hungarian algorithm
- 3D visualization of tracked objects and camera positions
- Camera calibration utilities

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- PyTorch
- FilterPy
- SciPy
- Matplotlib
- scikit-image
- pykalman

Install the required packages using:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── multi_camera_tracker.py  # Main tracking implementation
├── utils.py                # Utility functions for calibration and visualization
├── example.py             # Example usage script
└── requirements.txt       # Project dependencies
```

## Components

### 1. MultiCameraTracker
The main tracking class that handles:
- Multi-camera object tracking
- 3D position estimation
- Data association
- Track management

### 2. Object3D
Represents a tracked object with:
- 3D position
- Dimensions
- Rotation
- Kalman filter state

### 3. Utilities
- Camera calibration using checkerboard patterns
- 3D visualization
- 3D IoU calculation

## Usage

1. **Camera Calibration**
```python
from utils import calibrate_camera

# Calibrate each camera using checkerboard images
calibration_data = calibrate_camera(
    images=checkerboard_images,
    pattern_size=(9, 6),
    square_size=0.025
)
```

2. **Initialize Tracker**
```python
from multi_camera_tracker import MultiCameraTracker

tracker = MultiCameraTracker(camera_calibrations)
```

3. **Update with Detections**
```python
# Update tracker with detections from multiple cameras
tracked_objects = tracker.update(detections_by_camera)
```

4. **Visualize Results**
```python
from utils import visualize_3d_tracking

visualize_3d_tracking(tracked_objects, camera_positions)
```

## Example

See `example.py` for a complete working example that demonstrates:
- Setting up camera calibrations
- Initializing the tracker
- Processing detections
- Visualizing results

## Implementation Details

### State Estimation
The system uses a 9-dimensional state vector:
- Position (x, y, z)
- Velocity (vx, vy, vz)
- Acceleration (ax, ay, az)

### Data Association
- Uses the Hungarian algorithm for optimal assignment
- Implements 3D IoU for similarity measurement
- Handles unmatched detections and tracks

### Camera Calibration
- Uses OpenCV's calibration functions
- Supports multiple cameras
- Handles lens distortion

## Future Improvements

1. Implement proper 3D IoU calculation
2. Add more sophisticated triangulation
3. Implement object re-identification
4. Add support for object classification
5. Implement more advanced data association methods
6. Add support for camera synchronization
7. Implement real-time visualization

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the MIT License - see the LICENSE file for details. 