import numpy as np
import cv2
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import torch
import torchvision

class Object3D:
    def __init__(self, id, position, dimensions, rotation):
        self.id = id
        self.position = position  # [x, y, z]
        self.dimensions = dimensions  # [length, width, height]
        self.rotation = rotation  # [roll, pitch, yaw]
        self.kalman_filter = self._init_kalman_filter()
        
    def _init_kalman_filter(self):
        kf = KalmanFilter(dim_x=9, dim_z=6)  # State: [x, y, z, vx, vy, vz, ax, ay, az], Measurement: [x, y, z, vx, vy, vz]
        dt = 1.0/30.0  # Assuming 30 FPS
        
        # State transition matrix
        kf.F = np.array([
            [1, 0, 0, dt, 0, 0, 0.5*dt**2, 0, 0],
            [0, 1, 0, 0, dt, 0, 0, 0.5*dt**2, 0],
            [0, 0, 1, 0, 0, dt, 0, 0, 0.5*dt**2],
            [0, 0, 0, 1, 0, 0, dt, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, dt, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, dt],
            [0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])
        
        # Measurement noise
        kf.R = np.eye(6) * 0.1
        
        # Process noise
        kf.Q = np.eye(9) * 0.1
        
        # Initial state
        kf.x = np.array([self.position[0], self.position[1], self.position[2], 0, 0, 0, 0, 0, 0])
        
        return kf
    
    def update(self, measurement):
        self.kalman_filter.predict()
        self.kalman_filter.update(measurement)
        self.position = self.kalman_filter.x[:3]
        self.velocity = self.kalman_filter.x[3:6]

class MultiCameraTracker:
    def __init__(self, camera_calibrations):
        self.camera_calibrations = camera_calibrations
        self.tracked_objects = {}
        self.next_id = 0
        self.max_age = 30  # Maximum number of frames an object can be missing
        
    def _calculate_3d_position(self, detections, camera_id):
        """Convert 2D detections to 3D positions using camera calibration"""
        camera_matrix = self.camera_calibrations[camera_id]['camera_matrix']
        dist_coeffs = self.camera_calibrations[camera_id]['dist_coeffs']
        rvec = self.camera_calibrations[camera_id]['rvec']
        tvec = self.camera_calibrations[camera_id]['tvec']
        
        # Convert 2D points to 3D using camera parameters
        # This is a simplified version - you'll need to implement proper triangulation
        points_3d = []
        for detection in detections:
            point_2d = detection['center']
            point_3d = cv2.solvePnP(
                np.array([point_2d], dtype=np.float32),
                np.array([[0, 0, 0]], dtype=np.float32),
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            points_3d.append(point_3d[1].flatten())
        
        return points_3d
    
    def _calculate_iou_3d(self, box1, box2):
        """Calculate 3D IoU between two bounding boxes"""
        # Implement 3D IoU calculation
        # This is a simplified version - you'll need to implement proper 3D IoU
        return 0.0
    
    def _associate_detections_to_trackers(self, detections, trackers):
        """Associate detections to existing trackers using Hungarian algorithm"""
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 3), dtype=int)
        
        cost_matrix = np.zeros((len(detections), len(trackers)))
        
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                cost_matrix[d, t] = self._calculate_iou_3d(det, trk)
        
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in row_ind:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in col_ind:
                unmatched_trackers.append(t)
        
        matches = np.empty((0, 2), dtype=int)
        for d, t in zip(row_ind, col_ind):
            if cost_matrix[d, t] < 0.3:  # IoU threshold
                unmatched_detections.append(d)
                unmatched_trackers.append(t)
            else:
                matches = np.vstack((matches, np.array([d, t])))
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def update(self, detections_by_camera):
        """
        Update the tracker with new detections from multiple cameras
        
        Args:
            detections_by_camera: Dictionary mapping camera IDs to lists of detections
                Each detection should be a dictionary with 'bbox', 'center', and 'confidence'
        """
        # Convert 2D detections to 3D positions for each camera
        all_3d_detections = []
        for camera_id, detections in detections_by_camera.items():
            positions_3d = self._calculate_3d_position(detections, camera_id)
            for pos_3d, det in zip(positions_3d, detections):
                all_3d_detections.append({
                    'position': pos_3d,
                    'dimensions': det.get('dimensions', [1.0, 1.0, 1.0]),
                    'rotation': det.get('rotation', [0.0, 0.0, 0.0]),
                    'confidence': det['confidence']
                })
        
        # Associate detections with existing trackers
        matches, unmatched_detections, unmatched_trackers = self._associate_detections_to_trackers(
            all_3d_detections,
            list(self.tracked_objects.values())
        )
        
        # Update matched trackers
        for d, t in matches:
            self.tracked_objects[t].update(all_3d_detections[d]['position'])
        
        # Create new trackers for unmatched detections
        for i in unmatched_detections:
            self.tracked_objects[self.next_id] = Object3D(
                self.next_id,
                all_3d_detections[i]['position'],
                all_3d_detections[i]['dimensions'],
                all_3d_detections[i]['rotation']
            )
            self.next_id += 1
        
        # Remove old trackers
        self.tracked_objects = {k: v for k, v in self.tracked_objects.items() 
                              if v.age < self.max_age}
        
        return self.tracked_objects 