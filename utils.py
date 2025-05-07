import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calibrate_camera(images, pattern_size=(9, 6), square_size=0.025):
    """
    Calibrate a camera using a checkerboard pattern
    
    Args:
        images: List of images containing checkerboard patterns
        pattern_size: Size of the checkerboard pattern (inner corners)
        square_size: Size of each square in meters
    
    Returns:
        Dictionary containing camera calibration parameters
    """
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size
    
    objpoints = []
    imgpoints = []
    
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    return {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }

def visualize_3d_tracking(tracked_objects, camera_positions=None):
    """
    Visualize 3D tracking results
    
    Args:
        tracked_objects: Dictionary of tracked objects
        camera_positions: List of camera positions in 3D space
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot tracked objects
    for obj_id, obj in tracked_objects.items():
        position = obj.position
        dimensions = obj.dimensions
        
        # Create 3D box vertices
        vertices = np.array([
            [position[0] - dimensions[0]/2, position[1] - dimensions[1]/2, position[2] - dimensions[2]/2],
            [position[0] + dimensions[0]/2, position[1] - dimensions[1]/2, position[2] - dimensions[2]/2],
            [position[0] + dimensions[0]/2, position[1] + dimensions[1]/2, position[2] - dimensions[2]/2],
            [position[0] - dimensions[0]/2, position[1] + dimensions[1]/2, position[2] - dimensions[2]/2],
            [position[0] - dimensions[0]/2, position[1] - dimensions[1]/2, position[2] + dimensions[2]/2],
            [position[0] + dimensions[0]/2, position[1] - dimensions[1]/2, position[2] + dimensions[2]/2],
            [position[0] + dimensions[0]/2, position[1] + dimensions[1]/2, position[2] + dimensions[2]/2],
            [position[0] - dimensions[0]/2, position[1] + dimensions[1]/2, position[2] + dimensions[2]/2]
        ])
        
        # Plot box edges
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]
        
        for edge in edges:
            ax.plot3D(
                [vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]],
                'b-'
            )
        
        # Add object ID label
        ax.text(position[0], position[1], position[2], f'ID: {obj_id}')
    
    # Plot camera positions if provided
    if camera_positions:
        for i, pos in enumerate(camera_positions):
            ax.scatter(pos[0], pos[1], pos[2], c='r', marker='^')
            ax.text(pos[0], pos[1], pos[2], f'Camera {i}')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def calculate_3d_iou(box1, box2):
    """
    Calculate 3D IoU between two bounding boxes
    
    Args:
        box1: First bounding box (position, dimensions, rotation)
        box2: Second bounding box (position, dimensions, rotation)
    
    Returns:
        IoU value between 0 and 1
    """
    # This is a simplified version - implement proper 3D IoU calculation
    # You might want to use a library like trimesh for proper 3D intersection
    return 0.0 