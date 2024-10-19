import logging
import os
import cv2
import numpy as np
import open3d as o3d
import torch


frame_folder = "input_files/8_MR_20240620_121902.068000"
torchhub_model = "intel-isl/MiDaS"
model_type = "MiDaS_small"

def load_MiDaS_model():
    # Detect the device to use (MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load MiDaS model for depth estimation
    midas2 = torch.hub.load(torchhub_model, model_type).to(device)
    midas2.to(device)
    midas2.eval()
    
    midas_transforms = torch.hub.load(torchhub_model, "transforms")
    
    transform = midas_transforms.small_transform

    print(f"Loaded {torchhub_model} of type {model_type}")
    return device, model_type, midas2, midas_transforms, transform

# Parameters for Lucas-Kanade optical flow with CUDA support
lk_params = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Function to load frames
def load_frames(frame_folder, num_frames = 70): #fixme
    frames = []
    for i in range(num_frames):
        frame = cv2.imread(f'{frame_folder}/frame_{i:04d}.png')
        frames.append(frame)
    print(f"Loaded frames")
    return frames

# Function to estimate depth for each frame
def estimate_depth(frames):
    depth_maps = []
    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            depth_maps.append(prediction.squeeze().cpu().numpy())
    return depth_maps

# Function to detect and track features
def detect_and_track_features(frames):
    gray1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(gray1, maxCorners=1000, qualityLevel=0.01, minDistance=8, blockSize=19)
    
    all_tracked_features = [features]
    for i in range(1, len(frames)):
        gray_next = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        next_features, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray_next, features, None, **lk_params)
        valid_points = status == 1
        features = next_features[valid_points.flatten()]
        all_tracked_features.append(features)
        gray1 = gray_next
    return all_tracked_features

# Function to triangulate points
def triangulate_points(all_features, proj_matrices):
    all_points_3d = []
    for i in range(len(all_features) - 1):
        points1 = all_features[i]
        points2 = all_features[i + 1]
        proj_matrix1 = proj_matrices[i]
        proj_matrix2 = proj_matrices[i + 1]
        points4d_hom = cv2.triangulatePoints(proj_matrix1, proj_matrix2, points1.T, points2.T)
        points4d = points4d_hom / points4d_hom[3]
        all_points_3d.append(points4d[:3].T)
    return np.vstack(all_points_3d)

def main(input_directory, output_directory): 
    num_frames = 70
    frames = load_frames(frame_folder, num_frames)

    # Estimate depth
    depth_maps = estimate_depth(frames)
    print (f"Depth_maps calculated {len(depth_maps)}")
    # Detect and track features
    all_tracked_features = detect_and_track_features(frames)
    print (f"Features tracked: {len(all_tracked_features)}")
    # Triangulate points (using simplified projection matrices)
    proj_matrices = [np.hstack((np.eye(3), np.zeros((3, 1)))) for _ in range(num_frames)]
    all_points_3d = triangulate_points(all_tracked_features, proj_matrices)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points_3d)
    
    # Estimate normals
    pcd.estimate_normals()
    
    # Create mesh using Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    
    # Simplify mesh
    dec_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
    
    # Save the mesh
    o3d.io.write_triangle_mesh("skull_model.ply", dec_mesh)
    
    # Interactive visualization
    def custom_draw_geometry_with_key_callback(mesh):
        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            return False
    
        def move_forward(vis):
            ctr = vis.get_view_control()
            ctr.translate(0, 0, 0.1)
            return False
    
        def move_backward(vis):
            ctr = vis.get_view_control()
            ctr.translate(0, 0, -0.1)
            return False
    
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window()
        vis.add_geometry(mesh)
        vis.register_key_callback(262, rotate_view)
        vis.register_key_callback(87, move_forward)  # W key
        vis.register_key_callback(83, move_backward)  # S key
        vis.run()
        vis.destroy_window()

    custom_draw_geometry_with_key_callback(dec_mesh)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_directory = frame_folder
    output_directory = "output_files/"
    main(input_directory,input_directory)


