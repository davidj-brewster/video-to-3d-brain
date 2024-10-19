import logging
import os
import cv2
import numpy as np
import open3d as o3d
import torch

# Configuration
FRAME_FOLDER = "input_files/8_MR_20240620_121902.068000"
TORCHHUB_MODEL = "intel-isl/MiDaS"
MODEL_TYPE = "MiDaS_small"
OUTPUT_MESH_FILE = "skull_model.ply"
NUM_FRAMES = 70
OUTPUT_DIRECTORY = "output_files/"

def load_MiDaS_model(model_type: str, torchhub_model: str):
    """
    Loads the MiDaS model for depth estimation along with the necessary transforms.

    Args:
        model_type (str): The type of MiDaS model to load (e.g., "MiDaS_small").
        torchhub_model (str): The repository from which to load the model.

    Returns:
        tuple: A tuple containing the device, MiDaS model, transforms, and the specific transform.
    """
    # Detect the device to use (MPS, CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load MiDaS model for depth estimation
    midas = torch.hub.load(torchhub_model, model_type).to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load(torchhub_model, "transforms")
    if model_type == "MiDaS_small":
        transform = midas_transforms.small_transform
    else:
        transform = midas_transforms.default_transform

    logging.info(f"Loaded {torchhub_model} of type {model_type} on device {device}")
    return device, midas, midas_transforms, transform

# Parameters for Lucas-Kanade optical flow
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
)

def load_frames(frame_folder: str, num_frames: int):
    """
    Loads a specified number of frames from the given directory.

    Args:
        frame_folder (str): Path to the directory containing frame images.
        num_frames (int): Number of frames to load.

    Returns:
        list: A list of loaded frames as NumPy arrays.
    """
    frames = []
    for i in range(num_frames):
        frame_path = os.path.join(frame_folder, f'frame_{i:04d}.png')
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Frame {frame_path} could not be loaded and will be skipped.")
            continue
        frames.append(frame)
    logging.info(f"Loaded {len(frames)} frames from {frame_folder}")
    return frames

def estimate_depth(frames: list, device: torch.device, midas: torch.nn.Module, transform):
    """
    Estimates depth maps for each frame using the MiDaS model.

    Args:
        frames (list): List of frames as NumPy arrays.
        device (torch.device): The device to perform computations on.
        midas (torch.nn.Module): The loaded MiDaS model.
        transform: The transformation to apply to each frame before depth estimation.

    Returns:
        list: A list of depth maps as NumPy arrays.
    """
    depth_maps = []
    for idx, frame in enumerate(frames):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()
            depth_maps.append(depth_map)
        if idx % 10 == 0:
            logging.info(f"Estimated depth for frame {idx}/{len(frames)}")
    logging.info("Completed depth estimation for all frames.")
    return depth_maps

def create_point_clouds(frames: list, depth_maps: list, intrinsic_matrix: np.ndarray):
    """
    Creates point clouds from frames and their corresponding depth maps.

    Args:
        frames (list): List of frames as NumPy arrays.
        depth_maps (list): List of depth maps as NumPy arrays.
        intrinsic_matrix (np.ndarray): Camera intrinsic matrix.

    Returns:
        open3d.geometry.PointCloud: Combined point cloud from all frames.
    """
    combined_pcd = o3d.geometry.PointCloud()
    for idx, (frame, depth) in enumerate(zip(frames, depth_maps)):
        height, width = depth.shape
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # Create a mesh grid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        u = u.flatten()
        v = v.flatten()
        depth = depth.flatten()

        # Convert depth map to 3D points
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        points = np.vstack((x, y, z)).T

        # Get colors from the frame
        colors = frame[v, u, ::-1] / 255.0  # Convert BGR to RGB and normalize

        # Create Open3D point cloud for the current frame
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Optional: Downsample for performance
        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        # Transform point cloud to align frames (assuming no camera movement)
        # If camera movement exists, apply appropriate transformations here

        # Combine with the main point cloud
        combined_pcd += pcd

        if idx % 10 == 0:
            logging.info(f"Processed point cloud for frame {idx}/{len(frames)}")

    logging.info("Combined point cloud created from all frames.")
    return combined_pcd

def create_mesh_from_point_cloud(pcd: o3d.geometry.PointCloud, depth: int = 9):
    """
    Creates a mesh from the given point cloud using Poisson surface reconstruction.

    Args:
        pcd (open3d.geometry.PointCloud): The input point cloud.
        depth (int, optional): Depth parameter for Poisson reconstruction. Defaults to 9.

    Returns:
        open3d.geometry.TriangleMesh: The reconstructed mesh.
    """
    logging.info("Estimating normals for the point cloud.")
    pcd.estimate_normals()

    logging.info("Performing Poisson surface reconstruction.")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)

    # Remove low-density vertices to clean up the mesh
    density_threshold = np.percentile(densities, 5)
    vertices_to_remove = densities < density_threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    logging.info("Simplifying the mesh.")
    simplified_mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)

    return simplified_mesh

def visualize_mesh(mesh: o3d.geometry.TriangleMesh):
    """
    Opens an interactive window to visualize the mesh with custom key callbacks.

    Args:
        mesh (open3d.geometry.TriangleMesh): The mesh to visualize.
    """
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

    logging.info("Starting interactive mesh visualization.")
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(mesh)
    vis.register_key_callback(262, rotate_view)  # Right arrow key
    vis.register_key_callback(ord('W'), move_forward)  # W key
    vis.register_key_callback(ord('S'), move_backward)  # S key
    vis.run()
    vis.destroy_window()
    logging.info("Visualization ended.")

def main(input_directory: str, output_directory: str):
    """
    Main function to orchestrate the 3D reconstruction process from 2D MRI frames.

    Args:
        input_directory (str): Path to the directory containing input frame images.
        output_directory (str): Path to the directory where output files will be saved.
    """
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Load MiDaS model
    device, midas, midas_transforms, transform = load_MiDaS_model(MODEL_TYPE, TORCHHUB_MODEL)

    # Load frames
    frames = load_frames(input_directory, NUM_FRAMES)
    if not frames:
        logging.error("No frames loaded. Exiting program.")
        return

    # Estimate depth maps
    depth_maps = estimate_depth(frames, device, midas, transform)

    # Define camera intrinsic matrix (Assumed or obtained from calibration)
    # These values should be set according to your camera's specifications
    # For example purposes, we use placeholder values
    fx, fy = 1000, 1000  # Focal lengths in pixels
    cx, cy = frames[0].shape[1] / 2, frames[0].shape[0] / 2  # Principal point at image center
    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])

    # Create combined point cloud from all frames
    combined_pcd = create_point_clouds(frames, depth_maps, intrinsic_matrix)

    # Downsample the combined point cloud for efficiency
    combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.005)
    logging.info(f"Downsampled combined point cloud to {len(combined_pcd.points)} points.")

    # Create mesh from point cloud
    mesh = create_mesh_from_point_cloud(combined_pcd)

    # Save the mesh
    output_mesh_path = os.path.join(output_directory, OUTPUT_MESH_FILE)
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    logging.info(f"Saved mesh to {output_mesh_path}")

    # Visualize the mesh
    visualize_mesh(mesh)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Define input and output directories
    input_directory = FRAME_FOLDER
    output_directory = OUTPUT_DIRECTORY

    # Run the main function
    main(input_directory, output_directory)

