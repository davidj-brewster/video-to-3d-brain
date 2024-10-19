## Overview

The **3D Brain MRI Reconstruction** script is a Python-based tool designed to generate a 3D model of the brain from a sequence of two-dimensional (2D) MRI images. By leveraging advanced computer vision techniques and deep learning models, the script processes a series of MRI frames to estimate depth information, create point clouds, and reconstruct a detailed 3D mesh. This tool is particularly useful for medical professionals, researchers, and developers interested in visualizing and analyzing brain structures in three dimensions.

## Features

- **Depth Estimation:** Utilizes the MiDaS model for accurate depth map generation from 2D MRI frames.
- **Point Cloud Generation:** Converts depth maps into 3D point clouds representing spatial information.
- **Mesh Reconstruction:** Employs Open3D's Poisson surface reconstruction to create a coherent 3D mesh from combined point clouds.
- **Interactive Visualization:** Provides an interactive interface to view and manipulate the reconstructed 3D brain model.
- **Configurable Parameters:** Allows customization of input directories, output locations, and processing parameters.

## Requirements

- **Operating System:** Compatible with Windows, macOS, and Linux.
- **Python:** Version 3.7 or higher.
- **Libraries:**
  - [OpenCV](https://opencv.org/) (`cv2`)
  - [NumPy](https://numpy.org/) (`numpy`)
  - [Open3D](http://www.open3d.org/) (`open3d`)
  - [PyTorch](https://pytorch.org/) (`torch`)
- **Hardware:**
  - **GPU (Optional):** CUDA-compatible GPU or Apple Silicon (MPS) for accelerated processing.
  - **CPU:** Sufficient processing power for depth estimation and mesh reconstruction.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/3d-brain-mri-reconstruction.git
   cd 3d-brain-mri-reconstruction
Create a Virtual Environment (Optional but Recommended):
bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install Dependencies:
bash
Copy code
pip install -r requirements.txt
If requirements.txt is not provided, install the libraries manually:
bash
Copy code
pip install opencv-python numpy open3d torch
Verify Installation: Ensure that all libraries are installed correctly by running:
bash
Copy code
python -c "import cv2, numpy, open3d, torch; print('All dependencies are installed.')"
Usage

Prepare Input Frames:
Extract the MRI video into individual image frames.
Ensure that frames are named in the format frame_XXXX.png (e.g., frame_0000.png, frame_0001.png, etc.).
Place all frame images in a designated input directory, e.g., input_files/8_MR_20240620_121902.068000/.
Configure Script Parameters:
Open the script file and adjust the following configurations as needed:
python
Copy code
FRAME_FOLDER = "input_files/8_MR_20240620_121902.068000"
TORCHHUB_MODEL = "intel-isl/MiDaS"
MODEL_TYPE = "MiDaS_small"
OUTPUT_MESH_FILE = "skull_model.ply"
NUM_FRAMES = 70
OUTPUT_DIRECTORY = "output_files/"
Run the Script:
bash
Copy code
python reconstruct_3d_mri.py
Replace reconstruct_3d_mri.py with the actual script filename.
View the Output:
The script will generate a 3D mesh file (skull_model.ply) in the specified output directory.
An interactive visualization window will open, allowing you to explore the reconstructed 3D brain model.
Processing Steps

The script follows a structured pipeline to convert 2D MRI frames into a 3D brain model. Below is a detailed explanation of each processing step:
1. Load MiDaS Model
Purpose:
Initialize the MiDaS model for depth estimation, selecting the appropriate device (GPU or CPU) for computation.
Process:
Detect available hardware accelerators (MPS for Apple Silicon, CUDA for NVIDIA GPUs, or CPU).
Load the MiDaS model (MiDaS_small) from the Intel ISL repository using PyTorch Hub.
Load the corresponding transformation pipeline required for preprocessing input images.
2. Load Frames
Purpose:
Read and load a sequence of MRI frames from the specified input directory.
Process:
Iterate through the expected number of frames (NUM_FRAMES), reading each image file.
Validate successful loading of each frame, logging warnings for any missing or corrupted files.
Compile all successfully loaded frames into a list for further processing.
3. Estimate Depth Maps
Purpose:
Generate depth information for each 2D MRI frame to understand the spatial structure.
Process:
For each frame:
Convert the image from BGR to RGB color space.
Apply the necessary transformations using MiDaS's small_transform.
Pass the transformed image through the MiDaS model to obtain a depth map.
Resize the depth prediction to match the original image dimensions.
Store the resulting depth map for subsequent steps.
4. Create Point Clouds
Purpose:
Convert depth maps into 3D point clouds, representing the spatial positions of each pixel.
Process:
Define the camera intrinsic matrix, which includes focal lengths and principal point coordinates.
For each frame and its corresponding depth map:
Generate a mesh grid of pixel coordinates.
Calculate the 3D coordinates (x, y, z) for each pixel using the depth values and intrinsic parameters.
Extract color information from the original frame for each point.
Create an Open3D point cloud object, assigning both points and colors.
Optionally downsample the point cloud to improve performance.
Accumulate all individual point clouds into a combined point cloud representing the entire sequence.
5. Combine Point Clouds
Purpose:
Merge individual point clouds from all frames into a single, unified point cloud for comprehensive 3D reconstruction.
Process:
Aggregate all point clouds generated from each frame.
Apply voxel downsampling to the combined point cloud to reduce redundancy and computational load.
6. Create Mesh
Purpose:
Generate a continuous 3D mesh from the combined point cloud to represent the brain's surface.
Process:
Estimate normals for the combined point cloud, which are essential for surface reconstruction.
Perform Poisson surface reconstruction using Open3D, which creates a mesh by solving the Poisson equation based on the point cloud's geometry.
Remove low-density vertices from the reconstructed mesh to clean up artifacts.
Simplify the mesh using quadric decimation to reduce the number of triangles while preserving essential features.
7. Save Mesh
Purpose:
Export the reconstructed 3D mesh for storage, sharing, or further analysis.
Process:
Save the simplified mesh as a .ply file (skull_model.ply) in the specified output directory using Open3D's I/O functions.
8. Visualize Mesh
Purpose:
Provide an interactive interface to explore the reconstructed 3D brain model.
Process:
Open an interactive visualization window using Open3D.
Allow users to rotate, zoom, and translate the view of the mesh using keyboard callbacks:
Right Arrow Key: Rotate the view.
W Key: Move the view forward.
S Key: Move the view backward.
Close the visualization window upon user exit.
Output

Upon successful execution, the script generates the following outputs:
3D Mesh File:
Filename: skull_model.ply
Location: Specified output_directory (default: output_files/)
Description: A Polygon File Format (PLY) file containing the reconstructed 3D mesh of the brain. This file can be viewed using various 3D visualization tools or imported into 3D modeling software for further analysis.
Interactive Visualization:
An Open3D window displaying the 3D mesh, allowing for real-time exploration and inspection of the model's geometry.
Configuration

Several parameters within the script can be customized to suit different datasets and requirements:
Input Directory (FRAME_FOLDER):
Path to the directory containing the input MRI frame images.
Default: "input_files/8_MR_20240620_121902.068000"
Output Directory (OUTPUT_DIRECTORY):
Path to the directory where the output mesh and related files will be saved.
Default: "output_files/"
MiDaS Model Configuration:
Model Type (MODEL_TYPE): Type of MiDaS model to use (e.g., "MiDaS_small").
TorchHub Model (TORCHHUB_MODEL): Repository from which to load the MiDaS model ("intel-isl/MiDaS").
Number of Frames (NUM_FRAMES):
Specifies how many frames to load and process from the input directory.
Default: 70
Camera Intrinsic Matrix:
Focal Lengths (fx, fy): Focal lengths in pixels.
Principal Point (cx, cy): Typically set to the image center.
Note: These values should ideally be obtained from camera calibration data for accurate 3D reconstruction.
Mesh Reconstruction Parameters:
Poisson Depth (depth): Depth parameter for Poisson surface reconstruction (default: 9).
Simplification Target: Number of triangles to reduce the mesh complexity (default: 10000).
Troubleshooting

Common Issues and Solutions
Frames Not Loading Correctly:
Symptom: Warnings indicating that certain frames could not be loaded.
Solution:
Verify that all frame images exist in the specified input directory.
Ensure that frame filenames follow the frame_XXXX.png format.
Check for file corruption or incompatible image formats.
MiDaS Model Failing to Load:
Symptom: Errors related to loading the MiDaS model from TorchHub.
Solution:
Ensure an active internet connection for TorchHub to download the model.
Verify that the torch library is correctly installed and updated.
Check compatibility between PyTorch and CUDA versions if using a GPU.
Insufficient Memory:
Symptom: The script crashes or becomes unresponsive during depth estimation or mesh reconstruction.
Solution:
Reduce the number of frames (NUM_FRAMES) being processed.
Downsample frames or reduce the resolution of input images.
Ensure that your system has sufficient RAM and GPU memory.

