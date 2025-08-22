#!/usr/bin/env python3
"""
Test script for NDVI clustering module

This script provides a basic test for the NDVI clustering module using a sample NDVI GeoTIFF file.
It generates a synthetic NDVI raster if no input file is provided.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.transform import from_bounds
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_ndvi(output_path, width=500, height=500, nodata_value=-9999):
    """
    Create a synthetic NDVI raster for testing.
    
    Parameters:
    -----------
    output_path : str
        Path to save the synthetic NDVI raster
    width, height : int
        Dimensions of the raster
    nodata_value : int or float
        Value to use for nodata
    """
    logger.info(f"Creating synthetic NDVI raster with dimensions {width}x{height}")
    
    # Create base array with random NDVI values between -1 and 1
    np.random.seed(42)  # For reproducibility
    ndvi = np.random.uniform(-1, 1, (height, width)).astype(np.float32)
    
    # Create distinct regions with different NDVI values
    # Region 1: Water (-0.5 to -0.2)
    mask1 = np.zeros((height, width), dtype=bool)
    mask1[50:150, 50:200] = True
    ndvi[mask1] = np.random.uniform(-0.5, -0.2, np.sum(mask1))
    
    # Region 2: Bare soil (0.0 to 0.2)
    mask2 = np.zeros((height, width), dtype=bool)
    mask2[200:300, 50:200] = True
    ndvi[mask2] = np.random.uniform(0.0, 0.2, np.sum(mask2))
    
    # Region 3: Sparse vegetation (0.2 to 0.4)
    mask3 = np.zeros((height, width), dtype=bool)
    mask3[50:200, 250:450] = True
    ndvi[mask3] = np.random.uniform(0.2, 0.4, np.sum(mask3))
    
    # Region 4: Moderate vegetation (0.4 to 0.6)
    mask4 = np.zeros((height, width), dtype=bool)
    mask4[250:400, 250:400] = True
    ndvi[mask4] = np.random.uniform(0.4, 0.6, np.sum(mask4))
    
    # Region 5: Dense vegetation (0.6 to 0.9)
    mask5 = np.zeros((height, width), dtype=bool)
    mask5[350:450, 100:200] = True
    ndvi[mask5] = np.random.uniform(0.6, 0.9, np.sum(mask5))
    
    # Add some random noise
    ndvi += np.random.normal(0, 0.05, (height, width))
    
    # Ensure values are within -1 to 1
    ndvi = np.clip(ndvi, -1, 1)
    
    # Add some nodata areas
    mask_nodata = np.zeros((height, width), dtype=bool)
    mask_nodata[0:50, 0:50] = True
    mask_nodata[450:500, 450:500] = True
    ndvi[mask_nodata] = nodata_value
    
    # Create a transform (using a simple geographic projection)
    bounds = (0, 0, 10, 10)  # Simple bounds (xmin, ymin, xmax, ymax)
    transform = from_bounds(*bounds, width, height)
    
    # Save the synthetic NDVI raster
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.float32,
        crs='EPSG:4326',
        transform=transform,
        nodata=nodata_value
    ) as dst:
        dst.write(ndvi, 1)
    
    logger.info(f"Synthetic NDVI raster saved to {output_path}")
    
    # Create a visualization of the synthetic NDVI
    vis_path = output_path.replace('.tif', '_visualization.png')
    plt.figure(figsize=(10, 8))
    masked_ndvi = np.ma.masked_where(ndvi == nodata_value, ndvi)
    plt.imshow(masked_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('Synthetic NDVI')
    plt.axis('off')
    plt.savefig(vis_path, dpi=300)
    plt.close()
    
    logger.info(f"Synthetic NDVI visualization saved to {vis_path}")
    
    return output_path

def test_ndvi_clustering(input_path=None):
    """
    Test the NDVI clustering module.
    
    Parameters:
    -----------
    input_path : str
        Path to the NDVI GeoTIFF file to use for testing
    """
    # Import the NDVI clustering module
    try:
        from ndvi_clustering import (load_ndvi_raster, preprocess_for_clustering,
                                   apply_kmeans, create_clustered_image,
                                   apply_sieve_filter, save_clustered_image,
                                   create_cluster_visualization)
    except ImportError as e:
        logger.error(f"Failed to import NDVI clustering module: {str(e)}")
        logger.error("Make sure the module is correctly installed and all dependencies are met.")
        return False
    
    # Generate synthetic NDVI raster if no input path is provided
    if input_path is None:
        output_dir = "test_data"
        os.makedirs(output_dir, exist_ok=True)
        input_path = os.path.join(output_dir, "synthetic_ndvi.tif")
        create_synthetic_ndvi(input_path)
    
    try:
        # Load NDVI raster
        ndvi_array, profile = load_ndvi_raster(input_path)
        logger.info(f"NDVI data loaded with shape {ndvi_array.shape}")
        
        # Preprocess data
        features, mask = preprocess_for_clustering(ndvi_array)
        logger.info(f"Preprocessed {features.shape[0]} valid NDVI points")
        
        # Apply K-means clustering
        n_clusters = 5
        cluster_labels, cluster_centers = apply_kmeans(features, n_clusters=n_clusters)
        logger.info(f"K-means clustering completed with {n_clusters} clusters")
        
        # Create clustered image
        clustered_image = create_clustered_image(ndvi_array, mask, cluster_labels, n_clusters)
        logger.info("Clustered image created")
        
        # Apply sieve filter
        min_size = 10
        filtered_image = apply_sieve_filter(clustered_image, min_size=min_size)
        logger.info(f"Sieve filter applied with minimum size {min_size}")
        
        # Save clustered image
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "clustered_ndvi.tif")
        save_clustered_image(filtered_image, profile, output_path)
        logger.info(f"Clustered image saved to {output_path}")
        
        # Create visualization
        vis_output = os.path.join(output_dir, "ndvi_clusters_visualization.png")
        create_cluster_visualization(filtered_image, cluster_centers, vis_output, original_ndvi=ndvi_array)
        logger.info(f"Visualization saved to {vis_output}")
        
        logger.info("NDVI clustering test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"NDVI clustering test failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Test NDVI clustering module.')
    parser.add_argument('--input', type=str, 
                      help='Path to the NDVI GeoTIFF file to use for testing (optional, will generate synthetic data if not provided)')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Test NDVI clustering
    success = test_ndvi_clustering(input_path=args.input)
    
    if success:
        logger.info("Test completed successfully")
    else:
        logger.error("Test failed")
