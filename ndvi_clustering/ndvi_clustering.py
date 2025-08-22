#!/usr/bin/env python3
"""
NDVI Clustering Script

This script performs K-means clustering on NDVI GeoTIFF data to identify vegetation patterns.
It reads NDVI raster data, applies K-means clustering, and saves the clustered results as GeoTIFF files.

Usage:
    python ndvi_clustering.py --input monthly_max_ndvi.tif --output clustered_ndvi.tif --n-clusters 5
"""

import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import rasterio
from rasterio.features import sieve
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from pathlib import Path
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_ndvi_raster(file_path):
    """
    Load NDVI raster file into a numpy array.
    
    Parameters:
    -----------
    file_path : str
        Path to the NDVI GeoTIFF file
        
    Returns:
    --------
    tuple
        (ndvi_array, profile) - NDVI data as numpy array and rasterio profile
    """
    logger.info(f"Loading NDVI data from {file_path}")
    
    with rasterio.open(file_path) as src:
        ndvi_array = src.read(1)
        profile = src.profile.copy()
        
    logger.info(f"NDVI data loaded with shape {ndvi_array.shape}")
    
    return ndvi_array, profile

def preprocess_for_clustering(ndvi_array):
    """
    Preprocess NDVI data for clustering.
    
    Parameters:
    -----------
    ndvi_array : numpy.ndarray
        NDVI data array
        
    Returns:
    --------
    tuple
        (features, mask) - Preprocessed features for clustering and valid data mask
    """
    logger.info("Preprocessing NDVI data for clustering")
    
    # Create mask for valid NDVI values (between -1 and 1, not NaN)
    mask = ~np.isnan(ndvi_array) & (ndvi_array >= -1) & (ndvi_array <= 1)
    
    # Extract valid NDVI values for clustering
    valid_ndvi = ndvi_array[mask]
    
    # Reshape to 2D array expected by scikit-learn
    features = valid_ndvi.reshape(-1, 1)
    
    # Scale features to standardize for k-means
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    logger.info(f"Preprocessed {features.shape[0]} valid NDVI points")
    
    return scaled_features, mask

def apply_kmeans(features, n_clusters=5, random_state=42):
    """
    Apply K-means clustering to NDVI data.
    
    Parameters:
    -----------
    features : numpy.ndarray
        Preprocessed NDVI features
    n_clusters : int
        Number of clusters to identify
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    numpy.ndarray
        Cluster labels for each feature
    """
    logger.info(f"Applying K-means clustering with {n_clusters} clusters")
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(features)
    
    logger.info(f"K-means clustering completed")
    
    return cluster_labels, kmeans.cluster_centers_

def create_clustered_image(ndvi_array, mask, cluster_labels, n_clusters=5):
    """
    Create a clustered image from cluster labels.
    
    Parameters:
    -----------
    ndvi_array : numpy.ndarray
        Original NDVI array
    mask : numpy.ndarray
        Boolean mask of valid NDVI values
    cluster_labels : numpy.ndarray
        Cluster labels from K-means
    n_clusters : int
        Number of clusters
        
    Returns:
    --------
    numpy.ndarray
        Clustered image with same shape as original NDVI array
    """
    logger.info("Creating clustered image")
    
    # Create an output array with same shape as input, filled with nodata (-9999)
    clustered_image = np.full_like(ndvi_array, -9999, dtype=np.int16)
    
    # Assign cluster labels to the valid pixels in the output array
    clustered_image[mask] = cluster_labels
    
    # Set invalid pixels to nodata value
    clustered_image[~mask] = -9999
    
    logger.info("Clustered image created")
    
    return clustered_image

def apply_sieve_filter(clustered_image, min_size=10):
    """
    Apply a sieve filter to remove small isolated clusters.
    
    Parameters:
    -----------
    clustered_image : numpy.ndarray
        Clustered image
    min_size : int
        Minimum size of cluster to keep
        
    Returns:
    --------
    numpy.ndarray
        Filtered clustered image
    """
    logger.info(f"Applying sieve filter with minimum size {min_size}")
    
    # Create a mask for valid data
    mask = clustered_image != -9999
    
    # Apply sieve filter to each cluster separately
    filtered_image = clustered_image.copy()
    
    # Process only if we have valid data
    if np.any(mask):
        sieved = sieve(clustered_image, min_size)
        filtered_image[mask] = sieved[mask]
    
    logger.info("Sieve filtering completed")
    
    return filtered_image

def save_clustered_image(clustered_image, profile, output_path):
    """
    Save clustered image as GeoTIFF.
    
    Parameters:
    -----------
    clustered_image : numpy.ndarray
        Clustered image
    profile : dict
        Rasterio profile from the original NDVI raster
    output_path : str
        Output file path
    """
    logger.info(f"Saving clustered image to {output_path}")
    
    # Update profile for clustered image
    profile.update(
        dtype=np.int16,
        count=1,
        nodata=-9999,
        compress='lzw'
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as GeoTIFF
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(clustered_image, 1)
    
    logger.info(f"Clustered image saved to {output_path}")

def create_cluster_visualization(clustered_image, cluster_centers, output_path, original_ndvi=None):
    """
    Create a visualization of the clusters.
    
    Parameters:
    -----------
    clustered_image : numpy.ndarray
        Clustered image
    cluster_centers : numpy.ndarray
        Centers of clusters from K-means
    output_path : str
        Output file path for visualization
    original_ndvi : numpy.ndarray, optional
        Original NDVI array for additional visualization
    """
    logger.info(f"Creating cluster visualization to {output_path}")
    
    # Number of clusters
    n_clusters = len(cluster_centers)
    
    # Create a colormap for clusters
    cmap = plt.cm.get_cmap('viridis', n_clusters)
    
    # Create figure
    fig, axs = plt.subplots(1, 2 if original_ndvi is not None else 1, figsize=(16, 8))
    
    # Sort clusters by their NDVI value for consistent coloring and numbering
    sorted_centers = sorted([(i, centers[0]) for i, centers in enumerate(cluster_centers)], 
                           key=lambda x: x[1])
    
    # Create a mapping from original cluster IDs to ordered numbers (0 to n-1)
    cluster_to_number = {cluster_id: i for i, (cluster_id, _) in enumerate(sorted_centers)}
    
    # Create a numbered version of the clustered image for display
    numbered_image = np.full_like(clustered_image, -9999)
    for orig_id, new_id in cluster_to_number.items():
        numbered_image[clustered_image == orig_id] = new_id
    
    # Get unique values for legend labels (excluding nodata)
    cluster_values = np.unique(numbered_image[numbered_image != -9999])
    
    if original_ndvi is not None:
        # Plot original NDVI
        im1 = axs[0].imshow(original_ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        axs[0].set_title('Original NDVI')
        plt.colorbar(im1, ax=axs[0])
        
        # Plot clustered image with numeric legend
        masked_clusters = np.ma.masked_where(numbered_image == -9999, numbered_image)
        im2 = axs[1].imshow(masked_clusters, cmap=cmap, vmin=0, vmax=n_clusters-1)
        axs[1].set_title('NDVI Clusters')
        
        # Create custom colorbar with integer ticks
        cbar = plt.colorbar(im2, ax=axs[1], ticks=range(n_clusters))
        cbar.set_label('Cluster Number')
        cbar.set_ticklabels(range(n_clusters))
    else:
        # Plot only clustered image
        masked_clusters = np.ma.masked_where(numbered_image == -9999, numbered_image)
        im = axs.imshow(masked_clusters, cmap=cmap, vmin=0, vmax=n_clusters-1)
        axs.set_title('NDVI Clusters')
        
        # Create custom colorbar with integer ticks
        cbar = plt.colorbar(im, ax=axs, ticks=range(n_clusters))
        cbar.set_label('Cluster Number')
        cbar.set_ticklabels(range(n_clusters))
    
    # Add cluster information with NDVI ranges
    cluster_info_lines = []
    
    # Process clusters in sorted order (low to high NDVI)
    for new_id, (orig_id, _) in enumerate(sorted_centers):
        # Get mask for current cluster using the original IDs
        cluster_mask = (clustered_image == orig_id) & (clustered_image != -9999)
        if np.any(cluster_mask):
            # Extract NDVI values for this cluster
            ndvi_values = original_ndvi[cluster_mask] if original_ndvi is not None else []
            
            if len(ndvi_values) > 0:
                min_ndvi = np.min(ndvi_values)
                max_ndvi = np.max(ndvi_values)
                mean_ndvi = np.mean(ndvi_values)
                count = np.sum(cluster_mask)
                
                # Format the cluster info (using the new, sorted ID)
                cluster_info_lines.append(
                    f"Cluster {new_id}: NDVI {mean_ndvi:.2f} (range: {min_ndvi:.2f} to {max_ndvi:.2f}), pixels: {count}"
                )
            else:
                # Fallback to cluster center if we can't get the range
                cluster_info_lines.append(f"Cluster {new_id}: NDVI ~{cluster_centers[orig_id][0]:.2f}")
        else:
            # Fallback to cluster center if cluster not found in image
            cluster_info_lines.append(f"Cluster {new_id}: NDVI ~{cluster_centers[orig_id][0]:.2f}")
    
    # Join all lines
    cluster_info = "\n".join(cluster_info_lines)
    
    # Add the text to the figure
    plt.figtext(0.02, 0.02, cluster_info, wrap=True, fontsize=9, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Cluster visualization saved to {output_path}")

    # return plt

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Perform K-means clustering on NDVI raster data.')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=False,
                        default="../monthly_ndvi_calculation/output/monthly_max_ndvi.tif",
                        help='Path to input NDVI GeoTIFF file')
    
    # Optional arguments
    parser.add_argument('--output', type=str, default="output/clustered_ndvi.tif",
                        help='Path to output clustered NDVI GeoTIFF file')
    parser.add_argument('--n-clusters', type=int, default=5,
                        help='Number of clusters for K-means')
    parser.add_argument('--min-size', type=int, default=10,
                        help='Minimum size of clusters for sieve filtering')
    parser.add_argument('--visualize', action='store_true',
                        help='Create a visualization of the clusters')
    parser.add_argument('--vis-output', type=str, default="output/ndvi_clusters.png",
                        help='Path to output visualization file')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Load NDVI raster
    ndvi_array, profile = load_ndvi_raster(args.input)
    
    # Preprocess data
    features, mask = preprocess_for_clustering(ndvi_array)
    
    # Apply K-means clustering
    cluster_labels, cluster_centers = apply_kmeans(features, n_clusters=args.n_clusters)
    
    # Create clustered image
    clustered_image = create_clustered_image(ndvi_array, mask, cluster_labels, args.n_clusters)
    
    # Apply sieve filter
    filtered_image = apply_sieve_filter(clustered_image, min_size=args.min_size)
    
    # Save clustered image
    save_clustered_image(filtered_image, profile, args.output)
    
    # Create visualization if requested
    if args.visualize:
        create_cluster_visualization(filtered_image, cluster_centers, 
                                     args.vis_output, original_ndvi=ndvi_array)
        logger.info(f"Visualization saved to {args.vis_output}")
    
    logger.info("NDVI clustering completed successfully")
