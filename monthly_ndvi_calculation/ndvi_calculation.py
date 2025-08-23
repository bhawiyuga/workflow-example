#!/usr/bin/env python3
"""
NDVI Calculation Script

This script calculates the Normalized Difference Vegetation Index (NDVI) for a specific area
within a given time range using STAC and odc-stac. It performs cloud masking using the SCL band
and stores the output locally.

Usage:
    python ndvi_calculation.py --bbox 107.5 -7.0 107.7 -6.8 --time-range 2023-01-01 2023-01-31 \
        --resolution 100 --max-cloud-cover 80 --dask-client tcp://192.168.10.1:8786 \
        --chunks 549 549 30
"""

import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from datetime import datetime
from pystac_client import Client
from odc.stac import stac_load
import xarray as xr
from rasterio.transform import from_bounds
from dask.distributed import Client as DaskClient
import geopandas as gpd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def geojson_to_bbox(geojson_path):
    """
    Convert a GeoJSON file to a bounding box [min_x, min_y, max_x, max_y]
    
    Parameters:
    -----------
    geojson_path : str
        Path to the GeoJSON file
        
    Returns:
    --------
    list
        Bounding box coordinates [min_x, min_y, max_x, max_y]
    """
    
    # Read the GeoJSON file
    gdf = gpd.read_file(geojson_path)
    
    # Get the total bounds of all geometries
    minx, miny, maxx, maxy = gdf.total_bounds
    
    # Return as a list [min_x, min_y, max_x, max_y]
    return [minx, miny, maxx, maxy]

def mask_clouds(dataset):
    """
    Mask clouds using the Scene Classification Layer (SCL) band.
    
    SCL values:
    1: Saturated or defective
    2: Dark area pixels
    3: Cloud shadows
    4: Vegetation
    5: Bare soils
    6: Water
    7: Clouds low probability / Unclassified
    8: Clouds medium probability
    9: Clouds high probability
    10: Cirrus
    11: Snow / ice
    """
    if 'scl' not in dataset:
        logger.warning("SCL band not found in dataset. Cloud masking skipped.")
        return dataset
    
    # Create a mask for cloud-free pixels (not in [1, 3, 7, 8, 9, 10])
    cloud_values = [1, 3, 7, 8, 9, 10]
    cloud_mask = np.isin(dataset.scl.values, cloud_values, invert=True)
    
    # Apply the mask to all bands
    masked_dataset = dataset.copy()
    for band in dataset.data_vars:
        if band != 'scl':
            masked_dataset[band] = dataset[band].where(cloud_mask)
    
    return masked_dataset

def save_ndvi_tiff(ndvi_array, output_path):
    """
    Save NDVI array as a GeoTIFF file.
    """
    ndvi_array.odc.write_cog(output_path, overwrite=True)
    logger.info(f"NDVI saved to {output_path}")

def save_ndvi_plot(ndvi_array, output_path):
    """
    Create and save a visualization of the NDVI data.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(ndvi_array, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('NDVI')
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"NDVI visualization saved to {output_path}")

def calculate_area_ndvi(
    dataset,
    output_dir="./output",
):
    """
    Calculate NDVI for a specific area and time range.
    
    Parameters:
    -----------
    dataset : xarray.Dataset
        Dataset containing the Sentinel-2 data
    output_dir : str
        Directory to save output files
    """
    # Apply cloud masking
    logger.info("Applying cloud masking...")
    masked_dataset = mask_clouds(dataset)
    
    # Calculate NDVI for each time step
    logger.info("Calculating NDVI...")
    
    # Create a new DataArray for NDVI
    ndvi_data = (masked_dataset.red - masked_dataset.nir) / (masked_dataset.red + masked_dataset.nir)

    # Calculate monthly max NDVI
    monthly_max_ndvi = ndvi_data.groupby("time.month").max()
    
    return monthly_max_ndvi

def search_catalog(bbox, time_range, stac_api_url="https://earth-search.aws.element84.com/v1", collection="sentinel-2-l2a",max_cloud_cover=20):
    catalog = Client.open(stac_api_url)
    
    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=f"{time_range[0]}/{time_range[1]}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}}
    )
    
    items = search.item_collection()
    if not items:
        logger.error(f"No items found for the specified parameters.")
        return
    
    logger.info(f"Found {len(items)} items. Loading data...")
    
    return items

def load_dataset(items, resolution=100, chunks=None, groupby="solar_day"):
    kwargs = {
        'bands': ["red", "nir", "scl"],
        'resolution': resolution,
        'groupby': groupby,
    }
    
    # Only add chunks parameter if provided
    if chunks:
        kwargs['chunks'] = chunks
    
    dataset = stac_load(
        items,
        **kwargs
    )
    
    return dataset

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description='Calculate NDVI for a specific area and time range.')
    
    # Required arguments
    parser.add_argument('--bbox', type=float, nargs=4, required=False,
                        default=[107.5, -7.0, 107.7, -6.8],
                        help='Bounding box coordinates [min_lon, min_lat, max_lon, max_lat]')
    parser.add_argument('--time-range', type=str, nargs=2, required=False,
                        default=["2023-01-01", "2023-01-31"],
                        help='Time range in format YYYY-MM-DD YYYY-MM-DD')
    
    # Optional arguments
    parser.add_argument('--resolution', type=int, default=100,
                        help='Spatial resolution in meters')
    parser.add_argument('--max-cloud-cover', type=int, default=80,
                        help='Maximum cloud cover percentage')
    parser.add_argument('--dask-scheduler', type=str,
                        help='Dask scheduler address (optional, e.g., tcp://192.168.10.1:8786)')
    parser.add_argument('--stac-api-url', type=str, 
                        default="https://earth-search.aws.element84.com/v1",
                        help='STAC API URL')
    parser.add_argument('--collection', type=str, default="sentinel-2-l2a",
                        help='STAC collection name')
    parser.add_argument('--chunks', type=int, nargs=3,
                        help='Chunk sizes [x, y, time] (optional)')
    parser.add_argument('--groupby', type=str, default="solar_day",
                        help='Groupby parameter for stac_load')
    parser.add_argument('--output', type=str, default="monthly_max_ndvi.tif",
                        help='Output filename')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Connect to Dask client if scheduler is provided
    client = None
    if args.dask_scheduler:
        logger.info(f"Connecting to Dask client at {args.dask_scheduler}...")
        client = DaskClient(args.dask_scheduler)
        logger.info("Connected to Dask client.")
    else:
        logger.info("No Dask scheduler provided, running in local mode.")
    
    # Define area of interest (bbox: [min_lon, min_lat, max_lon, max_lat])
    bbox = args.bbox
    logger.info(f"Using bounding box: {bbox}")
    
    # Define time range
    time_range = tuple(args.time_range)
    logger.info(f"Using time range: {time_range}")

    # Max cloud cover
    max_cloud_cover = args.max_cloud_cover
    logger.info(f"Using max cloud cover: {max_cloud_cover}%")

    # Resolution
    resolution = args.resolution
    logger.info(f"Using resolution: {resolution}m")

    # Chunks
    chunks = None
    if args.chunks:
        chunks = {'x': args.chunks[0], 'y': args.chunks[1], 'time': args.chunks[2]}
        logger.info(f"Using chunks: {chunks}")
    else:
        logger.info("No chunks specified, using default chunking.")
    
    # Search catalog
    items = search_catalog(
        bbox, 
        time_range, 
        stac_api_url=args.stac_api_url,
        collection=args.collection,
        max_cloud_cover=max_cloud_cover
    )

    # Load dataset
    dataset = load_dataset(
        items, 
        resolution=resolution, 
        chunks=chunks, 
        groupby=args.groupby
    )
    
    # Calculate NDVI
    monthly_max_ndvi = calculate_area_ndvi(dataset)
    
    # Store results
    # Get script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set output directory
    output_path = args.output if args.output else os.path.join(script_dir, "output", "monthly_max_ndvi.tif")
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save NDVI as COG
    logger.info(f"Saving NDVI to {output_path}")
    monthly_max_ndvi.odc.write_cog(output_path, overview_levels=[], overwrite=True)
