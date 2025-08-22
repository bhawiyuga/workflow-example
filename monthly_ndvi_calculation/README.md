# NDVI Calculation Script

This script calculates the Normalized Difference Vegetation Index (NDVI) for a specific area within a given time range using STAC and odc-stac. It performs cloud masking using the Scene Classification Layer (SCL) band and stores the output locally.

## Features

- Uses STAC to define temporal and spatial extent with filtering on cloud percentage
- Uses odc-stac to load the raster images
- Performs cloud masking by checking the SCL band
- Stores output of calculation locally in multiple formats:
  - GeoTIFF (.tif)
  - Visualization plot (.png)
  - NetCDF dataset (.nc)

## Installation

1. First, install the geobench-cloud package in development mode:

```bash
cd /home/crib/geobench-cloud
pip install -e .
```

2. Install the additional dependencies required for the NDVI calculation:

```bash
pip install -r example/ndvi_requirements.txt
```

## Usage

You can run the script directly:

```bash
python example/ndvi_calculation.py
```

### Customizing Parameters

To customize the area of interest, time range, or other parameters, modify the following section in the script:

```python
if __name__ == "__main__":
    # Define area of interest (bbox: [min_lon, min_lat, max_lon, max_lat])
    bbox = [107.5, -7.0, 107.7, -6.8]  # Example for a region in Bandung, Indonesia
    
    # Define time range
    time_range = ("2023-01-01", "2023-01-31")
    
    # Calculate NDVI
    results = calculate_area_ndvi(
        bbox=bbox,
        time_range=time_range,
        output_dir="./ndvi_output",
        max_cloud_cover=20
    )
```

## Output

The script generates three types of output files:
1. **GeoTIFF**: Contains the georeferenced NDVI data
2. **PNG**: Visualization of the NDVI map with a color scale
3. **NetCDF**: Contains the full NDVI dataset with all time steps

All files are saved in the specified output directory with timestamps in their filenames.

## Notes

- The script uses Sentinel-2 L2A data by default
- Cloud masking is performed using the Scene Classification Layer (SCL)
- NDVI values range from -1 to 1, where:
  - Values close to 1 indicate dense vegetation
  - Values close to 0 indicate no vegetation
  - Negative values typically indicate water, snow, or clouds
