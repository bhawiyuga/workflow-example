# NDVI Clustering

This module performs KMeans clustering on NDVI (Normalized Difference Vegetation Index) raster data to identify different vegetation patterns or zones. It uses scikit-learn's K-means implementation to group similar NDVI values into clusters that may represent different vegetation types or land cover.

## Features

- Load and process NDVI GeoTIFF raster data
- Apply K-means clustering to identify vegetation patterns
- Apply sieve filtering to remove small isolated clusters
- Save clustered results as GeoTIFF
- Generate visualizations of clustering results
- Interpret clusters based on NDVI values

## Requirements

- Python 3.8+
- scikit-learn
- numpy
- matplotlib
- rasterio
- geopandas

You can install the required packages by running:

```bash
pip install scikit-learn numpy matplotlib rasterio geopandas
```

Or add the following to your requirements.txt:
```
scikit-learn==1.1.3
```

## Usage

### Command Line Interface

```bash
# Basic usage
python ndvi_clustering.py --input path/to/ndvi.tif --output path/to/clustered_ndvi.tif

# Specify number of clusters
python ndvi_clustering.py --input path/to/ndvi.tif --output path/to/clustered_ndvi.tif --n-clusters 5

# Create visualization
python ndvi_clustering.py --input path/to/ndvi.tif --output path/to/clustered_ndvi.tif --visualize

# Full options
python ndvi_clustering.py --input path/to/ndvi.tif --output path/to/clustered_ndvi.tif --n-clusters 5 --min-size 10 --visualize --vis-output path/to/clusters.png
```

### Python Module

You can also use the module in your Python scripts:

```python
from ndvi_clustering import (load_ndvi_raster, preprocess_for_clustering,
                           apply_kmeans, create_clustered_image,
                           apply_sieve_filter, save_clustered_image)

# Load NDVI raster
ndvi_array, profile = load_ndvi_raster("path/to/ndvi.tif")

# Preprocess data
features, mask = preprocess_for_clustering(ndvi_array)

# Apply K-means clustering
cluster_labels, cluster_centers = apply_kmeans(features, n_clusters=5)

# Create clustered image
clustered_image = create_clustered_image(ndvi_array, mask, cluster_labels, n_clusters=5)

# Apply sieve filter
filtered_image = apply_sieve_filter(clustered_image, min_size=10)

# Save clustered image
save_clustered_image(filtered_image, profile, "path/to/clustered_ndvi.tif")
```

### Jupyter Notebook

For interactive exploration and analysis, use the provided Jupyter notebook `ndvi_clustering.ipynb`.

## Interpreting Clusters

The clusters represent areas with similar NDVI values. Typical NDVI value ranges and their interpretations:

- < -0.1: Water bodies or shadows
- -0.1 to 0.2: Bare soil, urban areas, or very sparse vegetation
- 0.2 to 0.4: Sparse vegetation, grassland
- 0.4 to 0.6: Moderate vegetation, cropland
- > 0.6: Dense vegetation, forest

## Workflow

1. The NDVI raster data is loaded from a GeoTIFF file
2. Valid NDVI values are extracted and standardized
3. K-means clustering is applied to group similar NDVI values
4. The cluster labels are mapped back to the original raster shape
5. A sieve filter is applied to remove small isolated clusters
6. The clustered image is saved as a GeoTIFF
7. Optional visualizations are created to interpret the results
