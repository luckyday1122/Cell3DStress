# Cell3DStress
Cell3DStress is an image analysis tool used to calculate 3D displacement fields, strain fields and stress fields from cell images in the CZI format. This project is mainly applied in biomechanics research, helping researchers quantify the deformation and stress distribution of cells under mechanical loading.


# Attention:

1.Before running the code, convert the CZI file into a tiff file through figi.

2.If the file contains four-dimensional data, compared to the case of five-dimensional data where there are fewer channels, the load_data function needs to be modified.

```python
def load_data(file_path):
    stack = tifffile.imread(file_path)
    data = stack
    return data
```

3.The default position of the cells is at the center. If the cells are not located at the center, you can manually adjust the code. Alternatively, you can modify the detect_cell_region_3d function.

```python
ef detect_cell_region_3d(volume, top_percentile=95, min_size_ratio=0.05):
    """3D Cell Region Detection Based on Relative Brightness"""
   
    threshold = np.percentile(volume, top_percentile)
    
    mask = volume > threshold
    
    labeled_array, num_features = ndimage.label(mask)
    
    if num_features > 0:
        
        regions = ndimage.find_objects(labeled_array)
        max_intensities = []
        mean_intensities = []
        volumes = []
        
        for i, region in enumerate(regions):
            if region is not None:
                region_mask = labeled_array[region] == (i + 1)
                region_pixels = volume[region][region_mask]
                max_intensities.append(region_pixels.max())
                mean_intensities.append(region_pixels.mean())
                volumes.append(region_pixels.size)
        
        if volumes:  
            volumes = np.array(volumes)
            max_intensities = np.array(max_intensities)
            mean_intensities = np.array(mean_intensities)
            
 
            scores = (
                0.4 * volumes / volumes.max() + 
                0.4 * max_intensities / max_intensities.max() +
                0.2 * mean_intensities / mean_intensities.max()
            )
            
            best_idx = np.argmax(scores)
            best_region = regions[best_idx]
            
            
            z_slice, y_slice, x_slice = best_region
            x, y, z = x_slice.start, y_slice.start, z_slice.start
            w, h, d = x_slice.stop - x_slice.start, y_slice.stop - y_slice.start, z_slice.stop - z_slice.start
            
       
            return (x, y, z, w, h, d)
    depth, height, width = volume.shape
    center_ratio = 0.3
    crop_d = int(depth * center_ratio)
    crop_h = int(height * center_ratio)
    crop_w = int(width * center_ratio)
    center_x = width // 2 - crop_w // 2
    center_y = height // 2 - crop_h // 2
    center_z = depth // 2 - crop_d // 2
    return (center_x, center_y, center_z, crop_w, crop_h, crop_d)
```

4.This code mainly uses von Mises stress for plotting. However, the code also calculates other stresses, such as tensile stress.
