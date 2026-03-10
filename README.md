# Cell3DStress
Cell3DStress is an image analysis tool used to calculate 3D displacement fields, strain fields and stress fields from cell images in the CZI format. This project is mainly applied in biomechanics research, helping researchers quantify the deformation and stress distribution of cells under mechanical loading.


Attention:
1.Before running the code, convert the CZI file into a tiff file through figi.
2.If the file contains four-dimensional data, compared to the case of five-dimensional data where there are fewer channels, the load_data function needs to be modified.

```python
def load_data(file_path):
    stack = tifffile.imread(file_path)
    data = stack
    return data
```

3.
