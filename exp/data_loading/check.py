import rasterio
import numpy as np
arr = rasterio.open("datasets/sen1floods11_v1.1/data/JRCWaterHand/Bolivia_23014_JRCWaterHand.tif").read()
print(arr.min(), arr.max(), arr.dtype)