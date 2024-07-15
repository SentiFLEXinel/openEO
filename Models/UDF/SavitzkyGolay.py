
import numpy as np
import xarray as xr
from scipy.signal import savgol_filter
from openeo.udf import XarrayDataCube

init_xr = xr.DataArray()
def apply_datacube(cube: XarrayDataCube, context: dict) -> XarrayDataCube:

    array: xarray.DataArray = cube.get_array()
    filled = array.interpolate_na(dim='t')
    smoothed_array = savgol_filter(filled.values, 5, 2, axis=0)
    return XarrayDataCube(
        array=xarray.DataArray(smoothed_array, dims=array.dims, coords=array.coords)
    )

    # # filled = cube.interpolate_na(dim='t')
    # filled_array = cube.values
    # # smoothed_array = savgol_filter(filled_array, 5, 2, axis=0)

    # init_xr = filled_array
    # returned = xr.DataArray(init_xr)
    # return returned
