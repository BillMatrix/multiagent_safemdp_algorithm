from osgeo import gdal
from scipy import interpolate
import numpy as np
import os
import matplotlib.pyplot as plt
import GPy


__all__ = ['mars_map', 'initialize_SafeMDP_object', 'performance_metrics']


def mars_map(world_shape, plot_map=False, interpolation=False):
    """
    Extract the map for the simulation from the HiRISE data. If the HiRISE
    data is not in the current folder it will be downloaded and converted to
    GeoTiff extension with gdal.
    Parameters
    ----------
    plot_map: bool
        If true plots the map that will be used for exploration
    interpolation: bool
        If true the data of the map will be interpolated with splines to
        obtain a finer grid
    Returns
    -------
    altitudes: np.array
        1-d vector with altitudes for each node
    coord: np.array
        Coordinate of the map we use for exploration
    world_shape: tuple
        Size of the grid world (rows, columns)
    step_size: tuple
        Step size for the grid (row, column)
    num_of_points: int
        Interpolation parameter. Indicates the scaling factor for the
        original step size
    """

    # Define the dimension of the map we want to investigate and its resolution
    step_size = (1., 1.)

    # Download and convert to GEOtiff Mars data
    if not os.path.exists('./mars.tif'):
        if not os.path.exists("./mars.IMG"):
            import urllib

            print('Downloading MARS map, this make take a while...')
            # Download the IMG file
            urllib.request.urlretrieve(
                "http://www.uahirise.org/PDS/DTM/PSP/ORB_010200_010299"
                "/PSP_010228_1490_ESP_016320_1490"
                "/DTEEC_010228_1490_016320_1490_A01.IMG", "mars.IMG")

        # Convert to tif
        print('Converting map to geotif...')
        os.system("gdal_translate -of GTiff ./mars.IMG ./mars.tif")
        print('Done')

    # Read the data with gdal module
    gdal.UseExceptions()
    ds = gdal.Open("./mars.tif")
    band = ds.GetRasterBand(1)
    elevation = band.ReadAsArray()

    # Extract the area of interest
    startX = 2890
    startY = 1955
    altitudes = np.copy(elevation[startX:startX + world_shape[0],
                        startY:startY + world_shape[1]])

    # Center the data
    mean_val = (np.max(altitudes) + np.min(altitudes)) / 2.
    altitudes[:] = altitudes - mean_val

    # Define coordinates
    n, m = world_shape
    step1, step2 = step_size
    xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                         np.linspace(0, (m - 1) * step2, m), indexing="ij")
    coord = np.vstack((xx.flatten(), yy.flatten())).T

    # Interpolate data
    if interpolation:

        # Interpolating function
        spline_interpolator = interpolate.RectBivariateSpline(
            np.linspace(0, (n - 1) * step1, n),
            np.linspace(0, (m - 1) * step1, m), altitudes)

        # New size and resolution
        num_of_points = 1
        world_shape = tuple([(x - 1) * num_of_points + 1 for x in world_shape])
        step_size = tuple([x / num_of_points for x in step_size])

        # New coordinates and altitudes
        n, m = world_shape
        step1, step2 = step_size
        xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                             np.linspace(0, (m - 1) * step2, m), indexing="ij")
        coord = np.vstack((xx.flatten(), yy.flatten())).T

        altitudes = spline_interpolator(np.linspace(0, (n - 1) * step1, n),
                                        np.linspace(0, (m - 1) * step2, m))
    else:
        num_of_points = 1

    # Plot area
    if plot_map:
        plt.imshow(altitudes.T, origin="lower", interpolation="nearest")
        plt.colorbar()
        plt.show()

    return altitudes
