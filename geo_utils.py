import numpy as np

def check_geo_dim_convention(x):
    assert x.shape[0] == 3
    return x

def cart2sph(xyz, deg=True):
    check_geo_dim_convention(xyz)
    xy = xyz[0, :]**2 + xyz[1, :]**2
    r = np.sqrt(xy + xyz[2, :]**2)
    el = np.deg2rad(90) - np.arctan2(np.sqrt(xy), xyz[2, :])
    az = np.arctan2(xyz[1, :], xyz[0, :])
    if deg:
        el = np.rad2deg(el)
        az = np.rad2deg(az)
    razel = np.array([r, az, el])
    return check_geo_dim_convention(razel)


def sph2cart(razel, deg=True):
    check_geo_dim_convention(razel)
    r, az, el = razel[0, :], razel[1, :], razel[2, :]
    if deg:
        el = np.deg2rad(el)
        az = np.deg2rad(az)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    xyz = np.array([x, y, z])
    return check_geo_dim_convention(xyz)
