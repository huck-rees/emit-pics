import os
os.environ['GDAL_SKIP'] = 'netCDF'

import ast
from datetime import datetime
import time
import random
import re
import numpy as np
import rioxarray
import s3fs
import click
import requests
import pandas
import geopandas
import shapely
from shapely import geometry
import rasterio
import xarray as xr
import earthaccess
from matplotlib import pyplot as plt
from isofit import ray
from pyproj import Transformer
from scipy.spatial import KDTree
from spectral import envi
from isofit.core.common import envi_header


auth = earthaccess.login(persist=True)
print(auth.authenticated)


def row_col(longitude, latitude, tree, locs):
    dist, n = tree.query([longitude, latitude])
    sample_lon, sample_lat = locs[n, :]
    return np.where((lon == sample_lon) & (lat == sample_lat))


def parse_result(res):
    print(get_key(res['umm'], 'BeginningDateTime'))
    # Parse time
    try:
        dt = datetime.strptime(
            get_key(res['umm'], 'BeginningDateTime'), 
            '%Y-%m-%dT%H:%M:%SZ'
        )
    except:
        pattern = '(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})'
        dt_match = re.search(pattern, get_key(res['umm'], 'BeginningDateTime'))
        dt = datetime(
            int(dt_match.group(1)),
            int(dt_match.group(2)),
            int(dt_match.group(3)),
            int(dt_match.group(4)),
            int(dt_match.group(5)),
            int(dt_match.group(6)),
        )

    # Parse URLS
    url = get_key(res['umm'], 'GranuleUR')

    # Parse polygon
    coords = []
    polygon_points = get_key(res['umm'], 'GPolygons')
    if polygon_points:
        for points in polygon_points[0]['Boundary']['Points']:
            coords.append([points['Longitude'], points['Latitude']])
        coords = np.array(coords)
        poly = geometry.Polygon(coords)

        return poly, dt, url
    
    else:
        return None, None, None


def initialize_output(output_metadata, outpath, out_shape, keys_to_del, **kwargs):
    """
    Initialize output file by updating metadata and creating object.

    Args:
        output_metadata: dict - Dictionary with envi header information
        outpath: str - path to output file
        out_shape: tuple - dimensions of initialized file
        keys_to_del: list - keys to remove from output_metadata
        kwargs - key-argument pairs to add to output_metadata
    """
    for key, value in kwargs.items():
        output_metadata[key] = value

    for key in keys_to_del:
        if key in list(output_metadata.keys()):
            del output_metadata[key]

    out_file = envi.create_image(
        envi_header(outpath), ext="", metadata=output_metadata, force=True
    )
    out_mm = out_file.open_memmap(interleave="source", writable=True)
    out_mm[:, :] = np.zeros(out_shape, dtype=np.float32)
    del out_file

    return outpath


def pull_emit_data(url, latitude, longitude):
    rio_env = rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN='TRUE',
        GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
        GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt')
    )
    rio_env.__enter__()

    ds = rioxarray.open_rasterio(url)

    norm = lambda x, vmin, vmax: (x - vmin) / (vmax - vmin)
    bands = [50, 30, 18]

    # rfl = ds[2]['reflectance'][..., bands].values

    lon = np.squeeze(ds[0]['location_lon'].data)
    lat = np.squeeze(ds[0]['location_lat'].data)
    # rfl = np.squeeze(ds[2]['reflectance'].data)

    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    locs = np.array([lon_flat, lat_flat]).T
    tree = KDTree(locs)
    dist, n = tree.query([longitude, latitude])
    sample_lon, sample_lat = locs[n, :]

    row, col = np.where((lon == sample_lon) & (lat == sample_lat))
    # sample = ds[2]['reflectance'].isel(y=row, x=col)
    data = ds[2]['reflectance'][row, col, :].values

    # Classify

    plt.plot(data[0, 0, :])

    # plt.imshow(rfl)
    # plt.scatter(col, row, facecolor='white', edgecolor='black')
    # plt.show()

    return data


def get_key(dic, key, res=None):
    if dic.get(key):
        return dic[key]

    else:
        for v in dic.values():
            if res:
                break

            elif isinstance(v, dict):
                res = get_key(v, key)

        return res


start_date = '2023-01-01'
end_date = '2026-01-01'
emit_name = 'EMITL2ARFL'
template = 'https://data.lpdaac.earthdatacloud.nasa.gov/lp-prod-protected/EMITL2ARFL.001/{}/{}.nc'

path = "/Users/bgreenbe/Projects/SpectralHydrograph/Data/emit_overlap_rivers_polys.gpkg"
df = geopandas.read_file(path)

rio_env = rasterio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN='TRUE',
    GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
    GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt')
)
rio_env.__enter__()

# Get emit urls
url_data = {
    'i': [],
    'bbox': [],
    'url': []
}
url_geoms = []
j = 0
for i, row in df.iterrows():
    print(f"{i} of {len(df)}")
    geom = row['geometry']
    if not geom:
        continue

    bbox = tuple(list(geom.bounds))

    # Get emit urls
    emit_results = earthaccess.search_data(
        short_name=emit_name,
        bounding_box=bbox,
        temporal=(start_date, end_date),
        cloud_cover=(0, 20),
        count=-1
    )

    for res in emit_results:
        poly, dt, emit_url = parse_result(res)
        url = template.format(emit_url, emit_url)
        url_data['i'].append(j)
        url_data['bbox'].append(bbox)
        url_data['url'].append(url)
        url_geoms.append(geom)
        j += 1

url_df = pandas.DataFrame(data=url_data)
url_df = geopandas.GeoDataFrame(url_df, geometry=url_geoms)
out_gpkg = "/Users/bgreenbe/Projects/SpectralHydrograph/Data/emit_river_urls.gpkg"
# url_df.to_file(out_gpkg, driver="GPKG")

url_df = df = geopandas.read_file(out_gpkg)

wl = np.loadtxt('/Users/bgreenbe/Data/EMIT/Surfaces/wavelengths.txt')
bands = [
    np.argmin(np.abs(wl[:, 1] - .650)),
    np.argmin(np.abs(wl[:, 1] - .540)),
    np.argmin(np.abs(wl[:, 1] - .460)),
]
rio_env = rasterio.Env(
    GDAL_DISABLE_READDIR_ON_OPEN='TRUE',
    GDAL_HTTP_COOKIEFILE=os.path.expanduser('~/cookies.txt'),
    GDAL_HTTP_COOKIEJAR=os.path.expanduser('~/cookies.txt'),
    GDAL_SKIP='netCDF',
)
rio_env.__enter__()
for url, group in url_df.groupby('url'):
    ds = rioxarray.open_rasterio(url)
    lon = np.squeeze(ds[0]['location_lon'].data)
    lat = np.squeeze(ds[0]['location_lat'].data)
    # rfl = np.squeeze(ds[2]['reflectance'].data)

    lon_flat = lon.ravel()
    lat_flat = lat.ravel()
    locs = np.array([lon_flat, lat_flat]).T
    tree = KDTree(locs)

    for idx, poly in group.iterrows():
        if idx == 4:
            break
        bbox = ast.literal_eval(poly['bbox'])
        row1, col1 = row_col(bbox[0], bbox[1], tree, locs)
        row2, col2 = row_col(bbox[2], bbox[3], tree, locs)

        i = 35
        data = ds[2]['reflectance'][
            np.min([row1, row2])-i:np.max([row1, row2])+i,
            np.min([col1, col2])-i:np.max([col1, col2])+i,
            :
        ].values
        plt.imshow(norm(data[..., bands], 0, .12))
        plt.show()

        break
    break


