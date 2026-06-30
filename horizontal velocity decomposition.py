import xarray as xr
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import pyamg
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import linregress
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import BoundaryNorm, ListedColormap
import os

g = 9.81

def build_spherical_laplacian(lon, lat):
    R = 6371e3
    nlat, nlon = len(lat), len(lon)
    lat_rad = np.deg2rad(lat)
    cos_lat = np.cos(lat_rad)
    dlon = np.deg2rad(lon[1] - lon[0])
    dlat = np.deg2rad(lat[1] - lat[0])
    A = lil_matrix((nlat * nlon, nlat * nlon))
    for j in range(nlat):
        for i in range(nlon):
            idx = j * nlon + i
            cl = cos_lat[j]
            lon_coeff = 1.0 / (R * dlon * cl) ** 2 if cl != 0 else 0.0
            lat_coeff = 1.0 / (R * dlat) ** 2
            left, right = (i - 1) % nlon, (i + 1) % nlon
            idx_l, idx_r = j * nlon + left, j * nlon + right
            idx_d = (j - 1) * nlon + i if j > 0 else None
            idx_u = (j + 1) * nlon + i if j < nlat - 1 else None
            A[idx, idx] = -2 * (lon_coeff + lat_coeff)
            A[idx, idx_l] += lon_coeff
            A[idx, idx_r] += lon_coeff
            if idx_d and idx_u:
                A[idx, idx_d] += lat_coeff
                A[idx, idx_u] += lat_coeff
            else:
                edge_coeff = lat_coeff * 2
                if idx_d is None:
                    A[idx, idx] = -2 * lon_coeff - lat_coeff
                    A[idx, idx_u] += edge_coeff
                if idx_u is None:
                    A[idx, idx] = -2 * lon_coeff - lat_coeff
                    A[idx, idx_d] += edge_coeff
    return A.tocsr()

def solve_helmholtz(D, A):
    ml = pyamg.ruge_stuben_solver(A)
    D_flat = D - D.mean()
    return ml.solve(D_flat.ravel(), tol=1e-10).reshape(D.shape)

def process_level(args):
    lev_idx, D_t, A = args
    return solve_helmholtz(D_t, A)

def calculate_div_wind(chi, lon, lat):
    R = 6371e3
    dlon, dlat = np.deg2rad(lon[1]-lon[0]), np.deg2rad(lat[1]-lat[0])
    coslat = np.cos(np.deg2rad(lat))[:, None]
    dchi_lon = np.gradient(chi, axis=1)
    dchi_lat = np.gradient(chi, axis=0)
    u = dchi_lon / (R * dlon * coslat)
    v = dchi_lat / (R * dlat)
    return u, v

if __name__:
    # Step1 Velocity potential
    try:
        div_ds = xr.open_dataset(r'E:\divergence.nc')
        D = div_ds['divergence'].values
        lon, lat, lev, time = div_ds.lon.values, div_ds.lat.values, div_ds.lev.values, div_ds.time.values
    except FileNotFoundError:
        exit()
    A = build_spherical_laplacian(lon, lat)
    nt, nlev = len(time), len(lev)
    chi = np.empty_like(D)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    with tqdm(total=nt, desc="Solve potential") as pbar:
        for t in range(nt):
            res = pool.map(process_level, [(li, D[t], A) for li in range(nlev)])
            for li, val in enumerate(res): chi[t] = val
            pbar.update()
    pool.close()
    pool.join()
    try:
        xr.Dataset({'velocity_potential': (('time','lev','lat','lon'),chi)},
                   coords=dict(lon=lon,lat=lat,lev=lev,time=time)).to_netcdf(r'E:\velocity_potential.nc')
    except: pass

    # Step2 Divergent wind
    u_div, v_div = np.empty_like(chi), np.empty_like(chi)
    with tqdm(total=nt*nlev, desc="Calc div wind") as pbar:
        for t in range(nt):
            for li in range(nlev):
                u_div[t,li], v_div[t,li] = calculate_div_wind(chi[t,li], lon, lat)
                pbar.update()
    try:
        xr.Dataset({
            'u_div':(('time','lev','lat','lon'),u_div),
            'v_div':(('time','lev','lat','lon'),v_div)
        }, coords=dict(lon=lon,lat=lat,lev=lev,time=time)).to_netcdf(r'E:\div_wind.nc')
    except: pass

