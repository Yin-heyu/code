import xarray as xr
import dask
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm, ListedColormap
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# File paths
q_file = r'F:\era5\q.nc'
u_file = r'F:\era5\u.nc'
v_file = r'F:\era5\v.nc'

# Load data
with dask.config.set(scheduler='threads'):
    with tqdm(total=3, desc="Load files") as pbar:
        q = xr.open_dataset(q_file, chunks={'time': 48})['q']
        pbar.update(1)
        pbar.set_postfix_str("q loaded")

        U = xr.open_dataset(u_file, chunks={'time': 48})['u']
        pbar.update(1)
        pbar.set_postfix_str("u loaded")

        V = xr.open_dataset(v_file, chunks={'time': 48})['v']
        pbar.update(1)
        pbar.set_postfix_str("v loaded")


with tqdm(total=3, desc="Time slice") as pbar:
    q = q.sel(time=slice('1980-01-01', '2022-12-01'))
    pbar.update(1)
    U = U.sel(time=slice('1980-01-01', '2022-12-01'))
    pbar.update(1)
    V = V.sel(time=slice('1980-01-01', '2022-12-01'))
    pbar.update(1)

with tqdm(total=3, desc="Sort lev") as pbar:
    if not np.all(np.diff(q.lev) >= 0):
        q = q.reindex(lev=np.sort(q.lev.values))
    pbar.update(1)

    if not np.all(np.diff(U.lev) >= 0):
        U = U.reindex(lev=np.sort(U.lev.values))
    pbar.update(1)

    if not np.all(np.diff(V.lev) >= 0):
        V = V.reindex(lev=np.sort(V.lev.values))
    pbar.update(1)

# Time mean
with tqdm(total=3, desc="Time mean") as pbar:
    q_mean = q.mean(dim='time').compute()
    pbar.update(1)
    pbar.set_postfix_str("q mean done")

    U_mean = U.mean(dim='time').compute()
    pbar.update(1)
    pbar.set_postfix_str("u mean done")

    V_mean = V.mean(dim='time').compute()
    pbar.update(1)
    pbar.set_postfix_str("v mean done")

# Flux decomposition
with tqdm(total=6, desc="Flux calc") as pbar:
    horizontal_thermodynamic_flux_u = q * U_mean
    pbar.update(1)
    pbar.set_postfix_str("th flux u")

    horizontal_thermodynamic_flux_v = q * V_mean
    pbar.update(1)
    pbar.set_postfix_str("th flux v")

    horizontal_dynamic_flux_u = q_mean * U
    pbar.update(1)
    pbar.set_postfix_str("dyn flux u")

    horizontal_dynamic_flux_v = q_mean * V
    pbar.update(1)
    pbar.set_postfix_str("dyn flux v")

    horizontal_turbulent_flux_u = (U - U_mean) * (q - q_mean)
    pbar.update(1)
    pbar.set_postfix_str("eddy flux u")

    horizontal_turbulent_flux_v = (V - V_mean) * (q - q_mean)
    pbar.update(1)
    pbar.set_postfix_str("eddy flux v")

# Vertical integral
with tqdm(total=6, desc="Vertical integral") as pbar:
    thermodynamic_flux_u_int = horizontal_thermodynamic_flux_u.integrate(coord='lev').compute()
    pbar.update(1)
    pbar.set_postfix_str("th int u")

    thermodynamic_flux_v_int = horizontal_thermodynamic_flux_v.integrate(coord='lev').compute()
    pbar.update(1)
    pbar.set_postfix_str("th int v")

    dynamic_flux_u_int = horizontal_dynamic_flux_u.integrate(coord='lev').compute()
    pbar.update(1)
    pbar.set_postfix_str("dyn int u")

    dynamic_flux_v_int = horizontal_dynamic_flux_v.integrate(coord='lev').compute()
    pbar.update(1)
    pbar.set_postfix_str("dyn int v")

    turbulent_flux_u_int = horizontal_turbulent_flux_u.integrate(coord='lev').compute()
    pbar.update(1)
    pbar.set_postfix_str("eddy int u")

    turbulent_flux_v_int = horizontal_turbulent_flux_v.integrate(coord='lev').compute()
    pbar.update(1)
    pbar.set_postfix_str("eddy int v")


def horizontal_divergence(data_u, data_v, desc="Div calc"):
    with tqdm(total=4, desc=desc) as pbar:
        dx = 111e3 * np.cos(np.deg2rad(data_u.lat)) * np.abs(data_u.lon.diff('lon')[0])
        pbar.update(1)
        pbar.set_postfix_str("dx done")

        dy = 111e3 * np.abs(data_u.lat.diff('lat')[0])
        pbar.update(1)
        pbar.set_postfix_str("dy done")

        dudx = data_u.differentiate('lon') / dx
        pbar.update(1)
        pbar.set_postfix_str("dudx done")

        dvdy = data_v.differentiate('lat') / dy
        pbar.update(1)
        pbar.set_postfix_str("dvdy done")
    return dudx + dvdy


horizontal_thermodynamic_divergence = horizontal_divergence(
    thermodynamic_flux_u_int, thermodynamic_flux_v_int, "Thermo div"
).compute()

horizontal_dynamic_divergence = horizontal_divergence(
    dynamic_flux_u_int, dynamic_flux_v_int, "Dynamic div"
).compute()

horizontal_turbulent_divergence = horizontal_divergence(
    turbulent_flux_u_int, turbulent_flux_v_int, "Eddy div"
)

# Unit convert to mm/day
with tqdm(total=3, desc="Unit convert") as pbar:
    horizontal_thermodynamic_integral_mm_day = horizontal_thermodynamic_divergence / (1000 * 9.81)
    pbar.update(1)
    pbar.set_postfix_str("thermo unit")

    horizontal_dynamic_integral_mm_day =horizontal_dynamic_divergence / (1000 * 9.81)
    pbar.update(1)
    pbar.set_postfix_str("dynamic unit")

    horizontal_turbulent_integral_mm_day = horizontal_turbulent_divergence / (1000 * 9.81)
    pbar.update(1)
    pbar.set_postfix_str("eddy unit")


def linear_trend(data, desc="Trend calc"):
    data = data.transpose('time', 'lat', 'lon')
    time = data.time.dt.year.values
    trends = np.full(data.shape[1:], np.nan)
    total_points = data.shape[1] * data.shape[2]
    with tqdm(total=total_points, desc=desc, ncols=80) as pbar:
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                series = data[:, j, k].values
                valid_mask = ~np.isnan(series)
                valid_series = series[valid_mask]
                valid_time = time[valid_mask]
                if len(valid_series) >= 10:
                    norm_time = (valid_time - valid_time.min()) / (valid_time.max() - valid_time.min())
                    try:
                        poly_coeffs = np.polyfit(norm_time, valid_series, deg=1)
                        trends[j, k] = poly_coeffs[0] * (valid_time.max() - valid_time.min())
                    except np.linalg.LinAlgError:
                        pass
                pbar.update(1)
    return xr.DataArray(trends, dims=['lat', 'lon'], coords=[data.lat, data.lon])


dy = linear_trend(horizontal_dynamic_integral_mm_day, "Dynamic trend")
th = linear_trend(horizontal_thermodynamic_integral_mm_day, "Thermo trend")
te = linear_trend(horizontal_turbulent_integral_mm_day, "Eddy trend")

# Output dataset
with tqdm(total=1, desc="Build dataset") as pbar:
    trend_dataset = xr.Dataset(
        {
            'dy': dy,
            'th': th,
            'te': te
        },
        attrs={
            'description': 'Dynamic/Thermodynamic/Eddy moisture flux divergence trend',
            'units': '(mm/day)/decade',
            'time_period': '1980-2022',
        }
    )
    trend_dataset['dy'].attrs = {'long_name': 'Dynamic component trend'}
    trend_dataset['th'].attrs = {'long_name': 'Thermodynamic component trend'}
    trend_dataset['te'].attrs = {'long_name': 'Eddy turbulent component trend'}
    pbar.update(1)

output_file = r'F:\dy_th_te.nc'
with tqdm(total=1, desc="Save nc") as pbar:
    try:
        trend_dataset.to_netcdf(output_file)
        pbar.update(1)
    except Exception:
        pass


