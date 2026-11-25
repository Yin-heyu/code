import numpy as np
import xarray as xr
from scipy.stats import t as tdist

EARTH_R = 6371e3
G = 9.81
RHO_W = 1000.0
FACTOR = -(100.0 * 86400.0 * 1000.0) / (RHO_W * G)

Q_FILE = r'F:\era5\q.nc'
U_FILE = r'F:\era5\u.nc'
V_FILE = r'F:\era5\v.nc'
T0, T1 = '1980-01-01', '2022-12-01'
OUT_NC = r'F:\era5\dy_th_te.nc'

REGIONS = [('ETP', -5, 5, 150, 240), ('STP', -30, -5, 150, 205),
           ('MC', -15, 15, 100, 150), ('NTP', 5, 15, 160, 275)]


def read(name, path):
    da = xr.open_dataset(path, chunks={'time': 48})[name].sel(time=slice(T0, T1))
    if 'lev' in da.dims and not np.all(np.diff(da.lev) >= 0):
        da = da.reindex(lev=np.sort(da.lev.values))
    return da.astype('float32')


def spherical_divergence(Fu, Fv, lat, lon):
    dlon = np.deg2rad(lon[1] - lon[0])
    phi = np.deg2rad(lat)
    coslat = np.cos(phi)
    Fu = np.asarray(Fu, dtype=np.float64)
    Fv = np.asarray(Fv, dtype=np.float64)
    dFu = (np.roll(Fu, -1, axis=-1) - np.roll(Fu, 1, axis=-1)) / (2 * dlon)
    dG = np.gradient(Fv * coslat[None, :, None], phi, axis=-2)
    div = (dFu + dG) / (EARTH_R * coslat[None, :, None])
    div[:, np.abs(lat) >= 89.5, :] = np.nan
    return div


def linear_trend(Y, x):
    valid = ~np.isnan(Y)
    n = valid.sum(axis=0).astype(float)
    nsafe = np.maximum(n, 1.0)
    xv = np.asarray(x, float)[:, None, None]
    xm = np.nansum(np.where(valid, xv, 0), axis=0) / nsafe
    xc = xv - xm
    ym = np.nansum(np.where(valid, Y, 0), axis=0) / nsafe
    yc = np.where(valid, Y - ym, 0.0)
    x2 = np.maximum(np.nansum((xc ** 2) * valid, axis=0), 1e-30)
    slope = np.nansum(yc * xc, axis=0) / x2
    resid = np.nansum(((yc - slope * xc) ** 2) * valid, axis=0)
    se = np.sqrt((resid / np.maximum(n - 2, 1)) / x2)
    tstat = np.divide(slope, se, out=np.full_like(slope, np.nan), where=se > 0)
    p = 2 * tdist.sf(np.abs(tstat), np.maximum(n - 2, 1))
    ok = n >= 10
    return np.where(ok, slope * 10.0, np.nan), np.where(ok, p, np.nan)


def region_mean(arr, lat, lon, box):
    _, a0, a1, b0, b1 = box
    la = (lat >= a0) & (lat <= a1)
    lo = (lon >= b0) & (lon <= b1)
    sub = arr[np.ix_(la, lo)]
    w = np.cos(np.deg2rad(lat[la]))[:, None] * np.ones((1, int(lo.sum())))
    m = ~np.isnan(sub)
    return float(np.nansum(sub[m] * w[m]) / np.nansum(w[m]))


def main():
    q = read('q', Q_FILE)
    u = read('u', U_FILE)
    v = read('v', V_FILE)

    q_bar, u_bar, v_bar = q.mean('time'), u.mean('time'), v.mean('time')
    q_a, u_a, v_a = q - q_bar, u - u_bar, v - v_bar

    flux = xr.Dataset({
        'th_u': (q_a * u_bar).integrate(coord='lev'),
        'th_v': (q_a * v_bar).integrate(coord='lev'),
        'dy_u': (q_bar * u_a).integrate(coord='lev'),
        'dy_v': (q_bar * v_a).integrate(coord='lev'),
        'te_u': (q_a * u_a).integrate(coord='lev'),
        'te_v': (q_a * v_a).integrate(coord='lev'),
    }).compute()

    lat, lon = flux.lat.values, flux.lon.values
    mm = {}
    for k in ('th', 'dy', 'te'):
        d = spherical_divergence(
            flux[f'{k}_u'].transpose('time', 'lat', 'lon').values,
            flux[f'{k}_v'].transpose('time', 'lat', 'lon').values, lat, lon)
        mm[k] = d * FACTOR

    time_year = q.time.dt.year.values + (q.time.dt.month.values - 1) / 12.0
    trend, pval = {}, {}
    for k in ('th', 'dy', 'te'):
        trend[k], pval[k] = linear_trend(mm[k], time_year)

    coords = {'lat': lat, 'lon': lon}
    dy = xr.DataArray(trend['dy'], dims=['lat', 'lon'], coords=coords)
    th = xr.DataArray(trend['th'], dims=['lat', 'lon'], coords=coords)
    te = xr.DataArray(trend['te'], dims=['lat', 'lon'], coords=coords)

    xr.Dataset(
        {'dy': dy, 'th': th, 'te': te,
         'pdy': (('lat', 'lon'), pval['dy']),
         'pth': (('lat', 'lon'), pval['th']),
         'pte': (('lat', 'lon'), pval['te'])},
        coords=coords,
        attrs={'units': '(mm/day)/decade', 'period': f'{T0} to {T1}'}
    ).to_netcdf(OUT_NC)

    total = dy + th + te
    fields = [('total', total.values), ('dy', dy.values),
              ('th', th.values), ('te', te.values)]
    print('region      total        dy        th        te')
    for box in REGIONS:
        row = '  '.join(f'{region_mean(arr, lat, lon, box):8.4f}' for _, arr in fields)
        print(f'{box[0]:<6s} ' + row)


if __name__ == '__main__':
    main()
