import numpy as np
import xarray as xr

R = 6371000.0
G = 9.80665
lat_band = (-5, 5)

def walker_streamfunction(u_div_avg, lev_pa, lat_band_deg, a=R, g=G):
    phi0 = np.deg2rad(np.mean(lat_band_deg))
    dphi = np.deg2rad(lat_band_deg[1] - lat_band_deg[0])
    factor = a * np.cos(phi0) * dphi
    factor /= g
    nlon, nlev = u_div_avg.shape
    psi = np.zeros((nlon, nlev))
    for k in range(1, nlev):
        dp = abs(lev_pa[k] - lev_pa[k-1])
        u_mid = (u_div_avg[:, k] + u_div_avg[:, k-1]) / 2
        psi[:, k] = psi[:, k-1] - factor * u_mid * dp
    return psi

# Calculation core
if __name__ == "__main__":
    ds = xr.open_dataset(r"F:\div_wind.nc")
    u_div = ds["u_div"]
    lat = ds["lat"].values
    lon = ds["lon"].values
    lev_hpa = ds["lev"].values
    lev_pa = lev_hpa * 100
    ds.close()

    lat_mask = (lat >= lat_band[0] - 0.1) & (lat <= lat_band[1] + 0.1)
    lat_idx = np.where(lat_mask)[0]
    u_div_ts = u_div.isel(lat=lat_idx).mean(dim="lat")
    nt, nlev, nlon = u_div_ts.shape
    psi_ts = np.zeros((nt, nlev, nlon))

    for t in range(nt):
        u_t = u_div_ts[t].values.T
        psi_ts[t] = walker_streamfunction(u_t, lev_pa, lat_band)

    psi_ds = xr.DataArray(psi_ts, dims=["time", "lev", "lon"], coords={
        "time": u_div_ts.time, "lev": lev_hpa, "lon": lon
    })
    psi_200 = psi_ds.sel(lev=200, method="nearest")
    psi_w = psi_200.sel(lon=slice(120, 150)).mean("lon")
    psi_e = psi_200.sel(lon=slice(200, 280)).mean("lon")
    pwc_index = (psi_e - psi_w) / 1e11
    out = xr.Dataset({"psi": psi_ds, "PWC": pwc_index})
    out.to_netcdf(r"F:\PWC.nc4")