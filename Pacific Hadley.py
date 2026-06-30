import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

input_file = r"F:\div_wind.nc"
save_dir = r"F:"
R = 6.371e6
g = 9.8
target_lev = 500
lon_range = [110, 160]
lat_subtropics = [0, 40]
threshold_ratio = 0.3
integral_lev_range = [0, 500]


def clean_data(var):
    var = np.where(np.isinf(var), np.nan, var)
    if len(var.shape) == 3:
        tdim, latdim, levdim = var.shape
        for i in range(latdim):
            for j in range(levdim):
                col = var[:, i, j]
                if np.all(np.isnan(col)):
                    var[:, i, j] = np.nanmean(var)
                else:
                    mask = ~np.isnan(col)
                    x = np.where(mask)[0]
                    y = col[mask]
                    col_interp = np.interp(np.arange(len(col)), x, y)
                    var[:, i, j] = col_interp
    return var


def calculate_MSF(v_div_lon_avg, lat, lev):
    lev_pa = lev * 100
    lev_mask = (lev >= integral_lev_range[0]) & (lev <= integral_lev_range[1])
    lev_selected = lev[lev_mask]
    lev_selected_pa = lev_pa[lev_mask]
    lev_sorted_idx = np.argsort(lev_selected_pa)[::-1]
    lev_sorted_pa = lev_selected_pa[lev_sorted_idx]
    lat_rad = np.radians(lat)
    tnum, latnum, levnum = v_div_lon_avg.shape
    MSF = np.full((tnum, latnum, levnum), np.nan)
    for t in range(tnum):
        for i in range(latnum):
            v_prof = v_div_lon_avg[t, i, lev_mask]
            v_sorted = v_prof[lev_sorted_idx]
            int_v = np.trapz(v_sorted, x=lev_sorted_pa)
            psi = (2 * np.pi * R * np.cos(lat_rad[i]) / g) * int_v
            MSF[t, i, :] = psi / 1e10
    return MSF


def identify_WPHCE(MSF, lat, lev, target_lev=500, lat_sub=[0, 40], threshold=0.3):
    tlev_idx = np.argmin(np.abs(lev - target_lev))
    tnum = MSF.shape[0]
    WPHCE = np.full(tnum, np.nan)
    for t in range(tnum):
        msf_500 = MSF[t, :, tlev_idx]
        lat_mask = (lat >= lat_sub[0]) & (lat <= lat_sub[1])
        if not np.any(lat_mask):
            continue
        msf_sub = msf_500[lat_mask]
        lat_sub_arr = lat[lat_mask]
        msf_peak = np.max(msf_sub)
        msf_thres = msf_peak * threshold
        sort_idx = np.argsort(lat_sub_arr)
        lat_sorted = lat_sub_arr[sort_idx]
        msf_sorted = msf_sub[sort_idx]
        val = np.nan
        for i in range(len(lat_sorted)):
            if msf_sorted[i] <= msf_thres:
                val = lat_sorted[i]
                break
        WPHCE[t] = val if not np.isnan(val) else lat_sub[1]
    return WPHCE


if __name__ == "__main__":
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    try:
        with xr.open_dataset(input_file) as ds:
            v_div = ds["v_div"].transpose("time", "lat", "lev", "lon")
            lat = ds["lat"].values
            lon = ds["lon"].values
            lev = ds["lev"].values
            time_coords = ds["time"]
            v_lon_slice = v_div.sel(lon=slice(lon_range[0], lon_range[1]))
            v_lon_mean = v_lon_mean(dim="lon").transpose("time", "lat", "lev")
            v_clean = clean_data(v_lon_mean.values)
            MSF = calculate_MSF(v_clean, lat, lev)
            wphce_ts = identify_WPHCE(MSF, lat, lev)
            out_ds = xr.Dataset(
                {
                    "WPHCE": (["time"], wphce_ts, {
                        "units": "degree_north",
                        "integral_range": "0-500 hPa",
                        "threshold": f"{threshold_ratio * 100}% MSF peak"
                    }),
                    "MSF": (["time", "lat", "lev"], MSF, {"units": "1e10 kg/s"})
                },
                coords={"time": time_coords, "lat": lat, "lev": lev},
                attrs={"source": input_file}
            )
            nc_out = os.path.join(save_dir, "WPHCE.nc")
            out_ds.to_netcdf(nc_out)
            stats_list = []
            for m in range(1, 13):
                mon_ds = out_ds.sel(time=out_ds["time"].month == m)
                dat = mon_ds["WPHCE"].values
    except Exception as e:
        exit()