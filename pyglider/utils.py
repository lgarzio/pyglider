"""
Utilities that are used for processing scripts.
Modified by Lori Garzio 5/19/2025
Last Modified 6/9/2025
"""

import collections
import logging
from pathlib import Path

import gsw
import numpy as np
import xarray as xr
import yaml
from scipy.signal import argrelextrema

_log = logging.getLogger(__name__)


def get_distance_over_ground(ds):
    """
    Add a distance over ground variable to a netcdf structure

    Parameters
    ----------
    ds : `xarray.Dataset`
        Must have variable ``latitude`` and ``longitude`` indexed
        by ``time`` dimension.

    Returns
    -------
    ds : `.xarray.Dataset`
        With ``distance_over_ground`` key.
    """

    good = ~np.isnan(ds.latitude + ds.longitude)
    if np.any(good):
        dist = gsw.distance(ds.longitude[good].values, ds.latitude[good].values) / 1000
        dist = np.roll(np.append(dist, 0), 1)
        dist = np.cumsum(dist)
        dist = np.interp(ds.time, ds.time[good], dist)
    else:
        dist = 0 * ds.latitude.values
    attr = {
        'long_name': 'distance over ground flown since mission start',
        'method': 'get_distance_over_ground',
        'units': 'km',
        'sources': 'latitude longitude',
    }
    ds['distance_over_ground'] = (('time'), dist, attr)
    return ds


def get_glider_depth(ds):
    """
    Get glider depth from pressure sensor.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Must have variables ``pressure`` and ``latitude`` indexed
        by ``time`` dimension.  Assume pressure sensor in dbar.

    Returns
    -------
    ds : `.xarray.Dataset`
        With ``depth`` key.

    """
    good = np.where(~np.isnan(ds.pressure))[0]
    ds['depth'] = ds.pressure
    try:
        meanlat = ds.latitude.mean(skipna=True)
        ds['depth'].values = -gsw.z_from_p(
            ds.pressure.values, ds.latitude.fillna(meanlat).values
        )
    except AttributeError:
        pass
    # now we really want to know where it is, so interpolate:
    if len(good) > 0:
        ds['depth'].values = np.interp(
            np.arange(len(ds.depth)), good, ds['depth'].values[good]
        )

    attr = {
        'source': 'pressure',
        'long_name': 'glider depth',
        'standard_name': 'depth',
        'units': 'm',
        'comment': 'calculated from sci_water_pressure and interpolated',
        'instrument': 'instrument_ctd',
        'observation_type': 'calulated',
        'accuracy': 1.0,
        'precision': 2.0,
        'resolution': 0.02,
        'platform': 'platform',
        'valid_min': 0.0,
        'valid_max': 2000.0,
        'reference_datum': 'surface',
        'positive': 'down',
    }
    ds['depth'].attrs = attr
    return ds


def get_glider_depth_new(ds):
    """
    Get glider depth from the CTD pressure sensor. Lori: copied function "get_glider_depth" and
    changed the interpolation and added a step to fill in missing values with m_depth 
    (depth measured by the glider pressure transducer) when sci_water_pressure is missing.  
    This is because sometimes there are long stretches of missing pressure data, and m_depth 
    is often pretty close to the depth calculated from CTD pressure, so it can be used to fill in those gaps.

    Parameters
    ----------
    ds : `xarray.Dataset`
        Must have variables ``pressure`` and ``latitude`` indexed
        by ``time`` dimension.  Assume pressure sensor in dbar.

    Returns
    -------
    ds : `.xarray.Dataset`
        With ``depth`` key.

    """
    good = np.where(~np.isnan(ds.pressure))[0]
    ds['depth'] = ds.pressure
    try:
        meanlat = ds.latitude.mean(skipna=True)
        depth_m = -gsw.z_from_p(
            ds.pressure.values, ds.latitude.fillna(meanlat).values
        )
    except AttributeError:
        pass
    # now we really want to know where it is, so interpolate:
    if len(good) > 0:
        # ds['depth'].values = np.interp(
        #     np.arange(len(ds.depth)), good, ds['depth'].values[good]
        # )
        
        # added by Lori - convert to dataframe to use pandas interpolation
        # because I don't like the numpy interpolation because you can't easily limit
        # the amount that's interpolated, then convert back to xarray
        import pandas as pd
        df = pd.DataFrame({'depth': depth_m})
        depth_interp = df['depth'].interpolate(method='linear', limit_direction='both', limit=5).values
        
        # where depth_interp is nan, fill in with m_depth (depth measured by the glider pressure transducer)
        if np.sum(np.isnan(depth_interp)) > 0:
            # check that the max m_depth values are within +/- 10% of the max depth_interp values
            check_range = [np.nanmax(depth_interp) * .9, np.nanmax(depth_interp) * 1.1]
            if check_range[0] <= np.nanmax(ds.m_depth.values) <= check_range[1]:
                depth_interp[np.isnan(depth_interp)] = ds.m_depth.values[np.isnan(depth_interp)]
        
        ds['depth'].values = depth_interp

    attr = {
        'source': 'pressure, m_depth',
        'long_name': 'glider depth',
        'standard_name': 'depth',
        'units': 'm',
        'comment': 'Depth calculated from sci_water_pressure and interpolated. Then filled in with m_depth when sci_water_pressure is missing.',
        'instrument': 'instrument_ctd',
        'observation_type': 'calulated',
        'accuracy': 1.0,
        'precision': 2.0,
        'resolution': 0.02,
        'platform': 'platform',
        'valid_min': 0.0,
        'valid_max': 2000.0,
        'reference_datum': 'surface',
        'positive': 'down',
    }
    ds['depth'].attrs = attr
    return ds


def get_profiles(ds, min_dp=10.0, inversion=3.0, filt_length=7, min_nsamples=14):
    """
    Not currently used...

    make two variables: profile_direction and profile_index; this version
    is good for lots of data.  Less good for sparse data
    """
    if 'pressure' not in ds:
        _log.warning(
            'No "pressure" variable in the data set; not searching for profiles'
        )
        return ds
    profile = ds.pressure.values * np.nan
    direction = ds.pressure.values * np.nan
    pronum = 1
    lastpronum = 0

    good = np.where(~np.isnan(ds.pressure))[0]
    p = np.convolve(
        ds.pressure.values[good], np.ones(filt_length) / filt_length, 'same'
    )
    dpall = np.diff(p)
    inflect = np.where(dpall[:-1] * dpall[1:] < 0)[0]
    for n, i in enumerate(inflect[:-1]):
        nprofile = inflect[n + 1] - inflect[n]
        inds = np.arange(good[inflect[n]], good[inflect[n + 1]] + 1) + 1
        dp = np.diff(ds.pressure[inds[[-1, 0]]])
        if (nprofile >= min_nsamples) and (np.abs(dp) > 10):
            _log.debug('Good')
            direction[inds] = np.sign(dp)
            profile[inds] = pronum
            lastpronum = pronum
            pronum += 1
        else:
            profile[good[inflect[n]] : good[inflect[n + 1]]] = lastpronum + 0.5

    attrs = collections.OrderedDict(
        [
            ('long_name', 'profile index'),
            ('units', '1'),
            ('comment', 'N = inside profile N, N + 0.5 = between profiles N and N + 1'),
            ('sources', 'time pressure'),
            ('method', 'get_profiles'),
            ('min_dp', min_dp),
            ('filt_length', filt_length),
            ('min_nsamples', min_nsamples),
        ]
    )
    ds['profile_index'] = (('time'), profile, attrs)

    attrs = collections.OrderedDict(
        [
            ('long_name', 'glider vertical speed direction'),
            ('units', '1'),
            ('comment', '-1 = ascending, 0 = inflecting or stalled, 1 = descending'),
            ('sources', 'time pressure'),
            ('method', 'get_profiles'),
        ]
    )
    ds['profile_direction'] = (('time'), direction, attrs)
    return ds


def get_profiles_new(ds, min_dp=10.0, filt_time=100, profile_min_time=300):
    """
    Find profiles in a glider timeseries:

    Parameters
    ----------
    ds : `xarray.Dataset`
        Must have *time* coordinate and *depth* as a variable
    min_dp : float, default=10.0
        Minimum distance a profile must transit to be considered a profile, in m.
    filt_time : float, default=100
        Approximate length of time filter, in seconds.  Note that the filter
        is really implemented by sample, so the number of samples is
        ``filt_time / dt``
        where *dt* is the median time between samples in the time series.
    profile_min_time : float, default=300
        Minimum time length of profile in s.
    """

    if 'depth' not in ds:
        _log.warning(
            'No "depth" variable in the data set; not searching for profiles'
        )
        return ds

    profile = ds.depth.values * 0
    direction = ds.depth.values * 0
    pronum = 1

    good = np.where(np.isfinite(ds.depth))[0]

    # if depth is all nan or the depth change is < min_dp, profile id is zero
    if np.logical_or(len(good) == 0, np.nanmax(ds.depth.values) - np.nanmin(ds.depth.values) < min_dp):
        profile.fill(0)
        direction.fill(0)

        attrs = collections.OrderedDict(
        [
            ('long_name', 'profile index'),
            ('units', '1'),
            ('comment', 'Unique identifier of the profile, the mean profile timestamp'),
            ('sources', 'time pressure'),
            ('method', 'get_profiles_new'),
        ]
    )
    else:  # otherwise, index profiles
        # dt = float(
        #     np.median(np.diff(ds.time.values[good[:200000]]).astype(np.float64)) * 1e-9
        # )
        # Lori: convert timestamps
        tm = ds.time.values.astype('timedelta64[s]') + np.datetime64('1970-01-01T00:00:00').astype('datetime64[ns]')
        dt = float(
            np.median(np.diff(tm[good[:200000]]).astype(np.float64)) * 1e-9
        )
        _log.info(f'dt, {dt}')
        filt_length = int(filt_time / dt)

        min_nsamples = int(profile_min_time / dt)
        _log.info('Filt Len  %d, dt %f, min_n %d', filt_length, dt, min_nsamples)
        # if filt_length > 1:
        #     # performs a moving average smoothing
        #     p = np.convolve(
        #         ds.pressure.values[good], np.ones(filt_length) / filt_length, 'same'
        #     )
        # else:
        #     p = ds.pressure.values[good]

        # smooth the data
        from scipy.ndimage import gaussian_filter1d
        p = gaussian_filter1d(ds.depth.values[good], sigma=2)

        decim = int(filt_length / 3)
        if len(p) < 150:  # if there are <150 data points, don't decimate
            decim = 1
        elif decim < 2:
            decim = 2
        # why?  because argrelextrema doesn't like repeated values, so smooth
        # then decimate to get fewer values:
        pp = p[::decim]
        maxs = argrelextrema(pp, np.greater)[0]
        mins = argrelextrema(pp, np.less)[0]
        mins = good[mins * decim]
        maxs = good[maxs * decim]

        if np.logical_and(len(maxs) == 0, len(mins) == 0):
            # no mins or maxs found, see if the values are all increasing or all decreasing
            if np.all(np.diff(pp) >= 0):  # values are increasing
                #mins = np.concatenate(([0], mins))
                #maxs = np.concatenate((maxs, [good[-1]]))
                profile.fill(1)
                direction.fill(1)
            elif np.all(np.diff(pp) <= 0):  # values are decreasing
                profile.fill(1)
                direction.fill(-1)
            else:
                # if there are no minima or maxima, but the values are not all increasing or decreasing, assign everything zero
                profile.fill(0)
                direction.fill(0)
        elif np.logical_and(len(mins) > 0, len(maxs) == 0):
                # if there are minima but no maxima, assign everything zero (can't find profiles)
                profile.fill(0)
                direction.fill(0)
        else:
            if np.logical_and(len(mins) == 0, len(maxs) > 0):
                # if there are no minima, but there are maxima, assume the profile starts at index=0
                mins = np.concatenate(([0], mins))
            if mins[0] > maxs[0]:
                mins = np.concatenate(([0], mins))

            if mins[-1] < maxs[-1]:
                #mins = np.concatenate((mins, good[[-1]]))
                if good[-1] + 4 > len(ds.time):
                    endmin = len(ds.time)  # extend to the end of the time series if it is less than last valid pressure reading + 4
                else:
                    endmin = good[-1] + 4  # extend to the last valid pressure reading + 4
                mins = np.concatenate((mins, [endmin]))

            _log.debug(f'mins: {len(mins)} {mins} , maxs: {len(maxs)} {maxs}')

            pronum = 1
            p = ds.depth
            nmin = 0
            nmax = 0
            while (nmin < len(mins)) and (nmax < len(maxs)):
                nmax = np.where(maxs > mins[nmin])[0]
                if len(nmax) >= 1:
                    nmax = nmax[0]
                else:
                    break
                _log.debug(nmax)
                ins = range(int(mins[nmin]), int(maxs[nmax] + 1))
                _log.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
                _log.debug(f'Down, {ins}, {p[ins[0]].values},{p[ins[-1]].values}')
                if (len(ins) > min_nsamples) and (
                    np.nanmax(p[ins]) - np.nanmin(p[ins]) > min_dp
                ):
                    # downcast
                    profile[ins] = pronum
                    direction[ins] = +1
                    pronum += 1
                
                    # find depths < 0.1 and assign those profile ID 0
                    # these are downcasts so find the last instance and set everything before that to zero
                    if np.sum(p[ins] < 0.1) > 0:
                        ins = range(ins[0], ins[0] + np.max(np.where(p[ins] < 0.1)[0]))
                        profile[ins] = 0
                        direction[ins] = 0

                else:  # glider is hovering
                    profile[ins] = 0
                    direction[ins] = 0
                
                nmin = np.where(mins > maxs[nmax])[0]
                if len(nmin) >= 1:
                    nmin = nmin[0]
                else:
                    break
                ins = range(maxs[nmax], mins[nmin])
                _log.debug(f'{pronum}, {ins}, {len(p)}, {mins[nmin]}, {maxs[nmax]}')
                _log.debug(f'Up, {ins}, {p[ins[0]].values}, {p[ins[-1]].values}')
                if (len(ins) > min_nsamples) and (
                    np.nanmax(p[ins]) - np.nanmin(p[ins]) > min_dp
                ):
                    # upcast
                    profile[ins] = pronum
                    direction[ins] = -1
                    pronum += 1

                    # find depths < 0.1 and assign those profile ID 0
                    # these are upcasts so find the first instance and set everything after that to zero
                    if np.sum(p[ins] < 0.1) > 0:
                        ins = range(ins[0] + np.min(np.where(p[ins] < 0.1)[0]), ins[-1] + 1)
                        profile[ins] = 0
                        direction[ins] = 0

                else:  # glider is hovering
                    profile[ins] = 0
                    direction[ins] = 0

        # # added by Lori
        # # if the unique profile id is 0 (no profiles were indexed), make sure everything equals zero (remove nans)
        # unique_profileid = np.unique(profile[~np.isnan(profile)])
        # if np.array_equal(unique_profileid, [0]):
        #     profile.fill(0)
        #     direction.fill(0)
        # else:
        #     # otherwise, index where the glider is floating at the surface and assign that profile id 0
        #     surface_idx = np.where(ds.depth.values < 0.2)[0]
        #     profile[surface_idx] = 0
        #     direction[surface_idx] = 0
        
        # if there is anything leftover that wasn't indexed, assign those zero
        profile[np.where(np.isnan(profile))[0]] = 0
        direction[np.where(np.isnan(direction))[0]] = 0

        # # Calculate time elapsed for each profile and compare them
        # ptimes_elapsed = []
        # for pid in np.unique(profile):
        #     if pid > 0:
        #         ins = np.where(profile == pid)[0]
        #         xx = ds.time.values[ins]
        #         time_elapsed_minutes = (np.nanmax(xx) - np.nanmin(xx)) / 60
        #         ptimes_elapsed.append(time_elapsed_minutes)
        
        # Lori: convert the profile ID to the mean timestamp of the profile
        #profile_time = ds.depth.values * 0
        profile_ids = np.unique(profile)
        for pid in profile_ids:
            if pid > 0:
                ins = np.where(profile == pid)[0]
                profile[ins] = np.nanmean(ds.time.values[ins])

        # if there is no depth, assign those profiles zero
        depth_idx = np.where(np.isnan(ds.depth))[0]
        profile[depth_idx] = 0
        direction[depth_idx] = 0
        #profile_time[depth_idx] = 0
        
        attrs = collections.OrderedDict(
            [
                ('long_name', 'profile index'),
                ('units', '1'),
                #('comment', 'N = inside profile N, N + 0.5 = between profiles N and N + 1'),
                ('comment', 'Unique identifier of the profile, the mean profile timestamp'),
                ('sources', 'time depth'),
                ('method', 'get_profiles_new'),
                ('min_dp', min_dp),
                ('filt_length', filt_length),
                ('min_nsamples', min_nsamples),
            ]
        )
    
    ds['profile_id'] = (('time'), profile, attrs)

    attrs = collections.OrderedDict(
        [
            ('long_name', 'glider vertical speed direction'),
            ('units', '1'),
            #('comment', '-1 = ascending, 0 = inflecting or stalled, 1 = descending'),
            ('comment', '-1 = ascending, 0 = hovering or floating at surface, 1 = descending'),
            ('sources', 'time depth'),
            ('method', 'get_profiles_new'),
        ]
    )
    ds['profile_direction'] = (('time'), direction, attrs)

    #ds['profile_time'] = (('time'), profile_time, attrs)
    return ds


def get_derived_eos_raw(ds):
    """
    Calculate salinity, potential density, density, and potential temperature

    Parameters
    ----------
    ds : `xarray.Dataset`
        Must have *time* coordinate and *temperature*, *conductivity*, *pressure*,
        and *latitude* and *longitude* as variables.

    Returns
    -------
    ds : `xarray.Dataset`
        with *salinity*, *potential_density*, *density*, and *potential_temperature*
        as new variables.

    Notes
    -----
    Thermodynamic variables derived from the Gibbs seawater toolbox ``import gsw``.

    - *salinity*::
        gsw.conversions.SP_from_C(r, ds.temperature, ds.pressure)
    - *potential_density*::
        sa = gsw.SA_from_SP(ds['salinity'], ds['pressure'],
                            ds['longitude'], ds['latitude'])
        ct = gsw.CT_from_t(sa, ds['temperature'], ds['pressure'])
        ds['potential_density'] = (('time'),
                1000 + gsw.density.sigma0(sa, ct).values)
    - *density*::
        ds['density'] = (('time'), gsw.density.rho(
            ds.salinity, ds.temperature, ds.pressure).values)
    - *potential_temperature*::
        ds['potential_temperature'] = (('time'), gsw.conversions.pt0_from_t(
            ds.salinity, ds.temperature, ds.pressure).values)

    """

    # GPCTD and slocum ctd require a scale factor of 10 for conductivity.
    # Legato does not
    if 'S m' in ds.conductivity.units:
        r = 10 * ds.conductivity
    elif 'mS cm' in ds.conductivity.units:
        r = ds.conductivity
    else:
        raise ValueError(
            'Could not parse conductivity units in yaml. '
            "Expected 'S m-1' or 'mS cm-1'. "
            'Check yaml entry netcdf_variables: conductivity: units'
        )
    ds['salinity'] = (
        ('time'),
        gsw.conversions.SP_from_C(r, ds.temperature, ds.pressure).values,
    )
    attrs = collections.OrderedDict(
        [
            ('long_name', 'water salinity'),
            ('standard_name', 'sea_water_practical_salinity'),
            ('units', '1e-3'),
            ('comment', 'raw, uncorrected salinity'),
            ('sources', 'conductivity temperature pressure'),
            ('method', 'get_derived_eos_raw'),
            ('observation_type', 'calulated'),
            ('instrument', 'instrument_ctd'),
            ('valid_max', 40.0),
            ('valid_min', 0.0),
            ('accuracy', 0.01),
            ('precision', 0.01),
            ('resolution', 0.001),
        ]
    )
    attrs = fill_required_attrs(attrs)
    ds['salinity'].attrs = attrs
    long = ds.longitude.fillna(ds.longitude.mean(skipna=True))
    lat = ds.latitude.fillna(ds.latitude.mean(skipna=True))
    sa = gsw.SA_from_SP(ds['salinity'], ds['pressure'], long, lat)
    ct = gsw.CT_from_t(sa, ds['temperature'], ds['pressure'])
    ds['potential_density'] = (('time'), 1000 + gsw.density.sigma0(sa, ct).values)
    attrs = collections.OrderedDict(
        [
            ('long_name', 'water potential density'),
            ('standard_name', 'sea_water_potential_density'),
            ('units', 'kg m-3'),
            ('comment', 'raw, uncorrected salinity'),
            ('sources', 'salinity temperature pressure'),
            ('method', 'get_derived_eos_raw'),
            ('observation_type', 'calulated'),
            ('instrument', 'instrument_ctd'),
            ('accuracy', 0.01),
            ('precision', 0.01),
            ('resolution', 0.001),
        ]
    )
    attrs = fill_required_attrs(attrs)
    ds['potential_density'].attrs = attrs

    ds['density'] = (
        ('time'),
        gsw.density.rho(ds.salinity, ds.temperature, ds.pressure).values,
    )
    attrs = collections.OrderedDict(
        [
            ('long_name', 'Density'),
            ('standard_name', 'sea_water_density'),
            ('units', 'kg m-3'),
            ('comment', 'raw, uncorrected salinity'),
            ('observation_type', 'calulated'),
            ('sources', 'salinity temperature pressure'),
            ('instrument', 'instrument_ctd'),
            ('method', 'get_derived_eos_raw'),
            ('valid_min', 990.0),
            ('valid_max', 1040.0),
            ('accuracy', 0.01),
            ('precision', 0.01),
            ('resolution', 0.001),
        ]
    )
    attrs = fill_required_attrs(attrs)
    ds['density'].attrs = attrs
    ds['potential_temperature'] = (
        ('time'),
        gsw.conversions.pt0_from_t(ds.salinity, ds.temperature, ds.pressure).values,
    )
    attrs = collections.OrderedDict(
        [
            ('long_name', 'water potential temperature'),
            ('standard_name', 'sea_water_potential_temperature'),
            ('units', 'Celsius'),
            ('comment', 'raw, uncorrected salinity'),
            ('sources', 'salinity temperature pressure'),
            ('observation_type', 'calulated'),
            ('method', 'get_derived_eos_raw'),
            ('instrument', 'instrument_ctd'),
            ('accuracy', 0.002),
            ('precision', 0.001),
            ('resolution', 0.0001),
        ]
    )
    attrs = fill_required_attrs(attrs)
    ds['potential_temperature'].attrs = attrs

    return ds


def _time_to_datetime64(time):
    """
    Pass in a glider undecoded time (seconds since 1970-01-01), and
    get a np.datetime64[s] object back.
    """
    return time.astype('timedelta64[s]') + np.datetime64('1970-01-01')


def fill_required_attrs(attrs):
    required = {
        'comment': ' ',
        'accuracy': ' ',
        'precision': ' ',
        'platform': 'platform',
        'resolution': ' ',
        'ancillary_variables': ' ',
    }
    for k in required.keys():
        if k not in attrs.keys():
            attrs[k] = required[k]
    return attrs


def fill_required_qcattrs(attrs, varname):
    required = {
        'units': '1',
        'flag_values': np.array([1, 2, 3, 4, 9], dtype=np.int8),
        'valid_min': np.int8(1),
        'valid_max': np.int8(9),
        'flag_meanings': 'PASS NOT_EVALUATED SUSPECT FAIL MISSING',
        'standard_name': 'quality_flag',
        'long_name': 'Initial flag for {varname}',
    }
    for k in required.keys():
        if k not in attrs.keys():
            attrs[k] = required[k]
    return attrs


def get_file_id(ds):
    """
    Make a file id for a Dataset

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset to make an id for.  The attributes of the Dataset
        must have *glider_name* and *glider_serial* in them.

    Return
    ------
    id : string
        Id = *glider_name* + *glider_serial* + "YYYYMMDDTHHMM"

    """

    _log.debug(ds.time)
    if not ds.time.dtype == 'datetime64[ns]':
        dt = _time_to_datetime64(ds.time.values[0])
    else:
        dt = ds.time.values[0].astype('datetime64[s]')
    _log.debug(f'dt, {dt}')
    id = (
        ds.attrs['glider_name']
        + ds.attrs['glider_serial']
        + '-'
        + dt.item().strftime('%Y%m%dT%H%M')
    )
    return id


def fill_metadata(ds, metadata, sensor_data):
    """
    Add metadata to a Dataset

    Parameters
    ----------
    ds : `xarray.Dataset`
        Dataset must have *longtidue*, *latitude*, and *time* values
    metadata : dict
        dictionary of attributes to add to the global attributes.  Usually
        taken from *deployment.yml* file.
    sensor_data : dict
        dictionary of device data to add to the global attributes.

    Returns
    -------
    ds : `xarray.Dataset`
        Dataset with attributes filled out.

    * Modified by Lori Garzio 5/19/2025 - don't write over existing attributes
    """
    good = ~np.isnan(ds.latitude.values + ds.longitude.values)
    if np.any(good):
        ds.attrs['geospatial_lat_max'] = np.max(ds.latitude.values[good])
        ds.attrs['geospatial_lat_min'] = np.min(ds.latitude.values[good])
        ds.attrs['geospatial_lon_max'] = np.max(ds.longitude.values[good])
        ds.attrs['geospatial_lon_min'] = np.min(ds.longitude.values[good])
    else:
        ds.attrs['geospatial_lat_max'] = np.nan
        ds.attrs['geospatial_lat_min'] = np.nan
        ds.attrs['geospatial_lon_max'] = np.nan
        ds.attrs['geospatial_lon_min'] = np.nan

    # ds.attrs['geospatial_lat_units'] = 'degrees_north'
    # ds.attrs['geospatial_lon_units'] = 'degrees_east'
    ds.attrs['netcdf_version'] = '4.0'  # TODO get this somehow...
    ds.attrs['history'] = 'CPROOF glider toolbox version: pre-tag'
    for k, v in metadata.items():
        ds.attrs[k] = v
    ds.attrs['featureType'] = 'trajectory'
    ds.attrs['cdm_data_type'] = 'Trajectory'
    # ds.attrs['Conventions'] = 'CF-1.8'
    # ds.attrs['standard_name_vocabulary'] = 'CF STandard Name Table v72'
    ds.attrs['date_created'] = str(np.datetime64('now')) + 'Z'
    ds.attrs['date_issued'] = str(np.datetime64('now')) + 'Z'
    ds.attrs['date_modified'] = ' '
    # ds.attrs['id'] = get_file_id(ds)
    ds.attrs['deployment_name'] = metadata['deployment_name']
    # ds.attrs['title'] = ds.attrs['id']

    dt = ds.time.values
    ds.attrs['time_coverage_start'] = '%s' % dt[0]
    ds.attrs['time_coverage_end'] = '%s' % dt[-1]

    # ds.attrs['processing_level'] = (
    #     'Level 0 (L0) processed data timeseries; ' 'no corrections or data screening'
    # )

    for k, v in sensor_data.items():
        ds.attrs[k] = str(v)
    ds.attrs = collections.OrderedDict(sorted(ds.attrs.items()))

    return ds


def _zero_screen(val):
    val[val == 0] = np.nan
    return val


def nmea2deg(nmea):
    """
    Convert a NMEA float to a decimal degree float.  e.g. -12640.3232 = -126.6721
    """
    deg = np.fix(nmea / 100) + np.sign(nmea) * np.remainder(np.abs(nmea), 100) / 60
    return deg


def oxygen_concentration_correction(ds, ncvar):
    """
    Correct oxygen signal for salinity signal

    Parameters
    ----------
    ds : `xarray.Dataset`
        Should have *oxygen_concentration*, *potential_temperature*, *salinity*,
        on a *time* coordinate.
    ncvar : dict
        dictionary with netcdf variable definitions in it.  Should have
        *oxygen_concentration* as a key, which itself should specify
        a *reference_salinity* and have *correct_oxygen* set to ``"True"``.

    Returns
    -------
    ds : `xarray.Dataset`
        With *oxygen_concentration* corrected for the salinity effect.
    """

    oxy_yaml = ncvar['oxygen_concentration']
    if 'reference_salinity' not in oxy_yaml.keys():
        _log.warning(
            'No reference_salinity found in oxygen deployment yaml. '
            'Assuming reference salinity of 0 psu'
        )
        ref_sal = 0
    else:
        ref_sal = float(oxy_yaml['reference_salinity'])
    _log.info(f'Correcting oxygen using reference salinity {ref_sal} PSU')
    ds_oxy = ds.oxygen_concentration[~np.isnan(ds.oxygen_concentration)]
    # Match the nearest temperature and salinity values from their timestamps
    ds_temp = ds.potential_temperature[~np.isnan(ds.potential_temperature)].reindex(
        time=ds_oxy.time, method='nearest'
    )
    ds_sal = ds.salinity[~np.isnan(ds.salinity)].reindex(
        time=ds_oxy.time, method='nearest'
    )
    o2_sol = gsw.O2sol_SP_pt(ds_sal, ds_temp)
    o2_sat = ds_oxy / gsw.O2sol_SP_pt(ds_sal * 0 + ref_sal, ds_temp)
    ds['oxygen_concentration'].values[~np.isnan(ds.oxygen_concentration)] = (
        o2_sat * o2_sol
    )
    ds['oxygen_concentration'].attrs['oxygen_concentration_QC:RTQC_methodology'] = (
        f'oxygen concentration corrected for salinity using gsw.O2sol_SP_pt with'
        f'salinity and potential temperature from dataset. Original oxygen '
        f'concentration assumed to have been calculated using salinity = '
        f'{ref_sal} PSU'
    )
    return ds


def bar2dbar(val):
    """
    convert val bar to dbar
    """
    return val * 10


def dbar2bar(val):
    """
    convert val dbar to bar
    """
    return val / 10


def _passthrough(val):
    return val


def gappy_fill_vertical(data):
    """
    Fill vertical gaps from the first to last bin with data in them.
    Applied column-wise.

    data = gappy_fill_vertical(data)
    """
    m, n = np.shape(data)
    for j in range(n):
        ind = np.where(~np.isnan(data[:, j]))[0]
        if (
            len(ind) > 0
            and len(ind) < (ind[-1] - ind[0])
            and len(ind) > (ind[-1] - ind[0]) * 0.05
        ):
            int = np.arange(ind[0], ind[-1])
            data[:, j][ind[0] : ind[-1]] = np.interp(int, ind, data[ind, j])
    return data


def find_gaps(sample_time, timebase, maxgap):
    """
    Return an index into *timebase* where True are times in gaps of *sample_time* larger
    than maxgap.
    """
    # figure out which sample each time in time base belongs to:
    time_index = np.searchsorted(sample_time, timebase, side='right')
    time_index = np.clip(time_index, 0, len(sample_time) - 1)

    # figure out the space between sample pairs
    dt = np.concatenate(([0], np.diff(sample_time)))
    # get the gap size for each timebase data point:
    ddt = dt[time_index]

    # get the indices of timebase that are too large and account for the
    # degenerate case when a timebase point falls directly on a sample time.
    index = ~np.logical_or((ddt <= maxgap), (np.isin(timebase, sample_time)))

    return index


def _parse_gliderxml_pos(fname):
    """
    DEPRECATED: use slocum.parse_gliderState instead

    returns lon, lat, timestr
    """

    xmln = fname
    from bs4 import BeautifulSoup

    with open(xmln, 'r') as fin:
        y = BeautifulSoup(fin, features='xml')
        time = None
        lon = None
        lat = None
        for a in y.find_all('valid_location'):
            try:
                dtime = np.datetime64(a.time.text)
                if dtime > np.datetime64('2019-07-19'):
                    if time is not None:
                        time = np.append(time, dtime)
                        lat = np.append(lat, float(a.lat.text))
                        lon = np.append(lon, float(a.lon.text))
                    else:
                        time = np.array(dtime)
                        lat = np.array(float(a.lat.text))
                        lon = np.array(float(a.lon.text))
            except:
                pass
        lon = nmea2deg(lon)
        lat = nmea2deg(lat)
    return lon, lat, time


def _parse_gliderxml_surfacedatetime(fname):
    """
    DEPRECATED: use slocum.parse_gliderState instead

    returns lon, lat, timestr
    """

    xmln = fname
    from bs4 import BeautifulSoup

    with open(xmln, 'r') as fin:
        y = BeautifulSoup(fin, features='xml')
        time = None
        for a in y.find_all('report'):
            _log.debug(a)
            if a.text is not None:
                try:
                    time = np.append(time, np.datetime64(a.text))
                except:
                    pass
    return time


def example_gridplot(
    filename,
    outname,
    toplot=['potential_temperature', 'salinity', 'oxygen_concentration'],
    pdenlevels=np.arange(10, 30, 0.5),
    dpi=200,
    ylim=None,
):
    """
    Smoketest for plotting the netcdf files.
    """

    import matplotlib.pyplot as plt

    ntoplot = len(toplot)
    with xr.open_dataset(filename) as ds:
        fig, axs = plt.subplots(
            nrows=ntoplot,
            constrained_layout=True,
            figsize=(7, 3 * ntoplot),
            sharex=True,
            sharey=True,
        )
        for ax, vname in zip(axs, toplot):
            ds[vname].plot.pcolormesh(ax=ax)
            (ds['potential_density'] - 1000).plot.contour(ax=ax, levels=pdenlevels)
            if ylim:
                ax.set_ylim(ylim)
        fig.savefig(outname, dpi=dpi)


def _get_deployment(deploymentyaml):
    """
    Take the list of files in *deploymentyaml* and parse them
    for deployment information, with subsequent files overwriting
    previous files.
    """
    if isinstance(deploymentyaml, str):
        deploymentyaml = [
            deploymentyaml,
        ]
    deployment = {}
    for nn, d in enumerate(deploymentyaml):
        with open(d) as fin:
            deployment_ = yaml.safe_load(fin)
            for k in deployment_:
                deployment[k] = deployment_[k]

    return deployment


def _any_newer(dirname, filename):
    """
    Check if any files in dirname are newer than filename
    """
    filename = Path(filename)
    dirname = Path(dirname)
    print(filename, filename.exists())
    if not filename.exists():
        return True

    mod_time = filename.stat().st_mtime
    is_newer = False
    for file_path in dirname.iterdir():
        if file_path.is_file():
            if file_path.stat().st_mtime > mod_time:
                is_newer = True
                break

    return is_newer


def _get_glider_name_slocum(current_directory):
    glider = current_directory.parts[-2]
    mission = current_directory.parts[-1]
    print(f'Glider {glider} and mission: {mission}')
    slocum_glider = glider[4:]
    if slocum_glider[-4:-3].isnumeric():
        slocum_glider = slocum_glider[:-4] + '_' + slocum_glider[-4:]
    else:
        slocum_glider = slocum_glider[:-3] + '_' + slocum_glider[-3:]

    return glider, mission, slocum_glider


__all__ = [
    'get_distance_over_ground',
    'get_glider_depth',
    'get_profiles_new',
    'get_derived_eos_raw',
    'fill_metadata',
    'nmea2deg',
    'gappy_fill_vertical',
    'oxygen_concentration_correction',
]
