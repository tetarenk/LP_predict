import pandas  as pd
import datetime
import dateutil
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import calendar
from collections import namedtuple, OrderedDict
import numpy as np



def get_wvm_fromdisk(startdate, enddate):
    """
    This gets the WVM values from the YYYYMMDD.wvm files in
    /jcmtdata/raw/wvm/{date}, and converts them to a pandas dataframe.

    startdate and enddate should be datetime objects. They are inclusive.

    """
    # Both gaps filled with YYYYMMDD
    wvmfilepath = '/jcmtdata/raw/wvm/{date}/{date}.wvm'


    wvmcolumns = [
        'isoTime',#       Time and date string
        'airMass',#       Air Mass
        'tAmb',#          Ambient temperature (kelvin)
        'tSky_0',#       Sky temperature 0 (kelvin)
        'tSky_1',#       Sky temperature 1 (kelvin)
        'tSky_2',#       Sky temperature 2 (kelvin)
        'tWat',#          The effective temperature
        'tOff',#          The line of site opacity
        'mmH2O_a',#       The line-of sight water density in mm
        'mmH2O_z',#       The water density at zenith in mm
        'finalTau',#       The Tau at zenith
        'tSys_0',#       The system temperature of receiver 0
        'tSys_0',#       The system temperature of receiver 1
        'tSys_2',#       The system temperature of receiver 2
        'azimuthDeg',#    Antenna azimuth in degrees
    ]

    # Find all files between start and enddate.
    delta = datetime.timedelta(days=1)
    d = startdate
    dfs = []
    while d <= enddate:
        d_str = d.strftime('%Y%m%d')
        wvmfile = wvmfilepath.format(date=d_str)
        if os.path.isfile(wvmfile):
            dfs.append(pd.read_csv(wvmfile, delim_whitespace=True, names=wvmcolumns,
                                   parse_dates=['isoTime'], infer_datetime_format=True))
        d += delta
    fullwvm = pd.concat(dfs)
    fullwvm = fullwvm.set_index('isoTime')
    return fullwvm


def get_sampled_values(df, column, samplerate='H'):
    sampler = df.resample(samplerate)
    overall = sampler.mean().rename(columns={column : 'mean'})
    overall['median'] = sampler.median()[column]
    overall['count'] = sampler.count()[column]
    return overall


if __name__ == "__main__":
    startdate = datetime.datetime(2010,1,1)
    enddate = datetime.datetime(2018,12,31)
    hourstart = 4#6pm HST
    hourend = 16#6am HST

    wvmrawfile = 'wvmvalues_allone.csv'
    wvm_dailyfile = 'wvmvalues_onepernight.csv'
    ## comment this bit out if you've already doone it --- very slow!
    wvmvalues = get_wvm_fromdisk(startdate, enddate)
    wvmvalues = wvmvalues[['finalTau']]
    with open(wvmrawfile, 'w') as f:
        wvmvalues.to_csv(f)


    wvmvalues = pd.read_csv(wvmrawfile, index_col='isoTime', parse_dates=['isoTime'])


    hours = wvmvalues.index.hour + (wvmvalues.index.minute/60.0)
    nightlywvm = wvmvalues[(hours >=hourstart) & (hours <= hourend)]
    wvms_daily = get_sampled_values(nightlywvm, 'finalTau', samplerate='D')

    with open(wvm_dailyfile, 'w') as f:
        wvms_daily.to_csv(f)
