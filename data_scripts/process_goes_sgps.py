import argparse
import pprint
import sys
import datetime
import time
import os
import urllib.request
from tqdm import tqdm
import netCDF4
import pandas as pd
import numpy as np
from glob import glob

# Relevant documentation: https://www.noaasis.noaa.gov/pdf/ps-pvr/goes17/SEISS/SGPS/Provisional/GOES-17_SEISS_SGPS_Provisional_ReadMe.pdf
# 
# v3-0-0 file format, extracted from a netCDF sample file
#
# dimensions(sizes): 
# time(1440)
# diff_channels(13)
# sensor_units(2)
# diff_alpha_channels(11)
#
# variables(dimensions): 
# float64 time(time)
# uint32 L1bRecordsInAvg(time)
# uint8 yaw_flip_flag(time)
# float32 AvgDiffProtonFlux(time, sensor_units, diff_channels)
# float32 AvgDiffProtonFluxObserved(time, sensor_units, diff_channels)
# float32 AvgDiffProtonFluxUncert(time, sensor_units, diff_channels)
# uint32 DiffValidL1bSamplesInAvg(time, sensor_units, diff_channels)
# uint32 DiffDQFdtcSum(time, sensor_units, diff_channels)
# uint32 DiffDQFoobSum(time, sensor_units, diff_channels)
# uint32 DiffDQFerrSum(time, sensor_units, diff_channels)
# float32 AvgIntProtonFlux(time, sensor_units)
# float32 AvgIntProtonFluxObserved(time, sensor_units)
# float32 AvgIntProtonFluxUncert(time, sensor_units)
# uint32 IntValidL1bSamplesInAvg(time, sensor_units)
# uint32 IntDQFdtcSum(time, sensor_units)
# uint32 IntDQFoobSum(time, sensor_units)
# uint32 IntDQFerrSum(time, sensor_units)
# float32 DiffProtonLowerEnergy(sensor_units, diff_channels)
# float32 DiffProtonUpperEnergy(sensor_units, diff_channels)
# float32 DiffProtonEffectiveEnergy(sensor_units, diff_channels)
# float32 IntegralProtonEffectiveEnergy(sensor_units)
# uint8 ExpectedLUTNotFound()
# float32 AvgDiffAlphaFlux(time, sensor_units, diff_alpha_channels)
# float32 AvgDiffAlphaFluxObserved(time, sensor_units, diff_alpha_channels)
# float32 AvgDiffAlphaFluxUncert(time, sensor_units, diff_alpha_channels)
# float32 DiffAlphaLowerEnergy(sensor_units, diff_alpha_channels)
# float32 DiffAlphaUpperEnergy(sensor_units, diff_alpha_channels)
# float32 DiffAlphaEffectiveEnergy(sensor_units, diff_alpha_channels)
# uint8 DiffProtonIgnoredL1bDQFs(time, sensor_units, diff_channels)
# uint8 IntProtonIgnoredL1bDQFs(time, sensor_units)
# uint8 DiffAlphaIgnoredL1bDQFs(time, sensor_units, diff_alpha_channels)


def j2000_to_datetime(j2000):
    return datetime.datetime(2000, 1, 1, 12, 0) + datetime.timedelta(seconds=j2000)


def read_goes_sgps(file_name):
    # 0  p1  1.0 - 1.9 MeV
    # 1  p2a 1.9 - 2.3 MeV
    # 2  p2b 2.3 - 3.4 MeV
    # 3  p3  3.4 - 6.5 MeV
    # 4  p4  6.5 - 12  MeV
    # 5  p5  12  - 25  MeV
    # 6  p6  25  - 40  MeV
    # 7  p7  40  - 80  MeV
    # 8  p8a 83  - 99  MeV
    # 9  p8b 99  - 118 MeV
    # 10 p8c 118 - 150 MeV
    # 11 p9  150 - 275 MeV
    # 12 p10 275 - 500 MeV
    # 13 p11 > 500 MeV
    nc = netCDF4.Dataset(file_name)

    # diff_channel_names = ['p1','p2','p2a','p2b','p3','p4','p5','p6','p7','p8a','p8b','p8c','p9','p10']
    diff_channels = []
    for i in range(13):
        diff_channels.append(nc['AvgDiffProtonFlux'][:,0,i])
    diff_channels = np.stack(diff_channels)
    # p11 = nc['AvgIntProtonFlux'][:,0]
    gt10MeV = diff_channels[4:].sum(0) # + p11
    gt100MeV = diff_channels[9:].sum(0) # + p11
    if 'v1-0-1' in file_name or 'v2-0-0' in file_name:
        dates = [j2000_to_datetime(j2000) for j2000 in nc['L2_SciData_TimeStamp'][:]]
    elif 'v3-0-0' in file_name or 'v3-0-1' in file_name or 'v3-0-2' in file_name:
        dates = [j2000_to_datetime(j2000) for j2000 in  nc['time'][:]]
    else:
        raise ValueError('Unknown file format: {}'.format(file_name))
    return dates, gt10MeV, gt100MeV


def read_goes_sgps_dataset(source_dir):
    files = sorted(glob(os.path.join(source_dir, '**', '*.nc'), recursive=True))
    dates = []
    values10 = []
    values100 = []
    for file in tqdm(files):
        d, v10, v100 = read_goes_sgps(file)
        dates.extend(d)
        values10.append(v10)
        values100.append(v100)
    values10 = np.concatenate(values10)
    values100 = np.concatenate(values100)
    df = pd.DataFrame({'datetime': dates, '>10MeV': values10, '>100MeV': values100})
    return df


def main():
    description = 'FDL-X 2024, Radiation Team, GOES Solar and Galactic Proton Sensors (SGPS) data processor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_file', type=str, help='Target directory', required=True)
    
    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    print('Reading GOES SGPS data from {}'.format(args.source_dir))
    df = read_goes_sgps_dataset(args.source_dir)
    print('Writing GOES SGPS data to {}'.format(args.target_file))
    df.to_csv(args.target_file, index=False)


    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()