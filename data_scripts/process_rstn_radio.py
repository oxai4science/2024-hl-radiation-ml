import argparse
import pprint
import sys
import datetime
import time
import os
import urllib.request
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
from glob import glob
import gzip
#K7OL20090701084648      0      0      5      0      3      0      0     10
#012345678901234567890123456789012345678901234567890123456789012345678901234567890
#          1         2         3         4         5         6         7         8

# Format information
# https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-radio/rstn-1-second/docs/rstn-format-new.txt
# 
#         New  R S T N  FORMAT (LEARMONTH)  Nov 1999
# ===========================================================================
# Column  Fmt   Description 
# ---------------------------------------------------------------------------
#  1- 4   I4    4 letter station abbreviation LEAR (or APLM)
#  5- 8   I4    Year
#  9-10   I2    Month
# 11-12   I2    Day of month
# 13-18   I6    Universal Time of measurement in hours, minutes and seconds
# 19-24   I6    Frequency 245 MHz
# 25-30   I6    Frequency 410 MHz
# 31-36   I6    Frequency 610 MHz
# 37-42   I6    Frequency 1415 MHz
# 43-48   I6    Frequency 2695 MHz
# 49-54   I6    Frequency 4995 MHz
# 55-60   I6    Frequency 8800 MHz
# 61-66   I6    Frequency 15400 MHz
# ---------------------------------------------------------------------------

def read_rstn_radio(file_name):
    
    # open file_name as a text file
    dates = []
    values245 = []
    values410 = []
    values610 = []
    values1415 = []
    values2695 = []
    values4995 = []
    values8800 = []
    values15400 = []

    with gzip.open(file_name, 'rt') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if len(line) != 75:
                print('File {}\nskipping line: {}\n'.format(file_name, line))
                continue
            year = int(line[4:8])
            month = int(line[8:10])
            day = int(line[10:12])
            time = line[12:18]
            hour = int(time[:2])
            minute = int(time[2:4])
            second = int(time[4:6])
            dates.append(datetime.datetime(year, month, day, hour, minute, second))
            try:
                v245 = int(line[18:25])
            except:
                v245 = 0
            try:
                v410 = int(line[25:32])
            except:
                v410 = 0
            try:
                v610 = int(line[32:39])
            except:
                v610 = 0
            try:
                v1415 = int(line[39:46])
            except:
                v1415 = 0
            try:
                v2695 = int(line[46:53])
            except:
                v2695 = 0
            try:
                v4995 = int(line[53:60])
            except:
                v4995 = 0
            try:
                v8800 = int(line[60:67])
            except:
                v8800 = 0
            try:
                v15400 = int(line[67:74])
            except:
                v15400 = 0
            values245.append(v245)
            values410.append(v410)
            values610.append(v610)
            values1415.append(v1415)
            values2695.append(v2695)
            values4995.append(v4995)
            values8800.append(v8800)
            values15400.append(v15400)

    # make pandas dataframe
    df = pd.DataFrame({'datetime': dates, '245MHz': values245, '410MHz': values410, '610MHz': values610, '1415MHz': values1415, '2695MHz': values2695, '4995MHz': values4995, '8800MHz': values8800, '15400MHz': values15400}, index=dates)
    # data in df is given for each second, we need to resample it to 1 minute
    df = df.resample('1min', on='datetime').mean()

    dates = df.index.to_list()
    values245 = df['245MHz'].to_numpy()
    values410 = df['410MHz'].to_numpy()
    values610 = df['610MHz'].to_numpy()
    values1415 = df['1415MHz'].to_numpy()
    values2695 = df['2695MHz'].to_numpy()
    values4995 = df['4995MHz'].to_numpy()
    values8800 = df['8800MHz'].to_numpy()
    values15400 = df['15400MHz'].to_numpy()

    return dates, values245, values410, values610, values1415, values2695, values4995, values8800, values15400


def read_rstn_radio_dataset(source_dir):
    files = sorted(glob(os.path.join(source_dir, '**', '*.gz'), recursive=True))
    dates = []
    values245 = []
    values410 = []
    values610 = []
    values1415 = []
    values2695 = []
    values4995 = []
    values8800 = []
    values15400 = []
    for file in tqdm(files):
        d, v245, v410, v610, v1415, v2695, v4995, v8800, v15400 = read_rstn_radio(file)
        dates.extend(d)
        values245.extend(v245)
        values410.extend(v410)
        values610.extend(v610)
        values1415.extend(v1415)
        values2695.extend(v2695)
        values4995.extend(v4995)
        values8800.extend(v8800)
        values15400.extend(v15400)
    values245 = np.array(values245)
    values410 = np.array(values410)
    values610 = np.array(values610)
    values1415 = np.array(values1415)
    values2695 = np.array(values2695)
    values4995 = np.array(values4995)
    values8800 = np.array(values8800)
    values15400 = np.array(values15400)
    df = pd.DataFrame({'datetime': dates, '245MHz': values245, '410MHz': values410, '610MHz': values610, '1415MHz': values1415, '2695MHz': values2695, '4995MHz': values4995, '8800MHz': values8800, '15400MHz': values15400})
    return df


def main():
    description = 'FDL-X 2024, Radiation Team, Radio Solar Telescope Network (RSTN) Solar Radio Burst data processor'
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

    print('Reading RSTN data from {}'.format(args.source_dir))
    df = read_rstn_radio_dataset(args.source_dir)
    print('Writing RSTN data to {}'.format(args.target_file))
    df.to_csv(args.target_file, index=False)


    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()