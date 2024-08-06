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


def read_goes_sgps(file_name, column='AvgDiffProtonFlux'):
    ds = xr.open_dataset(file_name)
    df = ds.to_dataframe()
    if 'v1-0-1' in file_name or 'v2-0-0' in file_name:
        df = df.reset_index().drop_duplicates(subset=['L2_SciData_TimeStamp'], keep='first').rename(columns={'L2_SciData_TimeStamp':'datetime'}).filter(['datetime',column])
    elif 'v3-0-0' in file_name or 'v3-0-1' in file_name or 'v3-0-2' in file_name:
        df = df.reset_index().drop_duplicates(subset=['time'], keep='first').rename(columns={'time':'datetime'}).filter(['datetime',column])
    else:
        raise RuntimeError('Unknown file version: {}'.format(file_name))
    dates = [date.to_pydatetime() for date in df['datetime']]
    values = df[column].to_numpy()
    return dates, values


def read_goes_sgps_dataset(source_dir, column='AvgDiffProtonFlux'):
    files = sorted(glob(os.path.join(source_dir, '**', '*.nc'), recursive=True))
    dates = []
    values = []
    for file in tqdm(files):
        d, v = read_goes_sgps(file, column=column)
        dates.extend(d)
        values.append(v)
    values = np.concatenate(values)
    df = pd.DataFrame({'datetime': dates, column: values})
    return df


def main():
    description = 'FDL-X 2024, Radiation Team, GOES Solar and Galactic Proton Sensors (SGPS) data processor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_file', type=str, help='Target directory', required=True)
    parser.add_argument('--column', type=str, default='AvgDiffProtonFlux', help='Column name')
    
    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    print('Reading GOES SGPS data from {}'.format(args.source_dir))
    df = read_goes_sgps_dataset(args.source_dir, column=args.column)
    print('Writing GOES SGPS data to {}'.format(args.target_file))
    df.to_csv(args.target_file, index=False)


    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()