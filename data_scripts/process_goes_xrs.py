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


def read_goes_xrs(file_name, column='xrsb2_flux'):
    ds = xr.open_dataset(file_name)
    df = ds.to_dataframe()
    df = df.iloc[::4, :]
    df = df.filter([column])
    df = df.reset_index(0).reset_index(drop=True)
    df = df.rename(columns={'time':'datetime'})
    dates = [date.to_pydatetime() for date in df['datetime']]
    values = df[column].to_numpy()
    return dates, values


def read_goes_xrs_dataset(source_dir, column='xrsb2_flux'):
    files = sorted(glob(os.path.join(source_dir, '**', '*.nc'), recursive=True))
    dates = []
    values = []
    for file in tqdm(files):
        d, v = read_goes_xrs(file, column=column)
        dates.extend(d)
        values.append(v)
    values = np.concatenate(values)
    df = pd.DataFrame({'datetime': dates, column: values})
    return df


def main():
    description = 'FDL-X 2024, Radiation Team, GOES X-ray Sensor (XRS) data processor'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--source_dir', type=str, help='Source directory', required=True)
    parser.add_argument('--target_file', type=str, help='Target directory', required=True)
    parser.add_argument('--column', type=str, default='xrsb2_flux', help='Column name')
    
    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    print('Reading GOES XRS data from {}'.format(args.source_dir))
    df = read_goes_xrs_dataset(args.source_dir, column=args.column)
    print('Writing GOES XRS data to {}'.format(args.target_file))
    df.to_csv(args.target_file, index=False)


    print('End time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))



if __name__ == '__main__':
    main()