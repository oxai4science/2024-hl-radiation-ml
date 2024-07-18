import webdataset as wds
import wids

import torch
from torch.utils.data import Dataset
import numpy as np
from glob import glob
import os
import datetime
from tqdm import tqdm
import pandas as pd


class SDOMLlite(Dataset):
    def __init__(self, data_dir, channels=['hmi_m', 'aia_0094', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600']):
        self.data_dir = data_dir
        self.channels = channels
        index_file = glob(os.path.join(data_dir, '*.json'))
        if len(index_file) == 0:
            raise RuntimeError('No index file (.json) found')
        index_file = index_file[0]
        print('\nSDOML-lite')
        print('Directory  : {}'.format(self.data_dir))
        print('Index      : {}'.format(index_file))
        self.webdataset = wids.ShardListDataset(index_file)

        date_start, date_end = self.find_date_range()
        self.date_start = date_start
        self.date_end = date_end
        self.delta_minutes = 15
        print('Start date : {}'.format(self.date_start))
        print('End date   : {}'.format(self.date_end))
        print('Delta      : {} minutes'.format(self.delta_minutes))
        print('Channels   : {}'.format(', '.join(self.channels)))
        
        self.channels_webdataset_keys = ['.'+c+'.npy' for c in self.channels]
        
        self.date_to_index = {}
        self.dates = []
        dates_cache = os.path.join(self.data_dir, 'dates_cache_{}'.format('_'.join(self.channels)))
        if os.path.exists(dates_cache):
            print('Loading dates from cache: {}'.format(dates_cache))
            self.dates, self.date_to_index = torch.load(dates_cache)
        else:
            for i in tqdm(range(len(self.webdataset)), desc='Checking complete channels'):
                cs = self.webdataset[i].keys()
                has_all_channels = True
                for c in self.channels_webdataset_keys:
                    if c not in cs:
                        has_all_channels = False
                        break
                if has_all_channels:
                    date = self.get_date(i)
                    self.dates.append(date)
                    self.date_to_index[date] = i
            print('Saving dates to cache: {}'.format(dates_cache))
            torch.save((self.dates, self.date_to_index), dates_cache)            

        if len(self.dates) == 0:
            raise RuntimeError('No frames found with given list of channels')
                
        print('Frames total    : {:,}'.format(len(self.webdataset)))
        print('Frames available: {:,}'.format(len(self.dates)))
        print('Frames dropped  : {:,}'.format(len(self.webdataset) - len(self.dates)))                
           
    def get_date(self, index):
        return datetime.datetime.strptime(self.webdataset[index]['__key__'], '%Y/%m/%d/%H%M')
    
    def find_date_range(self):
        date_start = self.get_date(0)
        date_end = self.get_date(len(self.webdataset)-1) # wids doesn't support -1 indexing
        return date_start, date_end
    
    def __len__(self):
        return len(self.dates)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            date = self.dates[index]
        elif isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        else:
            raise ValueError('Expecting index to be int, datetime.datetime, or str (in the format of 2022-11-01T00:01:00)')
        data = self.get_data(date)    
        return data, date.isoformat()
    
    def get_data(self, date):
        if date < self.date_start or date > self.date_end:
            raise ValueError('Date () out of range for SDOML-lite ({} - {})'.format(date, self.date_start, self.date_end))
        if date not in self.date_to_index:
            print('Date not found in SDOML-lite: {}'.format(date))
            # Adjust the date to the previous minute that is a multiple of 15
            date = date.replace(second=0, microsecond=0)
            date -= datetime.timedelta(minutes=date.minute % 15)
            print('Adjusted date               : {}'.format(date))
            
        index = self.date_to_index[date]
        data = self.webdataset[index]
        channels = []
        for c in self.channels_webdataset_keys:
            channels.append(data[c])
        channels = np.stack(channels)
        channels = torch.from_numpy(channels)
        return channels
    


class BioSentinel(Dataset):
    def __init__(self, data_file, date_start='2022-11-16T11:00:00', date_end='2024-05-14T19:30:00'):
        self.data_file = data_file
        self.date_start = datetime.datetime.fromisoformat(date_start)
        self.date_end = datetime.datetime.fromisoformat(date_end)
        self.delta_minutes = 1

        print('\nBioSentinel')
        print('Data file : {}'.format(self.data_file))
        print('Start date: {}'.format(self.date_start))
        print('End date  : {}'.format(self.date_end))
        print('Delta     : {} minutes'.format(self.delta_minutes))
        self.data = self.process_data(data_file)


    def process_data(self, data_file):
        data = pd.read_csv(data_file)
        print('Rows before filtering: {:,}'.format(len(data)))
        data['datetime'] = pd.to_datetime(data['timestamp_utc'])
        # make np.datetime64
        data['datetime'] = data['datetime'].values.astype('datetime64[m]')

        # filter out rows before start date and after end date
        data = data[(data['datetime'] >= self.date_start) & (data['datetime'] <= self.date_end)]
        # erase all columns except absorbed_dose_rate and datetime
        data = data[['datetime', 'absorbed_dose_rate']]
        
        # remove all rows with 0 absorbed_dose_rate
        data = data[data['absorbed_dose_rate'] > 0]

        data['absorbed_dose_rate'] = data['absorbed_dose_rate'].astype(np.float32)

        print('Rows after  filtering: {:,}'.format(len(data)))
        return data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            date = self.data.iloc[index]['datetime']
        elif isinstance(index, datetime.datetime):
            date = index
        elif isinstance(index, str):
            date = datetime.datetime.fromisoformat(index)
        else:
            raise ValueError('Expecting index to be int, datetime.datetime, or str (in the format of 2022-11-01T00:01:00)')
        data = self.get_data(date)    
        return data, date.isoformat()

    def get_data(self, date):
        if date < self.date_start or date > self.date_end:
            raise ValueError('Date () out of range for BioSentinel ({} - {})'.format(date, self.date_start, self.date_end))        

        data = self.data[self.data['datetime'] == date]['absorbed_dose_rate']
        if len(data) == 0:
            print('Date not found in BioSentinel: {}'.format(date))
            # find the date in datetime column that is previous to the given date
            date = self.data[self.data['datetime'] < date]['datetime'].max()
            print('Adjusted date                : {}'.format(date))
            data = self.data[self.data['datetime'] == date]['absorbed_dose_rate']
        data = data.values[0]
        return data

