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
        print('\nSDOML-lite')
        print('Directory  : {}'.format(self.data_dir))

        self.date_start, self.date_end = self.find_date_range()
        self.delta_minutes = 15
        total_minutes = int((self.date_end - self.date_start).total_seconds() / 60)
        total_steps = total_minutes // self.delta_minutes
        print('Start date : {}'.format(self.date_start))
        print('End date   : {}'.format(self.date_end))
        print('Delta      : {} minutes'.format(self.delta_minutes))
        print('Channels   : {}'.format(', '.join(self.channels)))

        self.dates = []
        dates_cache = os.path.join(self.data_dir, 'dates_cache_{}'.format('_'.join(self.channels)))
        if os.path.exists(dates_cache):
            print('Loading dates from cache: {}'.format(dates_cache))
            self.dates = torch.load(dates_cache)
        else:        
            for i in tqdm(range(total_steps), desc='Checking complete channels'):
                date = self.date_start + datetime.timedelta(minutes=self.delta_minutes*i)
                exists = True
                for channel in self.channels:
                    file = os.path.join(self.data_dir, date.strftime('%Y/%m/%d/%H%M') +'.'+channel+'.npy')
                    if not os.path.exists(file):
                        exists = False
                        break
                if exists:
                    self.dates.append(date)
            print('Saving dates to cache: {}'.format(dates_cache))
            torch.save(self.dates, dates_cache)

        if len(self.dates) == 0:
            raise RuntimeError('No frames found with given list of channels')
        
        print('Frames total    : {:,}'.format(total_steps))
        print('Frames available: {:,}'.format(len(self.dates)))
        print('Frames dropped  : {:,}'.format(total_steps - len(self.dates)))

    def find_date_range(self):
        all_files = sorted(glob(os.path.join(self.data_dir,'**','*.npy'), recursive=True))
        if len(all_files) == 0:
            raise RuntimeError('No .npy files found in the directory')
        date_start = datetime.datetime.strptime(os.path.relpath(all_files[0], self.data_dir).split('.')[0], '%Y/%m/%d/%H%M')
        date_end = datetime.datetime.strptime(os.path.relpath(all_files[-1], self.data_dir).split('.')[0], '%Y/%m/%d/%H%M')
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
            raise ValueError('Date ({}) out of range for SDOML-lite ({} - {})'.format(date, self.date_start, self.date_end))

        if date not in self.dates:
            print('Date not found in SDOML-lite : {}'.format(date))
            # Adjust the date to the previous minute that is a multiple of 15
            date = date.replace(second=0, microsecond=0)
            date -= datetime.timedelta(minutes=date.minute % 15)
            time_out = 10
            while date not in self.dates:
                date -= datetime.timedelta(minutes=15)
                time_out -= 1
                if time_out == 0:
                    raise ValueError('Timeout while searching for date in SDOML-lite: {}'.format(date))
            print('Adjusted date                : {}'.format(date))    

        channels = []
        for channel in self.channels:
            file = os.path.join(self.data_dir, date.strftime('%Y/%m/%d/%H%M') +'.'+channel+'.npy')
            channel_data = np.load(file)
            channels.append(channel_data)
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
            raise ValueError('Date ({}) out of range for BioSentinel ({} - {})'.format(date, self.date_start, self.date_end))        

        data = self.data[self.data['datetime'] == date]['absorbed_dose_rate']
        if len(data) == 0:
            print('Date not found in BioSentinel: {}'.format(date))
            # find the date in datetime column that is previous to the given date
            date = self.data[self.data['datetime'] < date]['datetime'].max()
            print('Adjusted date                : {}'.format(date))
            data = self.data[self.data['datetime'] == date]['absorbed_dose_rate']
        data = torch.tensor(data.values[0])
        return data


class Sequences(Dataset):
    def __init__(self, datasets, delta_minutes=1, sequence_length=10):
        self.datasets = datasets
        self.delta_minutes = delta_minutes
        self.sequence_length = sequence_length

        self.date_start = max([dataset.date_start for dataset in self.datasets])
        self.date_end = min([dataset.date_end for dataset in self.datasets])
        if self.date_start > self.date_end:
            raise ValueError('No overlapping date range between datasets')
        self.length = int(((self.date_end - self.date_start).total_seconds() / 60) // self.delta_minutes) - self.sequence_length + 1

        print('\nSequences')
        print('Start date      : {}'.format(self.date_start))
        print('End date        : {}'.format(self.date_end))
        print('Delta           : {} minutes'.format(self.delta_minutes))
        print('Sequence length : {}'.format(self.sequence_length))
        print('Total sequences : {:,}'.format(self.length))

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        if not isinstance(index, int):
            raise ValueError('Expecting index to be an integer')

        sequence_start = self.date_start + datetime.timedelta(minutes=index*self.delta_minutes)
        all_data = []
        for dataset in self.datasets:
            data = []
            for i in range(self.sequence_length):
                date = sequence_start + datetime.timedelta(minutes=i*self.delta_minutes)
                d, _ = dataset[date]
                data.append(d)
            data = torch.stack(data)
            all_data.append(data)
        all_data.append('{} - {}'.format(sequence_start, sequence_start + datetime.timedelta(minutes=self.sequence_length*self.delta_minutes)))
        return tuple(all_data)