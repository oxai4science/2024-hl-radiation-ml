import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from glob import glob
import os
import datetime
from tqdm import tqdm
import pandas as pd
from io import BytesIO
import tarfile


class SDOMLlite(Dataset):
    def __init__(self, data_dir, channels=['hmi_m', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600'], date_start=None, date_end=None):
        self.data_dir = data_dir
        self.channels = channels
        print('\nSDOML-lite')
        print('Directory  : {}'.format(self.data_dir))

        self.date_start, self.date_end = self.find_date_range()
        if date_start is not None:
            date_start = datetime.datetime.fromisoformat(date_start)
            if (date_start >= self.date_start) and (date_start < self.date_end):
                self.date_start = date_start
            else:
                print('Start date out of range, using default')
        if date_end is not None:
            date_end = datetime.datetime.fromisoformat(date_end)
            if (date_end > self.date_start) and (date_end <= self.date_end):
                self.date_end = date_end
            else:
                print('End date out of range, using default')
        self.delta_minutes = 15
        total_minutes = int((self.date_end - self.date_start).total_seconds() / 60)
        total_steps = total_minutes // self.delta_minutes
        print('Start date : {}'.format(self.date_start))
        print('End date   : {}'.format(self.date_end))
        print('Delta      : {} minutes'.format(self.delta_minutes))
        print('Channels   : {}'.format(', '.join(self.channels)))

        # convert 2022-11-06 00:01:00+00:00 to datetime.datetime(2022, 11, 6, 0, 1)
        datetime.datetime.fromisoformat('2022-11-06T00:01:00+00:00')


        self.dates = []
        dates_cache = os.path.join(self.data_dir, 'dates_cache_{}_{}_{}'.format('_'.join(self.channels), self.date_start.isoformat(), self.date_end.isoformat()))
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

        self.dates_set = set(self.dates)

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

        if date not in self.dates_set:
            print('Date not found in SDOML-lite : {}'.format(date))
            return None

        channels = []
        for channel in self.channels:
            file = os.path.join(self.data_dir, date.strftime('%Y/%m/%d/%H%M') +'.'+channel+'.npy')
            channel_data = np.load(file)
            channels.append(channel_data)
        channels = np.stack(channels)
        channels = torch.from_numpy(channels)
        return channels


class BioSentinel(Dataset):
    def __init__(self, data_file, date_start='2022-11-16T11:00:00', date_end='2024-05-14T19:30:00', normalize=True):
        self.data_file = data_file
        self.date_start = datetime.datetime.fromisoformat(date_start)
        self.date_end = datetime.datetime.fromisoformat(date_end)
        self.delta_minutes = 1
        self.normalize = normalize

        print('\nBioSentinel')
        print('Data file : {}'.format(self.data_file))
        print('Start date: {}'.format(self.date_start))
        print('End date  : {}'.format(self.date_end))
        print('Delta     : {} minutes'.format(self.delta_minutes))
        self.data = self.process_data(data_file)
        self.dates = [date.to_pydatetime() for date in self.data['datetime']]
        self.dates_set = set(self.dates)

    def process_data(self, data_file):
        data = pd.read_csv(data_file)
        print('Rows before filtering       : {:,}'.format(len(data)))
        data['datetime'] = pd.to_datetime(data['timestamp_utc']).dt.tz_localize(None)

        # filter out rows before start date and after end date
        data = data[(data['datetime'] >=self.date_start) & (data['datetime'] <=self.date_end)]
        print('Rows after date filter      : {:,}'.format(len(data)))

        # erase all columns except absorbed_dose_rate and datetime
        data = data[['datetime', 'absorbed_dose_rate']]
        
        # remove all rows with 0 absorbed_dose_rate
        data = data[data['absorbed_dose_rate'] > 0]
        print('Rows after removing 0s      : {:,}'.format(len(data)))

        # q_low = data['absorbed_dose_rate'].quantile(0.01)
        # q_hi  = data['absorbed_dose_rate'].quantile(0.99)
        # data = data[(data['absorbed_dose_rate'] < q_hi) & (data['absorbed_dose_rate'] > q_low)]
        # print('Rows after removing outliers: {:,}'.format(len(data)))

        data['absorbed_dose_rate'] = data['absorbed_dose_rate'].astype(np.float32)

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

    def normalize_data(self, data):
        return torch.log(data + 1e-8)
    
    def unnormalize_data(self, data):
        return torch.exp(data) - 1e-8

    def get_data(self, date):
        if date < self.date_start or date > self.date_end:
            raise ValueError('Date ({}) out of range for BioSentinel ({} - {})'.format(date, self.date_start, self.date_end))        

        if date not in self.dates_set:
            print('Date not found in BioSentinel : {}'.format(date))
            return None

        data = self.data[self.data['datetime'] == date]['absorbed_dose_rate']
        if len(data) == 0:
            raise RuntimeError('Should not happen')
        data = torch.tensor(data.values[0])
        if self.normalize:
            data = self.normalize_data(data)

        return data


class Sequences(Dataset):
    def __init__(self, datasets, delta_minutes=1, sequence_length=10):
        super().__init__()
        self.datasets = datasets
        self.delta_minutes = delta_minutes
        self.sequence_length = sequence_length

        self.date_start = max([dataset.date_start for dataset in self.datasets])
        self.date_end = min([dataset.date_end for dataset in self.datasets])
        if self.date_start > self.date_end:
            raise ValueError('No overlapping date range between datasets')
        self.sequences = self.find_sequences()

        print('\nSequences')
        print('Start date              : {}'.format(self.date_start))
        print('End date                : {}'.format(self.date_end))
        print('Delta                   : {} minutes'.format(self.delta_minutes))
        print('Sequence length         : {}'.format(self.sequence_length))
        print('Number of sequences     : {:,}'.format(len(self.sequences)))

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, index):
        # print('constructing sequence')
        sequence = self.sequences[index]

        all_data = []
        for dataset in self.datasets:
            data = []
            for date in sequence:
                d, _ = dataset[date]
                data.append(d)
            data = torch.stack(data)
            all_data.append(data)
        all_data.append([str(date) for date in sequence])
        # print('done constructing sequence')
        return tuple(all_data)


    def find_sequences(self):
        sequences = []
        sequence_start = self.date_start
        while sequence_start < self.date_end - datetime.timedelta(minutes=self.sequence_length*self.delta_minutes):
            sequence = []
            sequence_available = True
            for i in range(self.sequence_length):
                date = sequence_start + datetime.timedelta(minutes=i*self.delta_minutes)
                for dataset in self.datasets:
                    if date not in dataset.dates_set:
                        sequence_available = False
                        break
                if not sequence_available:
                    break
                sequence.append(date)
            if sequence_available:
                sequences.append(sequence)
            sequence_start += datetime.timedelta(minutes=self.delta_minutes)
        return sequences


class TarRandomAccess():
    def __init__(self, tar_files):
        self.index = {}
        for tar_file in tar_files:
            with tarfile.open(tar_file) as tar:
                for info in tar.getmembers():
                    self.index[info.name] = (tar.name, info)
        self.file_names = list(self.index.keys())

    def __getitem__(self, file_name):
        d = self.index.get(file_name)
        if d is None:
            return None
        tar_name, info = d
        with tarfile.open(tar_name) as tar:
            data = BytesIO(tar.extractfile(info).read())
        return data

    
class WebDataset():
    def __init__(self, data_dir, decode_func=None):
        tar_files = glob(os.path.join(data_dir, '*.tar'))
        self.tars = TarRandomAccess(tar_files)
        if decode_func is None:
            self.decode_func = self.decode
        else:
            self.decode_func = decode_func
        
        self.samples = {}
        self.prefixes = []
        for file_name in self.tars.file_names:
            p = file_name.split('.', 1)
            if len(p) == 2:
                prefix, postfix = p
                if prefix not in self.samples:
                    self.samples[prefix] = []
                    self.prefixes.append(prefix)
                self.samples[prefix].append(postfix)

    def decode(self, data, file_name):
        if file_name.endswith('.npy'):
            data = np.load(data)
        else:
            raise ValueError('Unknown data type for file: {}'.format(file_name))    
        return data
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            prefix = self.prefixes[index]
        elif isinstance(index, str):
            prefix = index
        else:
            raise ValueError('Expecting index to be int or str')
        sample = self.samples.get(prefix)
        if sample is None:
            return None
        
        data = {}
        for postfix in sample:
            file_name = prefix + '.' + postfix
            d = self.decode(self.tars[file_name], file_name)
            data[postfix] = d
        return data                
            