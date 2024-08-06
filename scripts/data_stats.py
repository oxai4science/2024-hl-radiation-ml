import argparse
import datetime
import pprint
import os
import sys
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from datasets import RadLab, SDOMLlite, GOESXRS, GOESSGPS


matplotlib.use('Agg')

def main():
    description = 'FDL-X 2024, Radiation Team, data statistics'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory with datasets')
    parser.add_argument('--sdo_dir', type=str, default='sdoml-lite-biosentinel', help='SDOML-lite-biosentinel directory')
    parser.add_argument('--radlab_file', type=str, default='radlab/RadLab-20240625-duck.db', help='RadLab file')
    parser.add_argument('--goes_xrs_file', type=str, default='goes/goes-xrs.csv', help='GOES XRS file')
    parser.add_argument('--goes_sgps_file', type=str, default='goes/goes-sgps.csv', help='GOES SGPS file')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to use')
    # parser.add_argument('--instruments', nargs='+', default=['SDOML-lite', 'GOESXRS', 'GOESSGPS', 'BPD', 'CRaTER-D1D2'], help='Instruments')
    parser.add_argument('--instruments', nargs='+', default=['GOESSGPS'], help='Instruments')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')

    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    np.random.seed(args.seed)

    # make sure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)


    data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
    data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)
    data_dir_goes_xrs = os.path.join(args.data_dir, args.goes_xrs_file)
    data_dir_goes_sgps = os.path.join(args.data_dir, args.goes_sgps_file)


    for instrument in args.instruments:
        if instrument == 'SDOML-lite':
            runs = []
            sdo = SDOMLlite(data_dir_sdo)
            for channel in sdo.channels:
                runs.append(('{}_normalized'.format(channel), SDOMLlite(data_dir_sdo, channels=[channel]), ''))
        elif instrument == 'GOESXRS':
            runs = [ 
                ('normalized', GOESXRS(data_dir_goes_xrs, normalize=True), 'xrsb2_flux (normalized)'),
                ('unnormalized', GOESXRS(data_dir_goes_xrs, normalize=False), 'xrsb2_flux')
            ]
        elif instrument == 'GOESSGPS':
            runs = [ 
                ('normalized', GOESSGPS(data_dir_goes_sgps, normalize=True), 'AvgIntProtonFlux (normalized)'),
                ('unnormalized', GOESSGPS(data_dir_goes_sgps, normalize=False), 'AvgIntProtonFlux')
            ]
        else:
            runs = [ 
                ('normalized', RadLab(data_dir_radlab, instrument=instrument, normalize=True), 'Absorbed dose rate (normalized)'),
                ('unnormalized', RadLab(data_dir_radlab, instrument=instrument, normalize=False), 'Absorbed dose rate')
            ]

        for postfix, dataset, label in runs:
            print('\nProcessing {} {}'.format(instrument, postfix))
            if len(dataset) < args.num_samples:
                indices = list(range(len(dataset)))
            else:
                indices = np.random.choice(len(dataset), args.num_samples, replace=False)

            data = []
            for i in indices:
                data.append(dataset[int(i)][0])

            data = torch.stack(data).flatten()
            print('Data shape: {}'.format(data.shape))
            
            data_mean = torch.mean(data)
            data_std = torch.std(data)
            data_min = data.min()
            data_max = data.max()
            print('Mean: {}'.format(data_mean))
            print('Std : {}'.format(data_std))
            print('Min : {}'.format(data_min))
            print('Max : {}'.format(data_max))
        
            file_name_stats = os.path.join(args.target_dir, '{}_{}_data_stats.txt'.format(instrument, postfix))
            print('Saving data stats: {}'.format(file_name_stats))
            with open(file_name_stats, 'w') as f:
                f.write('Mean: {}\n'.format(data_mean))
                f.write('Std : {}\n'.format(data_std))
                f.write('Min : {}\n'.format(data_min))
                f.write('Max : {}\n'.format(data_max))

            file_name_hist = os.path.join(args.target_dir, '{}_{}_data_stats.pdf'.format(instrument, postfix))
            print('Saving histogram : {}'.format(file_name_hist))
            hist_samples = 10000
            indices = np.random.choice(len(data), hist_samples, replace=True)
            hist_data = data[indices]
            plt.figure()
            plt.hist(hist_data, log=True, bins=100)
            plt.tight_layout()
            plt.savefig(file_name_hist)

            if instrument != 'SDOML-lite':
                # plot the whole dataset time series
                dates = []
                values = []
                for i in range(0, len(dataset), len(dataset)//args.num_samples):
                    d = dataset[i]
                    dates.append(d[1])
                    values.append(d[0])

                file_name_ts = os.path.join(args.target_dir, '{}_{}_time_series.pdf'.format(instrument, postfix))
                print('Saving time series: {}'.format(file_name_ts))
                plt.figure(figsize=(24,6))
                plt.plot(dates, values)
                plt.ylabel(label)
                # Limit number of xticks
                plt.xticks(np.arange(0, len(dates), step=len(dates)//40))
                # Rotate xticks
                plt.xticks(rotation=45)
                # Shift xticks so that the end of the text is at the tick
                plt.xticks(ha='right')
                plt.tight_layout()
                plt.savefig(file_name_ts)





    print('\nEnd time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()