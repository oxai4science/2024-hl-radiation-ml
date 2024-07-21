import argparse
import datetime
import pprint
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from datasets import BioSentinel


matplotlib.use('Agg')

def main():
    description = 'FDL-X 2024, Radiation Team, data statistics'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--biosentinel_file', type=str, default='/hdd2-ssd-8T/data/biosentinel/BPD_readings.csv', help='BioSentinel file')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of samples to use')

    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    # make sure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)


    prefix = 'biosentinel'

    # load the dataset
    runs = [ 
        ('unnormalized', BioSentinel(args.biosentinel_file, normalize=False)),  
        ('normalized', BioSentinel(args.biosentinel_file, normalize=True))
    ]

    for postfix, dataset in runs:
        print('\nProcessing {} {}'.format(prefix, postfix))
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)

        data = []
        for i in indices:
            data.append(dataset[int(i)][0])

        data = np.array(data)
        
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
    
        file_name_stats = os.path.join(args.target_dir, '{}_data_stats_{}.txt'.format(prefix, postfix))
        print('Saving data stats: {}'.format(file_name_stats))
        with open(file_name_stats, 'w') as f:
            f.write('Mean: {}\n'.format(data_mean))
            f.write('Std : {}\n'.format(data_std))
            f.write('Min : {}\n'.format(data_min))
            f.write('Max : {}\n'.format(data_max))

        file_name_hist = os.path.join(args.target_dir, '{}_data_stats_{}.pdf'.format(prefix, postfix))
        print('Saving histogram : {}'.format(file_name_hist))
        plt.figure()
        plt.hist(data.flatten(), log=True, bins=100)
        plt.tight_layout()
        plt.savefig(file_name_hist)

        # plot the whole dataset time series
        dates = []
        values = []
        for i in range(0, len(dataset), len(dataset)//args.num_samples):
            d = dataset[i]
            dates.append(d[1])
            values.append(d[0])

        file_name_ts = os.path.join(args.target_dir, '{}_time_series_{}.pdf'.format(prefix, postfix))
        print('Saving time series: {}'.format(file_name_ts))
        plt.figure(figsize=(24,6))
        plt.plot(dates, values)
        plt.ylabel('Absorbed dose rate')
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