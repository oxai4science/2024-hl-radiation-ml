import argparse
import sys
import pprint
import time
import datetime
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split

from datasets import SDOMLlite, BioSentinel, Sequences


def seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    description = 'FDL-X 2024, Radiation Team, preliminary machine learning experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--sdo_dir', type=str, default='/hdd2-ssd-8T/data/sdoml-lite-biosentinel', help='SDOML-lite-biosentinel directory')
    parser.add_argument('--biosentinel_file', type=str, default='/hdd2-ssd-8T/data/biosentinel/BPD_readings.csv', help='BioSentinel file')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Delta minutes')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=0, help='Random number generator seed')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--log_interval', type=int, default=100, help='Log interval')
    parser.add_argument('--valid_proportion', type=float, default=0.1, help='Proportion of data to use for validation')

    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    seed(args.seed)

    sdo = SDOMLlite(args.sdo_dir)
    biosentinel = BioSentinel(args.biosentinel_file)
    sequences = Sequences([sdo, biosentinel], delta_minutes=15, sequence_length=10)

    valid_size = int(len(sequences) * args.valid_proportion)
    train_size = len(sequences) - valid_size
    train_dataset, valid_dataset = random_split(sequences, [train_size, valid_size])
    print('\nTrain size: {:,}'.format(len(train_dataset)))
    print('Valid size: {:,}'.format(len(valid_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    for sdo, biosentinel, date in train_loader:
        print(date)




    print('\nEnd time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()