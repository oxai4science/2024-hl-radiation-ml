import argparse
import sys
import pprint
import time
import datetime

from datasets import SDOMLlite, BioSentinel


def main():
    description = 'FDL-X 2024, Radiation Team, preliminary machine learning experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--sdo_dir', type=str, default='/hdd2-ssd-8T/data/sdoml-lite-biosentinel', help='SDOML-lite-biosentinel directory')
    parser.add_argument('--biosentinel_file', type=str, default='/hdd2-ssd-8T/data/biosentinel/BPD_readings.csv', help='BioSentinel file')

    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    sdo = SDOMLlite(args.sdo_dir)
    biosentinel = BioSentinel(args.biosentinel_file)


    # print(biosentinel[1])
    print()
    print(biosentinel[datetime.datetime.fromisoformat('2022-11-16T11:33:00')])
    print()
    print(biosentinel[datetime.datetime.fromisoformat('2022-11-16T11:34:00')])
    print()
    print(biosentinel[datetime.datetime.fromisoformat('2022-11-16T11:35:00')])    
    print()
    print(biosentinel[datetime.datetime.fromisoformat('2022-11-16T11:36:00')])
    # data = sdo[datetime.datetime.fromisoformat('2022-11-16T10:59:00')]

    print('\nEnd time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()