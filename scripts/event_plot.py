import argparse
import datetime
import pprint
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

from datasets import SDOMLlite, RadLab


matplotlib.use('Agg')

def main():
    description = 'FDL-X 2024, Radiation Team, data statistics'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory with datasets')
    parser.add_argument('--sdo_dir', type=str, default='sdoml-lite-biosentinel', help='SDOML-lite-biosentinel directory')
    parser.add_argument('--radlab_file', type=str, default='radlab/RadLab-20240625-duck.db', help='RadLab file')
    parser.add_argument('--date_start', type=str, default='2022-11-18T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2022-11-18T23:00:00', help='End date')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Time delta in minutes')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')

    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    # make sure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)

    data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
    data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)

    channels=['hmi_m', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600']
    vis_lims = {}
    vis_lims['hmi_m'] = 0., 1./2
    vis_lims['aia_0131'] = 0., 0.68/2
    vis_lims['aia_0171'] = 0., 1./2
    vis_lims['aia_0193'] = 0., 0.58/2
    vis_lims['aia_0211'] = 0., 1./2
    vis_lims['aia_1600'] = 0., 0.8/2

    sdo = SDOMLlite(data_dir_sdo, channels=channels, date_start=args.date_start, date_end=args.date_end)
    biosentinel = RadLab(data_dir_radlab, instrument='BPD', normalize=False)
    crater = RadLab(data_dir_radlab, instrument='CRaTER-D1D2', normalize=False)

    date_start = datetime.datetime.fromisoformat(args.date_start)
    date_end = datetime.datetime.fromisoformat(args.date_end)
    num_frames = int(((date_end - date_start).total_seconds() / 60) / args.delta_minutes) + 1

    print('\nDate start      : {}'.format(date_start))
    print('Date end        : {}'.format(date_end))
    print('Delta minutes   : {}'.format(args.delta_minutes))
    print('Number of frames: {:,}'.format(num_frames))

    fig, axs = plt.subplot_mosaic([[c for c in channels],['biosentinel' for _ in range(len(channels))],['crater' for _ in range(len(channels))]], figsize=(12, 8))

    ims = {}
    for c in channels:
        ax = axs[c]
        ax.set_title('SDO {}'.format(c))
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(np.zeros([512,512]), vmin=vis_lims[c][0], vmax=vis_lims[c][1], cmap='gray')
        ims[c] = im

    ax = axs['biosentinel']
    ax.set_title('Biosentinel BPD')
    bio_dates, bio_values = biosentinel.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
    ax.plot(bio_dates, bio_values, color='blue', alpha=0.75)
    # ax.tick_params(rotation=45)
    # ax.set_xticklabels([])
    ims['biosentinel'] = ax.axvline(date_start, color='red', linestyle='-')

    ax = axs['crater']
    ax.set_title('CRaTER-D1D2')
    crater_dates, crater_values = crater.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
    ax.plot(crater_dates, crater_values, color='green', alpha=0.75)
    ax.tick_params(rotation=45)
    ax.set_xticks(axs['biosentinel'].get_xticks())
    ax.set_xlim(axs['biosentinel'].get_xlim())
    ims['crater'] = ax.axvline(date_start, color='red', linestyle='-')
    
    with tqdm(total=num_frames) as pbar:
        def run(frame):
            date = date_start + datetime.timedelta(minutes=frame*args.delta_minutes)
            pbar.set_description('Frame {}'.format(date))
            pbar.update(1)
            sdo_data, _ = sdo[date]
            if sdo_data is None:
                return
            for i, c in enumerate(channels):
                data = sdo_data[i].cpu().numpy()
                ims[c].set_data(data)
            ims['biosentinel'].set_xdata([date, date])
            ims['crater'].set_xdata([date, date])

        plt.tight_layout()
        anim = animation.FuncAnimation(fig, run, interval=300, frames=num_frames)
        
        writervideo = animation.FFMpegWriter(fps=args.fps)
        file_name = os.path.join(args.target_dir, 'event_plot.mp4')
        anim.save(file_name, writer=writervideo)
        plt.close(fig)




    print('\nEnd time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()