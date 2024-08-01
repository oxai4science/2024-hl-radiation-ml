import argparse
import datetime
import pprint
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import sunpy.visualization.colormaps as sunpycm

from tqdm import tqdm

from datasets import SDOMLlite, RadLab

matplotlib.use('Agg')

sqrt_aia_cutoff = {}
sqrt_aia_cutoff['aia_0131'] = np.sqrt(2652.1470)
sqrt_aia_cutoff['aia_0171'] = np.sqrt(22816.1035)
sqrt_aia_cutoff['aia_0193'] = np.sqrt(23919.7168)
sqrt_aia_cutoff['aia_0211'] = np.sqrt(13458.3203)
sqrt_aia_cutoff['aia_1600'] = np.sqrt(3399.5896)

cms = {}
cms['hmi_m'] = sunpycm.cmlist.get('hmimag')
cms['aia_0131'] = sunpycm.cmlist.get('sdoaia131')
cms['aia_0171'] = sunpycm.cmlist.get('sdoaia171')
cms['aia_0193'] = sunpycm.cmlist.get('sdoaia193')
cms['aia_0211'] = sunpycm.cmlist.get('sdoaia211')
cms['aia_1600'] = sunpycm.cmlist.get('sdoaia1600')


def unnormalize(data, channel):
    if channel == 'hmi_m':
        mask = data > 0.05
        data = 2 * (data - 0.5)
        data = data * 1500 
        data = data * mask
    else:
        c = sqrt_aia_cutoff[channel]
        data = data * c
        data = data ** 2.
    return data


def main():
    description = 'FDL-X 2024, Radiation Team, data statistics'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory with datasets')
    parser.add_argument('--sdo_dir', type=str, default='sdoml-lite-biosentinel', help='SDOML-lite-biosentinel directory')
    parser.add_argument('--radlab_file', type=str, default='radlab/RadLab-20240625-duck.db', help='RadLab file')
    parser.add_argument('--date_start', type=str, default='2022-12-14T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2022-12-16T00:00:00', help='End date')
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

    fig, axs = plt.subplot_mosaic([['hmi_m', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600'],
                                   ['biosentinel', 'biosentinel', 'biosentinel', 'biosentinel', 'biosentinel', 'biosentinel'],
                                   ['crater', 'crater', 'crater', 'crater', 'crater', 'crater']], figsize=(20, 10), height_ratios=[2, 1, 1])

    vmin = {}
    vmax = {}
    sdo_sample, _ = sdo[0]
    for i, c in enumerate(channels):
        if c == 'hmi_m':
            vmin[c], vmax[c] = -1500, 1500
        else:
            vmin[c], vmax[c] = np.percentile(unnormalize(sdo_sample[i], c), (0.5, 98))


    ims = {}
    for c in channels:
        cmap = cms[c]
        # cmap = 'viridis'    
        ax = axs[c]
        ax.set_title('SDO {}'.format(c))
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(np.zeros([512,512]), vmin=vmin[c], vmax=vmax[c], cmap=cmap)
        ims[c] = im

    ax = axs['biosentinel']
    ax.set_title('Biosentinel BPD')
    bio_dates, bio_values = biosentinel.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
    ax.plot(bio_dates, bio_values, color='blue', alpha=0.75)
    # ax.tick_params(rotation=45)
    ax.set_xticklabels([])
    ax.grid(color='#f0f0f0', zorder=0)
    ims['biosentinel'] = ax.axvline(date_start, color='red', linestyle='-')

    ax = axs['crater']
    ax.set_title('CRaTER-D1D2')
    crater_dates, crater_values = crater.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
    ax.plot(crater_dates, crater_values, color='green', alpha=0.75)
    ax.tick_params(rotation=45)
    ax.set_xticks(axs['biosentinel'].get_xticks())
    ax.set_xlim(axs['biosentinel'].get_xlim())
    ax.grid(color='#f0f0f0', zorder=0)
    myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    ims['crater'] = ax.axvline(date_start, color='red', linestyle='-')
    
    with tqdm(total=num_frames) as pbar:
        def run(frame):
            date = date_start + datetime.timedelta(minutes=frame*args.delta_minutes)
            pbar.set_description('Frame {}'.format(date))
            pbar.update(1)

            ims['biosentinel'].set_xdata([date, date])
            ims['crater'].set_xdata([date, date])

            sdo_data, _ = sdo[date]
            for i, c in enumerate(channels):
                if sdo_data is None:
                    # ims[c].set_data(np.zeros([512,512]))
                    pass
                else:
                    ims[c].set_data(unnormalize(sdo_data[i].cpu().numpy(), c))

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