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
from events import events

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
    parser.add_argument('--date_start', type=str, default='2023-02-25 06:15:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2023-02-28 01:40:00', help='End date')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Time delta in minutes')
    parser.add_argument('--event_id', type=str, help='Event ID')
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

    if args.event_id is not None:
        print('\nEvent ID given, overriding date_start and date_end with event dates')
        if args.event_id not in events:
            raise ValueError('Event ID not found in events: {}'.format(args.event_id))
        args.date_start, args.date_end, max_pfu = events[args.event_id]
        print('Event ID: {}'.format(args.event_id))
        print('Date start: {}'.format(args.date_start))
        print('Date end: {}'.format(args.date_end))

    date_start = datetime.datetime.fromisoformat(args.date_start)
    date_end = datetime.datetime.fromisoformat(args.date_end)
    num_frames = int(((date_end - date_start).total_seconds() / 60) / args.delta_minutes) + 1
    duration_minutes = (date_end - date_start).total_seconds() / 60

    print('\nDate start      : {}'.format(date_start))
    print('Date end        : {}'.format(date_end))
    print('Duration        : {:,} minutes'.format(duration_minutes))
    print('Delta minutes   : {}'.format(args.delta_minutes))
    print('Number of frames: {:,}'.format(num_frames))

    channels=['hmi_m', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600']

    sdo = SDOMLlite(data_dir_sdo, channels=channels, date_start=args.date_start, date_end=args.date_end)
    biosentinel = RadLab(data_dir_radlab, instrument='BPD', normalize=False)
    crater = RadLab(data_dir_radlab, instrument='CRaTER-D1D2', normalize=False)

    file_name = 'event-plot-{}-{}.mp4'.format(date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))

    if args.event_id is not None:
        title_prefix = '{} (>10 MeV max: {} pfu) / '.format(args.event_id, max_pfu)
        file_name = 'event-{}-{}pfu-{}-{}.mp4'.format(args.event_id, max_pfu, date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
    else:
        title_prefix = ''
        file_name = 'event-{}-{}.mp4'.format(date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))

    file_name = os.path.join(args.target_dir, file_name)

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
            vmin[c], vmax[c] = np.percentile(unnormalize(sdo_sample[i], c), (0.2, 98))

    ims = {}
    for c in channels:
        cmap = cms[c]
        # cmap = 'viridis'    
        ax = axs[c]
        ax.set_title('SDOML-lite / {}'.format(c))
        ax.set_xticks([])
        ax.set_yticks([])
        im = ax.imshow(np.zeros([512,512]), vmin=vmin[c], vmax=vmax[c], cmap=cmap)
        ims[c] = im

    ax = axs['biosentinel']
    ax.set_title('Biosentinel BPD')
    ax.set_ylabel('Absorbed dose rate [mGy/min]')
    ax.yaxis.set_label_position("right")
    bio_dates, bio_values = biosentinel.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
    ax.plot(bio_dates, bio_values, color='blue', alpha=0.75)
    # ax.tick_params(rotation=45)
    ax.set_xticklabels([])
    ax.grid(color='#f0f0f0', zorder=0)
    ax.set_yscale('log')
    # ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    ims['biosentinel'] = ax.axvline(date_start, color='red', linestyle='-')

    ax = axs['crater']
    ax.set_title('CRaTER-D1D2')
    ax.set_ylabel('Absorbed dose rate [mGy/h]')
    ax.yaxis.set_label_position("right")
    crater_dates, crater_values = crater.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
    ax.plot(crater_dates, crater_values, color='green', alpha=0.75)
    # ax.tick_params(rotation=45)
    ax.set_xticks(axs['biosentinel'].get_xticks())
    ax.set_xlim(axs['biosentinel'].get_xlim())
    ax.grid(color='#f0f0f0', zorder=0)
    ax.set_yscale('log')
    myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    # ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
    ims['crater'] = ax.axvline(date_start, color='red', linestyle='-')

    title = plt.suptitle(title_prefix + str(date_start))
    
    with tqdm(total=num_frames) as pbar:
        def run(frame):
            date = date_start + datetime.timedelta(minutes=frame*args.delta_minutes)
            pbar.set_description('Frame {}'.format(date))
            pbar.update(1)

            title.set_text(title_prefix + str(date))
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
        anim.save(file_name, writer=writervideo)
        plt.close(fig)

    print('\nFile saved: {}'.format(file_name))

    print('\nEnd time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()