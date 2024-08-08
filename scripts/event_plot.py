import argparse
import datetime
import pprint
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates
import sunpy.visualization.colormaps as sunpycm

from tqdm import tqdm

from datasets import SDOMLlite, RadLab, GOESXRS, GOESSGPS, RSTNRadio
from events import EventCatalog

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
    description = 'FDL-X 2024, Radiation Team, event plotting'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory with datasets')
    parser.add_argument('--sdo_dir', type=str, default='sdoml-lite-biosentinel', help='SDOML-lite-biosentinel directory')
    parser.add_argument('--radlab_file', type=str, default='radlab/RadLab-20240625-duck.db', help='RadLab file')
    parser.add_argument('--goes_xrs_file', type=str, default='goes/goes-xrs.csv', help='GOES XRS file')
    parser.add_argument('--goes_sgps_file', type=str, default='goes/goes-sgps.csv', help='GOES SGPS file')
    parser.add_argument('--rstn_radio_file', type=str, default='rstn-radio/rstn-radio.csv', help='RSTN Radio file')    
    parser.add_argument('--date_start', type=str, default='2023-02-25 06:15:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2023-02-28 01:40:00', help='End date')
    parser.add_argument('--sequence_length', type=int, default=20, help='Sequence length')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Time delta in minutes')
    parser.add_argument('--event_id', nargs='+', default=['biosentinel01', 'biosentinel02', 'biosentinel03', 'biosentinel04', 'biosentinel05', 'biosentinel06', 'biosentinel07', 'biosentinel08', 'biosentinel09', 'biosentinel10', 'biosentinel11', 'biosentinel12', 'biosentinel13', 'biosentinel14', 'biosentinel15', 'biosentinel16', 'biosentinel17', 'biosentinel18', 'biosentinel19'], help='Test event IDs')
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
    data_dir_goes_sgps = os.path.join(args.data_dir, args.goes_sgps_file)
    data_dir_goes_xrs = os.path.join(args.data_dir, args.goes_xrs_file)
    # data_dir_rstn_radio = os.path.join(args.data_dir, args.rstn_radio_file)

    minutes_before_start = args.sequence_length * args.delta_minutes
    
    plots_to_produce = []
    
    if args.event_id is not None:
        print('\nEvent ID given, overriding date_start and date_end with event dates')
        
        for event_id in args.event_id:
            if event_id not in EventCatalog:
                raise ValueError('Event ID not found in events: {}'.format(event_id))
            date_start, date_end, max_pfu = EventCatalog[event_id]
            print('Event ID: {}'.format(event_id))
            date_start = datetime.datetime.fromisoformat(date_start) - datetime.timedelta(minutes=minutes_before_start)
            date_end = datetime.datetime.fromisoformat(date_end)
            title_prefix = 'Event: {} (>10 MeV max: {} pfu) '.format(event_id, max_pfu)
            file_name = 'event-{}-{}pfu-{}-{}'.format(event_id, max_pfu, date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
            plots_to_produce.append((date_start, date_end, file_name, title_prefix))
    else:
        date_start = datetime.datetime.fromisoformat(args.date_start) - datetime.timedelta(minutes=minutes_before_start)
        date_end = datetime.datetime.fromisoformat(args.date_end)
        title_prefix = ''
        file_name = 'event-{}-{}'.format(date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
        plots_to_produce.append((date_start, date_end, file_name, title_prefix))


    for date_start, date_end, file_name, title_prefix in plots_to_produce:
        num_frames = int(((date_end - date_start).total_seconds() / 60) / args.delta_minutes) + 1
        duration_minutes = (date_end - date_start).total_seconds() / 60

        print('\nDate start      : {}'.format(date_start))
        print('Date end        : {}'.format(date_end))
        print('Duration        : {:,} minutes'.format(duration_minutes))
        print('Delta minutes   : {}'.format(args.delta_minutes))
        print('Number of frames: {:,}'.format(num_frames))

        channels=['hmi_m', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600']

        sdo = SDOMLlite(data_dir_sdo, channels=channels, date_start=date_start, date_end=date_end)
        biosentinel = RadLab(data_dir_radlab, instrument='BPD', normalize=False)
        # crater = RadLab(data_dir_radlab, instrument='CRaTER-D1D2', normalize=False)
        goessgps10 = GOESSGPS(data_dir_goes_sgps, normalize=False, column='>10MeV')
        goessgps100 = GOESSGPS(data_dir_goes_sgps, normalize=False, column='>100MeV')
        goesxrs = GOESXRS(data_dir_goes_xrs, normalize=False)

        file_name = os.path.join(args.target_dir, file_name)

        fig, axs = plt.subplot_mosaic([['hmi_m', 'aia_0131', 'aia_0171', 'aia_0193', 'aia_0211', 'aia_1600'],
                                    ['biosentinel', 'biosentinel', 'biosentinel', 'biosentinel', 'biosentinel', 'biosentinel'],
                                    ['goessgps10', 'goessgps10', 'goessgps10', 'goessgps10', 'goessgps10', 'goessgps10'],
                                    ['goessgps100', 'goessgps100', 'goessgps100', 'goessgps100', 'goessgps100', 'goessgps100'],
                                    ['goesxrs', 'goesxrs', 'goesxrs', 'goesxrs', 'goesxrs', 'goesxrs']
                                    ], figsize=(20, 10), height_ratios=[2.5, 1, 1, 1, 1])

        hours_locator = matplotlib.dates.HourLocator(interval=1)

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
        # ax.set_title('Biosentinel absorbed dose rate')
        ax.text(0.005, 0.96, 'BioSentinel absorbed dose rate', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        ax.set_ylabel('μGy/min')
        ax.yaxis.set_label_position("right")
        bio_dates, bio_values = biosentinel.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
        if bio_dates is not None:
            ax.plot(bio_dates, bio_values, color='blue', alpha=0.75)
        ax.xaxis.set_minor_locator(hours_locator)
        ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
        ax.grid(color='lightgray', zorder=0, which='major')
        ax.set_xticklabels([])
        ax.set_yscale('log')
        ims['biosentinel'] = ax.axvline(date_start, color='black', linestyle='-', linewidth=1)

        ax = axs['goessgps10']
        # ax.set_title('GOES SGPS proton flux (>10MeV)')
        ax.text(0.005, 0.96, 'GOES SGPS proton flux (>10MeV)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        ax.set_ylabel('part./(cm^2 s sr)')
        ax.yaxis.set_label_position("right")
        goessgps10_dates, goessgps10_values = goessgps10.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
        if goessgps10_dates is not None:
            ax.plot(goessgps10_dates, goessgps10_values, color='darkred', alpha=0.75)
        ax.xaxis.set_minor_locator(hours_locator)
        ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
        ax.grid(color='lightgray', zorder=0, which='major')
        ax.set_xticks(axs['biosentinel'].get_xticks())
        ax.set_xlim(axs['biosentinel'].get_xlim()) 
        ax.set_xticklabels([])
        ax.set_yscale('log')
        ims['goessgps10'] = ax.axvline(date_start, color='black', linestyle='-', linewidth=1)

        ax = axs['goessgps100']
        # ax.set_title('GOES SGPS proton flux (>10MeV)')
        ax.text(0.005, 0.96, 'GOES SGPS proton flux (>100MeV)', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        ax.set_ylabel('part./(cm^2 s sr)')
        ax.yaxis.set_label_position("right")
        goessgps100_dates, goessgps100_values = goessgps100.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
        if goessgps100_dates is not None:
            ax.plot(goessgps100_dates, goessgps100_values, color='green', alpha=0.75)
        ax.xaxis.set_minor_locator(hours_locator)
        ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
        ax.grid(color='lightgray', zorder=0, which='major')
        ax.set_xticks(axs['biosentinel'].get_xticks())
        ax.set_xlim(axs['biosentinel'].get_xlim()) 
        ax.set_xticklabels([])
        ax.set_yscale('log')
        ims['goessgps100'] = ax.axvline(date_start, color='black', linestyle='-', linewidth=1)

        ax = axs['goesxrs']
        # ax.set_title('GOES XRS X-ray flux')
        ax.text(0.005, 0.96, 'GOES XRS X-ray flux', ha='left', va='top', transform=ax.transAxes, fontsize=12)
        ax.set_ylabel('W/m^2')
        ax.yaxis.set_label_position("right")
        goes_dates, goes_values = goesxrs.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
        if goes_dates is not None:
            ax.plot(goes_dates, goes_values, color='purple', alpha=0.75)
        ax.xaxis.set_minor_locator(hours_locator)
        ax.grid(color='#f0f0f0', zorder=0, which='minor', axis='x')
        ax.grid(color='lightgray', zorder=0, which='major')
        ax.set_xticks(axs['biosentinel'].get_xticks())
        ax.set_xlim(axs['biosentinel'].get_xlim())
        ax.set_yscale('log')
        myFmt = matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M')
        ax.xaxis.set_major_formatter(myFmt)
        # ax.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))
        ims['goesxrs'] = ax.axvline(date_start, color='black', linestyle='-', linewidth=1)


        title = plt.suptitle(title_prefix + str(date_start))
        
        with tqdm(total=num_frames) as pbar:
            def run(frame):
                date = date_start + datetime.timedelta(minutes=frame*args.delta_minutes)
                pbar.set_description('Frame {}'.format(date))
                pbar.update(1)

                title.set_text(title_prefix + str(date))
                ims['biosentinel'].set_xdata([date, date])
                # ims['crater'].set_xdata([date, date])
                # ims['rstnradio'].set_xdata([date, date])
                ims['goessgps10'].set_xdata([date, date])
                ims['goessgps100'].set_xdata([date, date])
                ims['goesxrs'].set_xdata([date, date])

                sdo_data, _ = sdo[date]
                for i, c in enumerate(channels):
                    if sdo_data is None:
                        # ims[c].set_data(np.zeros([512,512]))
                        pass
                    else:
                        ims[c].set_data(unnormalize(sdo_data[i].cpu().numpy(), c))

            # plt.tight_layout()
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            anim = animation.FuncAnimation(fig, run, interval=300, frames=num_frames)
            
            file_name_mp4 = file_name + '.mp4'
            writer_mp4 = animation.FFMpegWriter(fps=args.fps)
            anim.save(file_name_mp4, writer=writer_mp4)

            # pbar.reset()
            # file_name_gif = file_name + '.gif'
            # writer_gif = animation.ImageMagickWriter(fps=args.fps)
            # anim.save(file_name_gif, writer=writer_gif)
            # plt.close(fig)

        print('\nResults saved in:')
        print('{}'.format(file_name_mp4))
        # print('{}'.format(file_name_gif))

    print('\nEnd time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()


# Can use gifgen to convert mp4 to gif
# https://github.com/lukechilds/gifgen