import argparse
import datetime
import pprint
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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
    parser.add_argument('--date_end', type=str, default='2022-11-18T05:00:00', help='End date')
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
    biosentinel = RadLab(data_dir_radlab, instrument='BPD', date_start=args.date_start, date_end=args.date_end)

    date_start = datetime.datetime.fromisoformat(args.date_start)
    date_end = datetime.datetime.fromisoformat(args.date_end)
    num_frames = int(((date_end - date_start).total_seconds() / 60) / args.delta_minutes)

    print('\nDate start      : {}'.format(date_start))
    print('Date end        : {}'.format(date_end))
    print('Delta minutes   : {}'.format(args.delta_minutes))
    print('Number of frames: {:,}'.format(num_frames))

    fig, axs = plt.subplots(6, 2, figsize=(12, 12))

    ims = {}
    for i, channel in enumerate(channels):
        axs[i,0].set_title('SDO/AIA {}'.format(channel.upper()))
        axs[i,0].set_xticks([])
        axs[i,0].set_yticks([])        
        im = axs[i,0].imshow(np.zeros([512,512]), vmin=0, vmax=1, cmap='gray')
        ims[(i,0)] = im

    def run(frame):
        date = date_start + datetime.timedelta(minutes=frame*args.delta_minutes)
        sdo_data, _ = sdo[date]
        if sdo_data is None:
            return
        print('Frame {:,} - {}'.format(frame, date))
        for i, channel in enumerate(channels):
            data = sdo_data[i].cpu().numpy()
            print(data.min(), data.max())
            ims[(i,0)].set_data(data)

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