import argparse
import sys
import pprint
import time
import datetime
import torch
import random
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
import shutil
import traceback

from datasets import SDOMLlite, RadLab, GOESXRS, Sequences
from models import RadRecurrent
from events import EventCatalog

matplotlib.use('Agg')

class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()


def seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_test_file(prediction_dates, goesxrs_predictions, biosentinel_predictions, goesxrs_ground_truth_dates, goesxrs_ground_truth_values, biosentinel_ground_truth_dates, biosentinel_ground_truth_values, file_name):
    print('Saving test results: {}'.format(file_name))

    goesxrs_prediction_mean = np.mean(goesxrs_predictions, axis=0)
    goesxrs_prediction_std = np.std(goesxrs_predictions, axis=0)

    biosentinel_prediction_mean = np.mean(biosentinel_predictions, axis=0)
    biosentinel_prediction_std = np.std(biosentinel_predictions, axis=0)

    with open(file_name, 'w') as f:
        f.write('date,goesxrs_prediction_mean,goesxrs_prediction_std,biosentinel_prediction_mean,biosentinel_prediction_std,ground_truth_goesxrs,ground_truth_biosentinel\n')
        for i in range(len(prediction_dates)):
            date = prediction_dates[i]
            goesxrs_prediction_mean_value = goesxrs_prediction_mean[i]
            goesxrs_prediction_std_value = goesxrs_prediction_std[i]
            biosentinel_prediction_mean_value = biosentinel_prediction_mean[i]
            biosentinel_prediction_std_value = biosentinel_prediction_std[i]

            if date in goesxrs_ground_truth_dates:
                goesxrs_ground_truth_value = goesxrs_ground_truth_values[goesxrs_ground_truth_dates.index(date)]
            else:
                goesxrs_ground_truth_value = float('nan')

            if date in biosentinel_ground_truth_dates:
                biosentinel_ground_truth_value = biosentinel_ground_truth_values[biosentinel_ground_truth_dates.index(date)]
            else:
                biosentinel_ground_truth_value = float('nan')

            f.write('{},{},{},{},{},{},{}\n'.format(date, goesxrs_prediction_mean_value, goesxrs_prediction_std_value, biosentinel_prediction_mean_value, biosentinel_prediction_std_value, goesxrs_ground_truth_value, biosentinel_ground_truth_value))
            

def save_test_plot(prediction_dates, goesxrs_predictions, biosentinel_predictions, goesxrs_ground_truth_dates, goesxrs_ground_truth_values, biosentinel_ground_truth_dates, biosentinel_ground_truth_values, file_name, title=None):
    print('Saving test plot: {}'.format(file_name))
    fig, axs = plt.subplot_mosaic([['biosentinel'],['goesxrs']], figsize=(20, 10), height_ratios=[1,1])

    num_samples = goesxrs_predictions.shape[0]
    prediction_alpha = 0.1

    ax = axs['biosentinel']
    ax.set_title('Biosentinel BPD')
    ax.set_ylabel('Absorbed dose rate\n[mGy/min]')
    ax.yaxis.set_label_position("right")
    for i in range(num_samples):
        label = 'Prediction' if i == 0 else None
        ax.plot(prediction_dates, biosentinel_predictions[i], label=label, color='gray', alpha=prediction_alpha)
    ax.plot(biosentinel_ground_truth_dates, biosentinel_ground_truth_values, color='blue', label='Ground truth', alpha=0.75)
    ax.grid(color='#f0f0f0', zorder=0)
    ax.set_xticklabels([])
    ax.grid(color='#f0f0f0', zorder=0)
    ax.set_yscale('log')    
    ax.legend()

    ax = axs['goesxrs']
    ax.set_title('GOES XRS')
    ax.set_ylabel('X-ray flux\n[W/m^2]')
    ax.yaxis.set_label_position("right")
    for i in range(num_samples):
        label = 'Prediction' if i == 0 else None
        ax.plot(prediction_dates, goesxrs_predictions[i], label=label, color='gray', alpha=prediction_alpha)
    ax.plot(goesxrs_ground_truth_dates, goesxrs_ground_truth_values, color='purple', label='Ground truth', alpha=0.75)
    ax.grid(color='#f0f0f0', zorder=0)
    ax.set_yscale('log')
    myFmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    if title is not None:
        plt.suptitle(title)
    plt.savefig(file_name)


def run_model(model, context, prediction_window):
    batch_size = context.shape[0]
    model.init(batch_size)
    context_output = model(context)
    x = context_output[:, -1, :].unsqueeze(1)
    prediction = []
    for _ in range(prediction_window):
        prediction.append(x)
        x = model(x)
    prediction = torch.cat(prediction, dim=1)
    return prediction


def run_test(model, date_start, date_end, file_prefix, title, args):
        # data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
        data_dir_goes_xrs = os.path.join(args.data_dir, args.goes_xrs_file)
        data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)

        date_context_start = date_start - datetime.timedelta(minutes=args.context_window * args.delta_minutes)

        context_steps = args.context_window
        prediction_steps = int((date_end - date_start).total_seconds() / (args.delta_minutes * 60))

        dataset_goes_xrs = GOESXRS(data_dir_goes_xrs, date_start=date_context_start, date_end=date_end)
        dataset_biosentinel = RadLab(data_dir_radlab, instrument='BPD', date_start=date_context_start, date_end=date_end)
        dataset_sequences = Sequences([dataset_goes_xrs, dataset_biosentinel], delta_minutes=args.delta_minutes, sequence_length=args.context_window)

        context_sequence = dataset_sequences[0]

        context_goesxrs = context_sequence[0][:context_steps].unsqueeze(1)
        context_biosentinel = context_sequence[1][:context_steps].unsqueeze(1)
        context_goesxrs = context_goesxrs.to(args.device)
        context_biosentinel = context_biosentinel.to(args.device)

        context = torch.cat([context_goesxrs, context_biosentinel], dim=1)
        context_batch = context.unsqueeze(0).repeat(args.num_samples, 1, 1)
        prediction_batch = run_model(model, context_batch, prediction_steps).detach()

        prediction_date_start = datetime.datetime.fromisoformat(context_sequence[2][-1])
        prediction_dates = [prediction_date_start + datetime.timedelta(minutes=i*args.delta_minutes) for i in range(prediction_steps)]

        goesxrs_predictions = prediction_batch[:, :, 0]
        biosentinel_predictions = prediction_batch[:, :, 1]

        goesxrs_predictions = dataset_goes_xrs.unnormalize_data(goesxrs_predictions)
        biosentinel_predictions = dataset_biosentinel.unnormalize_data(biosentinel_predictions)

        goesxrs_predictions = goesxrs_predictions.cpu().numpy()
        biosentinel_predictions = biosentinel_predictions.cpu().numpy()

        goesxrs_ground_truth_dates, goesxrs_ground_truth_values = dataset_goes_xrs.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
        goesxrs_ground_truth_values = dataset_goes_xrs.unnormalize_data(goesxrs_ground_truth_values)
        biosentinel_ground_truth_dates, biosentinel_ground_truth_values = dataset_biosentinel.get_series(date_start, date_end, delta_minutes=args.delta_minutes)
        biosentinel_ground_truth_values = dataset_biosentinel.unnormalize_data(biosentinel_ground_truth_values)

        file_name = os.path.join(args.target_dir, file_prefix)
        test_file = file_name + '.csv'
        save_test_file(prediction_dates, goesxrs_predictions, biosentinel_predictions, goesxrs_ground_truth_dates, goesxrs_ground_truth_values, biosentinel_ground_truth_dates, biosentinel_ground_truth_values, test_file)

        test_plot_file = file_name + '.pdf'
        save_test_plot(prediction_dates, goesxrs_predictions, biosentinel_predictions, goesxrs_ground_truth_dates, goesxrs_ground_truth_values, biosentinel_ground_truth_dates, biosentinel_ground_truth_values, test_plot_file, title=title)

            

    
def save_loss_plot(train_losses, valid_losses, plot_file):
    print('Saving plot to {}'.format(plot_file))
    plt.figure(figsize=(12, 6))
    plt.plot(*zip(*train_losses), label='Training')
    plt.plot(*zip(*valid_losses), label='Validation')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(color='#f0f0f0', zorder=0)
    plt.tight_layout()
    plt.savefig(plot_file)    


def main():
    description = 'FDL-X 2024, Radiation Team, preliminary machine learning experiments'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--target_dir', type=str, required=True, help='Directory to store results')
    parser.add_argument('--data_dir', type=str, required=True, help='Root directory with datasets')
    parser.add_argument('--sdo_dir', type=str, default='sdoml-lite-biosentinel', help='SDOML-lite-biosentinel directory')
    parser.add_argument('--radlab_file', type=str, default='radlab/RadLab-20240625-duck.db', help='RadLab file')
    parser.add_argument('--goes_xrs_file', type=str, default='goes-xrs/goes-xrs.csv', help='GOES XRS file')
    parser.add_argument('--context_window', type=int, default=15, help='Context window')
    parser.add_argument('--prediction_window', type=int, default=15, help='Prediction window')
    parser.add_argument('--num_samples', type=int, default=20, help='Number of samples for MC dropout inference')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Delta minutes')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=0, help='Random number generator seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-05, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--valid_proportion', type=float, default=0.05, help='Validation frequency in iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--lstm_depth', type=int, default=2, help='LSTM depth')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Mode', required=True)
    parser.add_argument('--date_start', type=str, default='2022-11-16T11:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-05-14T09:15:00', help='End date')
    parser.add_argument('--test_event_id', nargs='+', default=['biosentinel01', 'biosentinel07', 'biosentinel19'], help='Test event IDs')
    parser.add_argument('--test_seen_event_id', nargs='+', default=['biosentinel04', 'biosentinel15', 'biosentinel18'], help='Test event IDs seen during training')
    parser.add_argument('--model_file', type=str, help='Model file')

    args = parser.parse_args()

    # make sure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)

    log_file = os.path.join(args.target_dir, 'log.txt')

    with Tee(log_file):
        print(description)    
        print('Log file: {}'.format(log_file))
        start_time = datetime.datetime.now()
        print('Start time: {}'.format(start_time))
        print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
        print('Config:')
        pprint.pprint(vars(args), depth=2, width=50)

        seed(args.seed)
        device = torch.device(args.device)


        # data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
        data_dir_goes_xrs = os.path.join(args.data_dir, args.goes_xrs_file)
        data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)

        sys.stdout.flush()
        if args.mode == 'train':
            print('\n*** Training mode\n')

            date_exclusions = []
            if args.test_event_id is not None:
                for event_id in args.test_event_id:
                    if event_id not in EventCatalog:
                        raise ValueError('Event ID not found in events: {}'.format(event_id))
                    date_start, date_end, _ = EventCatalog[event_id]
                    date_start = datetime.datetime.fromisoformat(date_start)
                    date_end = datetime.datetime.fromisoformat(date_end)
                    date_exclusions.append((date_start, date_end))

            # For training and validation
            # dataset_sdo = SDOMLlite(data_dir_sdo, date_exclusions=date_exclusions)
            dataset_goes_xrs = GOESXRS(data_dir_goes_xrs, date_start=args.date_start, date_end=args.date_end, date_exclusions=date_exclusions)
            dataset_biosentinel = RadLab(data_dir_radlab, instrument='BPD', date_start=args.date_start, date_end=args.date_end, date_exclusions=date_exclusions)
            # dataset_sequences = Sequences([dataset_sdo, dataset_biosentinel], delta_minutes=args.delta_minutes, sequence_length=args.sequence_length)
            training_sequence_length = args.context_window + args.prediction_window
            dataset_sequences = Sequences([dataset_goes_xrs, dataset_biosentinel], delta_minutes=args.delta_minutes, sequence_length=training_sequence_length)


            # Split sequences into train and validation
            valid_size = int(args.valid_proportion * len(dataset_sequences))
            train_size = len(dataset_sequences) - valid_size
            dataset_sequences_train, dataset_sequences_valid = random_split(dataset_sequences, [train_size, valid_size])

            print('\nTrain size: {:,}'.format(len(dataset_sequences_train)))
            print('Valid size: {:,}'.format(len(dataset_sequences_valid)))

            train_loader = DataLoader(dataset_sequences_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(dataset_sequences_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            model_data_dim = 2
            model_lstm_dim = 1024
            model_lstm_depth = args.lstm_depth
            model_dropout = 0.2
            model = RadRecurrent(data_dim=model_data_dim, lstm_dim=model_lstm_dim, lstm_depth=model_lstm_depth, dropout=model_dropout)
            model = model.to(device)
            model.train()

            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print('\nNumber of parameters: {:,}\n'.format(num_params))

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            iteration = 0
            train_losses = []
            valid_losses = []
            for epoch in range(args.epochs):
                print('\n*** Epoch: {:,}/{:,}'.format(epoch+1, args.epochs))
                print('*** Training')
                model.train()
                with tqdm(total=len(train_loader)) as pbar:
                    for i, (goesxrs, biosentinel, _) in enumerate(train_loader):
                        batch_size = goesxrs.shape[0]

                        goesxrs = goesxrs.to(device)
                        biosentinel = biosentinel.to(device)
                        goesxrs = goesxrs.unsqueeze(-1)
                        biosentinel = biosentinel.unsqueeze(-1)
                        data = torch.cat([goesxrs, biosentinel], dim=2)
                        
                        input = data[:, :-1]
                        target = data[:, 1:]
                       
                        model.init(batch_size)
                        optimizer.zero_grad()
                        output = model(input)
                        loss = torch.nn.functional.mse_loss(output, target)
                        loss.backward()
                        optimizer.step()

                        train_losses.append((iteration, float(loss)))

                        pbar.set_description('Epoch: {:,}/{:,} | Iter: {:,}/{:,} | Loss: {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), float(loss)))
                        pbar.update(1)

                        iteration += 1

                print('*** Validation')
                with torch.no_grad():
                    valid_loss = 0.
                    valid_seqs = 0
                    # model.eval()
                    with tqdm(total=len(valid_loader), desc='Validation') as pbar:
                        for goesxrs, biosentinel, _ in valid_loader:
                            batch_size = goesxrs.shape[0]

                            goesxrs = goesxrs.to(device)
                            biosentinel = biosentinel.to(device)
                            goesxrs = goesxrs.unsqueeze(-1)
                            biosentinel = biosentinel.unsqueeze(-1)
                            data = torch.cat([goesxrs, biosentinel], dim=2)

                            input = data[:, :-1]
                            target = data[:, 1:]

                            model.init(batch_size)
                            output = model(input)
                            loss = torch.nn.functional.mse_loss(output, target)
                            valid_loss += float(loss)
                            valid_seqs += 1
                            pbar.update(1)

                    valid_loss /= valid_seqs
                    print('\nEpoch: {:,}/{:,} | Iter: {:,}/{:,} | Loss: {:.4f} | Valid loss: {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), float(loss), valid_loss))
                    valid_losses.append((iteration, valid_loss))
                

                # Save model
                model_file = '{}/epoch-{:03d}-model.pth'.format(args.target_dir, epoch+1)
                print('Saving model to {}'.format(model_file))
                checkpoint = {
                    'model': 'SDOSequence',
                    'epoch': epoch,
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'valid_losses': valid_losses,
                    'model_data_dim': model_data_dim,
                    'model_lstm_dim': model_lstm_dim,
                    'model_lstm_depth': model_lstm_depth,
                    'model_dropout': model_dropout
                }
                torch.save(checkpoint, model_file)

                # Plot losses
                plot_file = '{}/epoch-{:03d}-loss.pdf'.format(args.target_dir, epoch+1)
                save_loss_plot(train_losses, valid_losses, plot_file)

                if args.test_event_id is not None:
                    for event_id in args.test_event_id:
                        if event_id not in EventCatalog:
                            raise ValueError('Event ID not found in events: {}'.format(event_id))
                        date_start, date_end, max_pfu = EventCatalog[event_id]
                        print('Event ID: {}'.format(event_id))
                        date_start = datetime.datetime.fromisoformat(date_start)
                        date_end = datetime.datetime.fromisoformat(date_end)
                        file_prefix = 'epoch-{:03d}-test-event-{}-{}pfu-{}-{}'.format(epoch+1, event_id, max_pfu, date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
                        title = 'Event: {} (>10 MeV max: {} pfu)'.format(event_id, max_pfu)
                        run_test(model, date_start, date_end, file_prefix, title, args)

                if args.test_seen_event_id is not None:
                    for event_id in args.test_seen_event_id:
                        if event_id not in EventCatalog:
                            raise ValueError('Event ID not found in events: {}'.format(event_id))
                        date_start, date_end, max_pfu = EventCatalog[event_id]
                        print('Event ID: {}'.format(event_id))
                        date_start = datetime.datetime.fromisoformat(date_start)
                        date_end = datetime.datetime.fromisoformat(date_end)
                        file_prefix = 'epoch-{:03d}-test-seen-event-{}-{}pfu-{}-{}'.format(epoch+1, event_id, max_pfu, date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
                        title = 'Event: {} (>10 MeV max: {} pfu)'.format(event_id, max_pfu)
                        run_test(model, date_start, date_end, file_prefix, title, args)

        if args.mode == 'test':
            print('\n*** Testing mode\n')

            checkpoint = torch.load(args.model_file)
            model_data_dim = checkpoint['model_data_dim']
            model_lstm_dim = checkpoint['model_lstm_dim']
            model_lstm_depth = checkpoint['model_lstm_depth']
            model_dropout = checkpoint['model_dropout']
            
            model = RadRecurrent(data_dim=model_data_dim, lstm_dim=model_lstm_dim, lstm_depth=model_lstm_depth, dropout=model_dropout)
            # model = SDOSequence(channels=6, embedding_dim=1024, sequence_length=args.sequence_length)
            model = model.to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.train() # set to train mode to use MC dropout

            tests_to_run = []
            if args.test_event_id is not None:
                print('\nEvent IDs given, will ignore date_start and date_end arguments and use event dates')

                for event_id in args.test_event_id:
                    if event_id not in EventCatalog:
                        raise ValueError('Event ID not found in events: {}'.format(event_id))
                    date_start, date_end, max_pfu = EventCatalog[event_id]
                    print('Event ID: {}'.format(event_id))

                    date_start = datetime.datetime.fromisoformat(date_start)
                    date_end = datetime.datetime.fromisoformat(date_end)
                    file_prefix = 'test-event-{}-{}pfu-{}-{}'.format(event_id, max_pfu, date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
                    title = 'Event: {} (>10 MeV max: {} pfu)'.format(event_id, max_pfu)
                    tests_to_run.append((date_start, date_end, file_prefix, title))

            else:
                print('\nEvent IDs not given, will use date_start and date_end arguments')

                date_start = datetime.datetime.fromisoformat(args.date_start)
                date_end = datetime.datetime.fromisoformat(args.date_end)
                file_prefix = 'test-event-{}-{}'.format(date_start.strftime('%Y%m%d%H%M'), date_end.strftime('%Y%m%d%H%M'))
                title = None
                tests_to_run.append((date_start, date_end, file_prefix, title))


            for date_start, date_end, file_prefix, title in tests_to_run:
                run_test(model, date_start, date_end, file_prefix, title, args)


        print('\nEnd time: {}'.format(datetime.datetime.now()))
        print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()