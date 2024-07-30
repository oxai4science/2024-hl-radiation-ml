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
from tqdm import tqdm
import shutil
import traceback

from datasets import SDOMLlite, RadLab, Sequences
from models import SDOSequence

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


def test(model, test_date_start, test_date_end, data_dir_sdo, data_dir_radlab, args):
    test_sdo = SDOMLlite(data_dir_sdo, date_start=test_date_start, date_end=test_date_end)
    test_biosentinel = RadLab(data_dir_radlab, instrument='BPD', date_start=test_date_start, date_end=test_date_end)
    test_sequences = Sequences([test_sdo], delta_minutes=args.delta_minutes, sequence_length=args.sequence_length)
    test_loader = DataLoader(test_sequences, batch_size=args.batch_size, shuffle=False)
    
    test_dates = []
    test_predictions = []
    test_ground_truths = []
    device = next(model.parameters()).device
    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Testing') as pbar:
            for sdo, dates in test_loader:
                sdo = sdo.to(device)
                input = sdo
                output = model(input)
                
                for i in range(len(output)):
                    prediction_date = dates[i][-1]
                    test_dates.append(prediction_date)
                    prediction_value = output[i]
                    test_predictions.append(prediction_value)
                    ground_truth_value, _ = test_biosentinel[prediction_date]
                    if ground_truth_value is None:
                        ground_truth_value = torch.tensor(float('nan'))
                    else:
                        ground_truth_value = ground_truth_value
                    test_ground_truths.append(ground_truth_value)
                pbar.update(1)
    test_predictions = torch.stack(test_predictions)
    test_ground_truths = torch.stack(test_ground_truths)
    return test_dates, test_predictions, test_ground_truths


def save_test_file(test_dates, test_predictions, test_ground_truths, test_file):
    if torch.is_tensor(test_predictions):
        test_predictions = test_predictions.cpu().numpy()
    if torch.is_tensor(test_ground_truths):
        test_ground_truths = test_ground_truths.cpu().numpy()
    print('\nSaving test results to {}'.format(test_file))
    with open(test_file, 'w') as f:
        f.write('date,prediction,ground_truth\n')
        for i in range(len(test_predictions)):
            f.write('{},{},{}\n'.format(test_dates[i], test_predictions[i], test_ground_truths[i]))


def save_test_plot(test_dates, test_predictions, test_ground_truths, test_plot_file):
    if torch.is_tensor(test_predictions):
        test_predictions = test_predictions.cpu().numpy()
    if torch.is_tensor(test_ground_truths):
        test_ground_truths = test_ground_truths.cpu().numpy()
    print('Saving test plot to {}'.format(test_plot_file))
    plt.figure(figsize=(24, 6))
    plt.plot(test_dates, test_predictions, label='Prediction', alpha=0.75)
    plt.plot(test_dates, test_ground_truths, label='Ground truth', alpha=0.75)
    # plt.xlabel('Date')
    plt.ylabel('Absorbed dose rate')
    # Limit number of xticks
    plt.xticks(np.arange(0, len(test_predictions), step=len(test_predictions)//40))
    # Rotate xticks
    plt.xticks(rotation=45)
    # Shift xticks so that the end of the text is at the tick
    plt.xticks(ha='right')
    plt.legend()
    plt.grid(color='#f0f0f0', zorder=0)
    plt.tight_layout()
    plt.savefig(test_plot_file)    


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
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Delta minutes')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=0, help='Random number generator seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--valid_proportion', type=float, default=0.05, help='Validation frequency in iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Mode', required=True)
    parser.add_argument('--date_start', type=str, default='2024-01-01T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-05-01T00:00:00', help='End date')
    parser.add_argument('--test_date_start', type=str, default='2023-05-01T00:00:00', help='Start date')
    parser.add_argument('--test_date_end', type=str, default='2024-05-14T19:30:00', help='End date')
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


        data_dir_sdo = os.path.join(args.data_dir, args.sdo_dir)
        data_dir_radlab = os.path.join(args.data_dir, args.radlab_file)

        sys.stdout.flush()
        if args.mode == 'train':
            print('\n*** Training mode\n')

            # For training and validation
            dataset_sdo = SDOMLlite(data_dir_sdo)
            dataset_biosentinel = RadLab(data_dir_radlab, instrument='BPD', date_start=args.date_start, date_end=args.date_end)
            dataset_sequences = Sequences([dataset_sdo, dataset_biosentinel], delta_minutes=args.delta_minutes, sequence_length=args.sequence_length)

            # Testing with data seen during training
            # Use the last 14 days in the training data
            test_seen_date_start = (datetime.datetime.fromisoformat(args.date_end) - datetime.timedelta(days=14)).isoformat()
            test_seen_date_end = args.date_end

            # Split sequences into train and validation
            valid_size = int(args.valid_proportion * len(dataset_sequences))
            train_size = len(dataset_sequences) - valid_size
            dataset_sequences_train, dataset_sequences_valid = random_split(dataset_sequences, [train_size, valid_size])

            print('\nTrain size: {:,}'.format(len(dataset_sequences_train)))
            print('Valid size: {:,}'.format(len(dataset_sequences_valid)))

            train_loader = DataLoader(dataset_sequences_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            valid_loader = DataLoader(dataset_sequences_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

            model = SDOSequence(channels=len(dataset_sdo.channels), embedding_dim=1024, sequence_length=args.sequence_length)
            model = model.to(device)

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
                    for i, (sdo, biosentinel, _) in enumerate(train_loader):
                        # print('move data to device', flush=True)
                        sdo = sdo.to(device)
                        biosentinel = biosentinel.to(device)

                        input = sdo
                        target = biosentinel[:, -1].unsqueeze(1)

                        # print('run model', flush=True)
                        optimizer.zero_grad()
                        output = model(input)
                        loss = torch.nn.functional.mse_loss(output, target)
                        loss.backward()
                        optimizer.step()

                        train_losses.append((iteration, float(loss)))
                        # print('Epoch: {:,}/{:,} | Iter: {:,}/{:,} | Loss: {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), float(loss)))
                        pbar.set_description('Epoch: {:,}/{:,} | Iter: {:,}/{:,} | Loss: {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), float(loss)))
                        pbar.update(1)

                        # print('getting data', flush=True)
                        # if iteration % args.valid_every == 0:
                        iteration += 1

                print('*** Validation')
                with torch.no_grad():
                    valid_loss = 0.
                    valid_seqs = 0
                    model.eval()
                    with tqdm(total=len(valid_loader), desc='Validation') as pbar:
                        for sdo, biosentinel, _ in valid_loader:
                            sdo = sdo.to(device)
                            biosentinel = biosentinel.to(device)

                            input = sdo
                            target = biosentinel[:, -1].unsqueeze(1)

                            output = model(input)
                            loss = torch.nn.functional.mse_loss(output, target)
                            valid_loss += float(loss)
                            valid_seqs += 1
                            pbar.update(1)

                    valid_loss /= valid_seqs
                    print('\nEpoch: {:,}/{:,} | Iter: {:,}/{:,} | Loss: {:.4f} | Valid loss: {:.4f}'.format(epoch+1, args.epochs, i+1, len(train_loader), float(loss), valid_loss))
                    valid_losses.append((iteration, valid_loss))
                

                # Save model
                model_file = '{}/epoch_{:03d}_model.pth'.format(args.target_dir, epoch+1)
                print('Saving model to {}'.format(model_file))
                checkpoint = {
                    'model': 'SDOSequence',
                    'epoch': epoch,
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'valid_losses': valid_losses
                }
                torch.save(checkpoint, model_file)

                # Plot losses
                plot_file = '{}/epoch_{:03d}_loss.pdf'.format(args.target_dir, epoch+1)
                save_loss_plot(train_losses, valid_losses, plot_file)

                # Test with unseen data
                print('*** Testing with unseen data')
                test_dates, test_predictions_normalized, test_ground_truths_normalized = test(model, args.test_date_start, args.test_date_end, data_dir_sdo, data_dir_radlab, args)

                test_file_normalized = '{}/epoch_{:03d}_test_unseen_normalized.csv'.format(args.target_dir, epoch+1)
                save_test_file(test_dates, test_predictions_normalized, test_ground_truths_normalized, test_file_normalized)
                test_plot_file_normalized = '{}/epoch_{:03d}_test_unseen_normalized.pdf'.format(args.target_dir, epoch+1)
                save_test_plot(test_dates, test_predictions_normalized, test_ground_truths_normalized, test_plot_file_normalized)

                test_predictions_unnormalized = dataset_biosentinel.unnormalize_data(test_predictions_normalized)
                test_ground_truths_unnormalized = dataset_biosentinel.unnormalize_data(test_ground_truths_normalized)

                test_file_unnormalized = '{}/epoch_{:03d}_test_unseen_unnormalized.csv'.format(args.target_dir, epoch+1)
                save_test_file(test_dates, test_predictions_unnormalized, test_ground_truths_unnormalized, test_file_unnormalized)
                test_plot_file_unnormalized = '{}/epoch_{:03d}_test_unseen_unnormalized.pdf'.format(args.target_dir, epoch+1)
                save_test_plot(test_dates, test_predictions_unnormalized, test_ground_truths_unnormalized, test_plot_file_unnormalized)


                # Test with seen data
                print('*** Testing with seen data')
                test_dates, test_seen_predictions_normalized, test_seen_ground_truths_normalized = test(model, test_seen_date_start, test_seen_date_end, data_dir_sdo, data_dir_radlab, args)

                test_seen_file_normalized = '{}/epoch_{:03d}_test_seen_normalized.csv'.format(args.target_dir, epoch+1)
                save_test_file(test_dates, test_seen_predictions_normalized, test_seen_ground_truths_normalized, test_seen_file_normalized)
                test_seen_plot_file_normalized = '{}/epoch_{:03d}_test_seen_normalized.pdf'.format(args.target_dir, epoch+1)
                save_test_plot(test_dates, test_seen_predictions_normalized, test_seen_ground_truths_normalized, test_seen_plot_file_normalized)

                test_seen_predictions_unnormalized = dataset_biosentinel.unnormalize_data(test_seen_predictions_normalized)
                test_seen_ground_truths_unnormalized = dataset_biosentinel.unnormalize_data(test_seen_ground_truths_normalized)

                test_seen_file_unnormalized = '{}/epoch_{:03d}_test_seen_unnormalized.csv'.format(args.target_dir, epoch+1)
                save_test_file(test_dates, test_seen_predictions_unnormalized, test_seen_ground_truths_unnormalized, test_seen_file_unnormalized)
                test_seen_plot_file_unnormalized = '{}/epoch_{:03d}_test_seen_unnormalized.pdf'.format(args.target_dir, epoch+1)
                save_test_plot(test_dates, test_seen_predictions_unnormalized, test_seen_ground_truths_unnormalized, test_seen_plot_file_unnormalized)

                shutil.copyfile(model_file, '{}/latest_model.pth'.format(args.target_dir))
                shutil.copyfile(plot_file, '{}/latest_loss.pdf'.format(args.target_dir))

                shutil.copyfile(test_file_normalized, '{}/latest_test_unseen_normalized.csv'.format(args.target_dir))
                shutil.copyfile(test_plot_file_normalized, '{}/latest_test_unseen_normalized.pdf'.format(args.target_dir))
                shutil.copyfile(test_file_unnormalized, '{}/latest_test_unseen_unnormalized.csv'.format(args.target_dir))
                shutil.copyfile(test_plot_file_unnormalized, '{}/latest_test_unseen_unnormalized.pdf'.format(args.target_dir))

                shutil.copyfile(test_seen_file_normalized, '{}/latest_test_seen_normalized.csv'.format(args.target_dir))
                shutil.copyfile(test_seen_plot_file_normalized, '{}/latest_test_seen_normalized.pdf'.format(args.target_dir))
                shutil.copyfile(test_seen_file_unnormalized, '{}/latest_test_seen_unnormalized.csv'.format(args.target_dir))
                shutil.copyfile(test_seen_plot_file_unnormalized, '{}/latest_test_seen_unnormalized.pdf'.format(args.target_dir))
            

        print('\nEnd time: {}'.format(datetime.datetime.now()))
        print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()