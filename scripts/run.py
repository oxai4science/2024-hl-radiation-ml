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

from datasets import SDOMLlite, BioSentinel, Sequences
from models import SDOSequence

matplotlib.use('Agg')

def seed(seed=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    print('Setting seed to {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test(model, test_date_start, test_date_end, args):
    test_sdo = SDOMLlite(args.sdo_dir, date_start=test_date_start, date_end=test_date_end)
    test_biosentinel = BioSentinel(args.biosentinel_file, date_start=test_date_start, date_end=test_date_end)
    test_sequences = Sequences([test_sdo], delta_minutes=args.delta_minutes, sequence_length=args.sequence_length)
    test_loader = DataLoader(test_sequences, batch_size=args.batch_size, shuffle=False)
    
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
                    prediction_value = float(test_biosentinel.unnormalize_data(output[i]))
                    ground_truth_value, _ = test_biosentinel[prediction_date]
                    test_predictions.append((prediction_date, prediction_value))
                    if ground_truth_value is None:
                        ground_truth_value = float('nan')
                    else:
                        ground_truth_value = float(test_biosentinel.unnormalize_data(ground_truth_value))
                    test_ground_truths.append((prediction_date, ground_truth_value))
                pbar.update(1)
    return test_predictions, test_ground_truths


def save_test_file(test_predictions, test_ground_truths, test_file):
    print('\nSaving test results to {}'.format(test_file))
    with open(test_file, 'w') as f:
        f.write('date,prediction,ground_truth\n')
        for i in range(len(test_predictions)):
            f.write('{},{},{}\n'.format(test_predictions[i][0], test_predictions[i][1], test_ground_truths[i][1]))


def save_test_plot(test_predictions, test_ground_truths, test_plot_file):
    print('Saving test plot to {}'.format(test_plot_file))
    plt.figure(figsize=(24, 6))
    plt.plot(*zip(*test_predictions), label='Prediction', alpha=0.75)
    plt.plot(*zip(*test_ground_truths), label='Ground truth', alpha=0.75)
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
    parser.add_argument('--sdo_dir', type=str, default='/disk2-ssd-8tb/data/sdoml-lite-biosentinel', help='SDOML-lite-biosentinel directory')
    parser.add_argument('--biosentinel_file', type=str, default='/disk2-ssd-8tb/data/biosentinel/BPD_readings.csv', help='BioSentinel file')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length')
    parser.add_argument('--delta_minutes', type=int, default=15, help='Delta minutes')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=0, help='Random number generator seed')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--valid_proportion', type=float, default=0.1, help='Validation frequency in iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Mode', required=True)
    parser.add_argument('--date_start', type=str, default='2024-04-01T00:00:00', help='Start date')
    parser.add_argument('--date_end', type=str, default='2024-05-01T00:00:00', help='End date')
    parser.add_argument('--test_date_start', type=str, default='2024-05-01T00:00:00', help='Start date')
    parser.add_argument('--test_date_end', type=str, default='2024-05-14T19:30:00', help='End date')
    parser.add_argument('--model_file', type=str, help='Model file')

    args = parser.parse_args()

    print(description)    
    
    start_time = datetime.datetime.now()
    print('Start time: {}'.format(start_time))
    print('Arguments:\n{}'.format(' '.join(sys.argv[1:])))
    print('Config:')
    pprint.pprint(vars(args), depth=2, width=50)

    seed(args.seed)
    device = torch.device(args.device)

    # make sure the target directory exists
    os.makedirs(args.target_dir, exist_ok=True)

    if args.mode == 'train':
        print('\n*** Training mode\n')

        # For training and validation
        sdo = SDOMLlite(args.sdo_dir)
        biosentinel = BioSentinel(args.biosentinel_file, date_start=args.date_start, date_end=args.date_end)
        sequences = Sequences([sdo, biosentinel], delta_minutes=args.delta_minutes, sequence_length=args.sequence_length)

        # Testing with data seen during training
        # Use the last 14 days in the training data
        test_seen_date_start = (datetime.datetime.fromisoformat(args.date_end) - datetime.timedelta(days=14)).isoformat()
        test_seen_date_end = args.date_end

        # Split sequences into train and validation
        valid_size = int(args.valid_proportion * len(sequences))
        train_size = len(sequences) - valid_size
        sequences_train, sequences_valid = random_split(sequences, [train_size, valid_size])

        print('\nTrain size: {:,}'.format(len(sequences_train)))
        print('Valid size: {:,}'.format(len(sequences_valid)))

        train_loader = DataLoader(sequences_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = DataLoader(sequences_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = SDOSequence(channels=len(sdo.channels), embedding_dim=512, sequence_length=args.sequence_length)
        model = model.to(device)

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('\nNumber of parameters: {:,}\n'.format(num_params))

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        iteration = 0
        train_losses = []
        valid_losses = []
        for epoch in range(args.epochs):
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
            test_predictions, test_ground_truths = test(model, args.test_date_start, args.test_date_end, args)
            test_file = '{}/epoch_{:03d}_test.csv'.format(args.target_dir, epoch+1)
            save_test_file(test_predictions, test_ground_truths, test_file)
            test_plot_file = '{}/epoch_{:03d}_test.pdf'.format(args.target_dir, epoch+1)
            save_test_plot(test_predictions, test_ground_truths, test_plot_file)

            # Test with seen data
            print('*** Testing with seen data')
            test_seen_predictions, test_seen_ground_truths = test(model, test_seen_date_start, test_seen_date_end, args)
            test_seen_file = '{}/epoch_{:03d}_test_seen.csv'.format(args.target_dir, epoch+1)
            save_test_file(test_seen_predictions, test_seen_ground_truths, test_seen_file)
            test_seen_plot_file = '{}/epoch_{:03d}_test_seen.pdf'.format(args.target_dir, epoch+1)
            save_test_plot(test_seen_predictions, test_seen_ground_truths, test_seen_plot_file)

            shutil.copyfile(model_file, '{}/latest_model.pth'.format(args.target_dir))
            shutil.copyfile(plot_file, '{}/latest_loss.pdf'.format(args.target_dir))
            shutil.copyfile(test_file, '{}/latest_test.csv'.format(args.target_dir))
            shutil.copyfile(test_plot_file, '{}/latest_test.pdf'.format(args.target_dir))
            shutil.copyfile(test_seen_file, '{}/latest_test_seen.csv'.format(args.target_dir))
            shutil.copyfile(test_seen_plot_file, '{}/latest_test_seen.pdf'.format(args.target_dir))
        

    print('\nEnd time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()