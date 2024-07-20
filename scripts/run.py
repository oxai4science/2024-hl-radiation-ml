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
    parser.add_argument('--valid_every', type=int, default=250, help='Validation frequency in iterations')
    parser.add_argument('--device', type=str, default='cpu', help='Device')

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

    sdo_train = SDOMLlite(args.sdo_dir, date_end='2024-05-01')
    sdo_valid = SDOMLlite(args.sdo_dir, date_start='2024-05-01')

    biosentinel_train = BioSentinel(args.biosentinel_file, date_end='2024-05-01', normalize=True)
    biosentinel_valid = BioSentinel(args.biosentinel_file, date_start='2024-05-01', normalize=True)

    sequences_train = Sequences([sdo_train, biosentinel_train], delta_minutes=args.delta_minutes, sequence_length=args.sequence_length, shuffle=True)
    sequences_valid = Sequences([sdo_valid, biosentinel_valid], delta_minutes=args.delta_minutes, sequence_length=args.sequence_length)

    print('\nTrain size: {:,}'.format(sequences_train.length))
    print('Valid size: {:,}'.format(sequences_valid.length))

    train_loader = DataLoader(sequences_train, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    valid_loader = DataLoader(sequences_valid, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SDOSequence(channels=len(sdo_train.channels), embedding_dim=512, sequence_length=args.sequence_length)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nNumber of parameters: {:,}'.format(num_params))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    iteration = 0
    train_losses = []
    valid_losses = []
    for epoch in range(args.epochs):
        for sdo, biosentinel, _ in train_loader:
            sdo = sdo.to(device)
            biosentinel = biosentinel.to(device)

            input = sdo
            target = biosentinel[:, -1].unsqueeze(1)

            optimizer.zero_grad()
            output = model(input)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

            train_losses.append((iteration, float(loss)))
            print('Epoch: {:,} | Iter: {:,} | Loss: {:.4f}'.format(epoch+1, iteration, float(loss)))

            if iteration % args.valid_every == 0:
                print('Validation', end=' ')
                with torch.no_grad():
                    valid_loss = 0.
                    valid_seqs = 0
                    for sdo, biosentinel, _ in valid_loader:
                        sdo = sdo.to(device)
                        biosentinel = biosentinel.to(device)

                        input = sdo
                        target = biosentinel[:, -1].unsqueeze(1)

                        output = model(input)
                        loss = torch.nn.functional.mse_loss(output, target)
                        valid_loss += float(loss)
                        valid_seqs += 1

                    valid_loss /= valid_seqs
                    print('Epoch: {:,} | Iter: {:,} | Valid loss: {:.4f}'.format(epoch+1, iteration, valid_loss))
                    valid_losses.append((iteration, valid_loss))
                

                # Save model
                model_file = '{}/model_iter_{}.pth'.format(args.target_dir, iteration)
                print('Saving model to {}'.format(model_file))
                torch.save(model.state_dict(), model_file)

                # Plot losses
                plot_file = '{}/loss_iter_{}.pdf'.format(args.target_dir, iteration)
                print('Saving plot to {}'.format(plot_file))
                plt.figure()
                plt.plot(*zip(*train_losses), label='Training')
                plt.plot(*zip(*valid_losses), label='Validation')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.yscale('log')
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_file)

            iteration += 1


    print('\nEnd time: {}'.format(datetime.datetime.now()))
    print('Duration: {}'.format(datetime.datetime.now() - start_time))

if __name__ == "__main__":
    main()