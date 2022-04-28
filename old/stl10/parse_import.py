import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Command Line Arguments')
    parser.add_argument('--num_epochs', default=10, type=int,
                        help='number of epochs')
    parser.add_argument('--num_batches', default=5, type=int,
                        help='number of batches after which progress is displayed')
    parser.add_argument('--checkpoint_freq', default=20, type=int,
                        help='frequency of checkpointing')
    parser.add_argument('--from_checkpoint', default=None, type=str,
                        help='path of checkpoint to start from')
    parser.add_argument('--train', default=True, type=bool,
                        help='train or test (false for test)')
    parser.add_argument('--gpuid', default=0, type=int,
                        help='gpu id')
    return parser

def parse_commandline_args():
    return create_parser().parse_args()

# `python train_denoiser_for_mcmc.py --num_epochs=500 --num_batches=5 --checkpoint_freq=20 --denoise=gaussian`