"""
This code is currently adapted from flwr. It will however be rewritten before use
"""
from argparse import ArgumentParser

import torch

import flwr as fl

import mnist

import json

DATA_ROOT = "./data/mnist"

if __name__ == "__main__":
    # Training settings
    parser = ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--server_address",
        type=str,
        default="[::]:8080",
        help=f"gRPC server address (default: '[::]:8080')",
    )
    parser.add_argument(
        "--cid",
        type=int,
        metavar="N",
        default=0,
        help="ID of current client (default: 0)",
    )
    parser.add_argument(
        "--nb_clients",
        type=int,
        default=2,
        metavar="N",
        help="Total number of clients being launched (default: 2)",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )

    args = parser.parse_args()

    # Load MNIST data
    train_loader, test_loader = mnist.load_data(
        data_root=DATA_ROOT,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        cid=args.cid,
        nb_clients=args.nb_clients,
    )

    # pylint: disable=no-member
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pylint: enable=no-member

    # Instantiate client
    client = mnist.PytorchMNISTClient(
        cid=args.cid,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        device=device,
    )

    # Start client
    fl.client.start_client(args.server_address, client)
