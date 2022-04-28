import flwr as fl

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch MNIST Server")
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=3,
        help=f"Number of rounds",
    )
    args = parser.parse_args()
    fl.server.start_server(config={"num_rounds": args.num_rounds})
