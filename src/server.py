import flwr as fl
from torchvision import datasets, transforms
import torch
from client import DATA_ROOT
import mnist
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from collections import OrderedDict
import typing
from flwr.server.strategy import FedAvg

DATA_ROOT = "./data/mnist"


def get_eval_fn(model: torch.nn.Module, device: torch.device):
    # Load MNIST data

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    test_data = datasets.MNIST(DATA_ROOT, train=False, transform=transform)
    test_loader = DataLoader(test_data, batch_size=1000)

    def evaluate(
        weights: fl.common.Weights,
    ) -> typing.Optional[typing.Tuple[float, dict]]:

        # Set weight
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights)}
        )
        model.load_state_dict(state_dict, strict=True)

        _, loss, accuracy = mnist.test(model, test_loader, device)
        return loss, {"accuracy": accuracy}

    return evaluate


if __name__ == "__main__":
    parser = ArgumentParser(description="PyTorch MNIST Server")
    parser.add_argument(
        "--num_rounds",
        type=int,
        default=10,
        help=f"Number of rounds",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = mnist.MNISTNet()
    args = parser.parse_args()
    strategy = FedAvg(
        min_available_clients=2,
        fraction_fit=1,
        eval_fn=get_eval_fn(model, device),
    )

    fl.server.start_server(strategy=strategy, config={"num_rounds": args.num_rounds})
