#!/usr/bin/env python3
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets, transforms
import numpy as np
import torch
import flwr as fl
from torch.multiprocessing import Process, Pipe, set_start_method
import mnist
import asyncio
from flwr.server.strategy import FedAvg
import typing
import flwr as fl
from collections import OrderedDict

DATA_ROOT = "./data/mnist"


def partitioner(data_root: str, partitions: int, iid: bool, batch_size: int):
    def splitter(dataset: Dataset, partitions: int, iid: bool, batch_size: int):

        # Get the data corresponding to this client
        size = len(dataset) // partitions
        if iid:
            dataset_indices = np.arange(len(dataset))
            np.random.shuffle(dataset_indices)
        else:
            dataset_indices = []
            for label in sorted(torch.unique(dataset.targets).tolist()):
                dataset_indices += (
                    torch.nonzero(dataset.targets == label).flatten().tolist()
                )
        # Get starting and ending indices w.r.t CLIENT_ID
        data_loaders = []
        for i in range(partitions):
            data_sampler = SubsetRandomSampler(
                dataset_indices[size * i : size * (i + 1)]
            )
            data_loaders.append(
                DataLoader(
                    dataset, batch_size=batch_size, shuffle=False, sampler=data_sampler
                )
            )
        return data_loaders

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_root, train=False, transform=transform)

    np.random.seed(123)

    return list(
        zip(
            splitter(train_dataset, partitions, iid, batch_size),
            splitter(test_dataset, partitions, iid, 1000),
        )
    )


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


def start_server(
    num_rounds: int,
    num_clients: int,
    fraction_fit: float,
    device: torch.device,
    centralized_eval: bool,
    send_server,
):
    """Start the server with a slightly adjusted FedAvg strategy."""

    model = mnist.MNISTNet()

    if centralized_eval:
        eval_fn = get_eval_fn(model, device)
    else:
        eval_fn = None
    strategy = FedAvg(
        min_available_clients=num_clients,
        fraction_fit=fraction_fit,
        eval_fn=eval_fn,
        fraction_eval=1,
    )
    # Exposes the server by default on port 8080
    result = fl.server.start_server(
        strategy=strategy, config={"num_rounds": num_rounds}
    )
    send_server.send(result)


def start_client(data, epochs: int, device: torch.device):
    # Instantiate client
    client = mnist.PytorchMNISTClient(
        cid=0,
        train_loader=data[0],
        test_loader=data[1],
        epochs=epochs,
        device=device,
    )
    # Start client
    fl.client.start_client("0.0.0.0:8080", client)


def simulation(
    num_rounds: int,
    num_clients: int,
    fraction_fit: float,
    epochs: int,
    iid: bool = True,
    partitions: int = None,
    centralized_eval: bool = True,
):

    if partitions == None:
        partitions = num_clients
    # This will hold all the processes which we are going to create
    processes = []
    torch.set_num_threads(1)
    # Choose gpu if availeble
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Start the server
    recv_server, send_server = Pipe(False)
    server_process = Process(
        target=start_server,
        args=(
            num_rounds,
            num_clients,
            fraction_fit,
            device,
            centralized_eval,
            send_server,
        ),
    )
    server_process.start()
    processes.append(server_process)
    part = partitioner(DATA_ROOT, partitions=partitions, iid=iid, batch_size=64)

    for i in range(num_clients):
        client_process = Process(
            target=start_client,
            args=(part[i * partitions // num_clients], epochs, device),
        )
        client_process.start()
        processes.append(client_process)
    # Block until all processes are finished
    for p in processes:
        p.join()
    results = recv_server.recv()
    return results


if __name__ == "__main__":
    simulation(5, 10, 0.5, 3, True, False)
