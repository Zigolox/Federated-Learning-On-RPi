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


def get_eval_fn(model: torch.nn.Module, args):
    # Load MNIST data
    (x_train, y_train), _ = mnist.load_data(
        data_root=DATA_ROOT,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        cid=args.cid,
        nb_clients=args.nb_clients,
    )
    max_ind = len(x_train)
    x_val, y_val = x_train[max_ind-5000:max_ind], y_train[max_ind-5000:max_ind]

    def evaluate(
        weights: fl.common.Weights,
    ) -> typing.Optional[typing.Tuple[float, float]]:
        model.set_weights(weights)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        return loss, accuracy

    return evaluate

def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""

    model = mnist.MNISTNet()

    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), weights=None)}
    )
    model.load_state_dict()
    model.eval()

    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit, eval_fn=get_eval_fn(model))
    # Exposes the server by default on port 8080
    fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds})


def start_client(data, epochs: int, send_end):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    send_end.send(1)


def simulation(num_rounds: int, num_clients: int, fraction_fit: float, epochs: int):

    # This will hold all the processes which we are going to create
    processes = []
    torch.set_num_threads(1)
    # set_start_method("spawn", force=True)
    # Start the server
    server_process = Process(
        target=start_server, args=(num_rounds, num_clients, fraction_fit)
    )
    server_process.start()
    processes.append(server_process)
    partitions = partitioner(DATA_ROOT, partitions=num_clients, iid=True, batch_size=64)
    pipe_list = []

    for partition in partitions:
        recv_end, send_end = Pipe(False)
        client_process = Process(
            target=start_client, args=(partition, epochs, send_end)
        )
        pipe_list.append(recv_end)
        client_process.start()
        processes.append(client_process)
    # server_process.start()
    # Block until all processes are finished
    for p in processes:
        p.join()
    results = [i.recv() for i in pipe_list]
    print(results)


simulation(5, 100, 0.5, 3)
