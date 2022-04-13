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


def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
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

    # m = torch.nn.Linear(784, 32)
    # for i, _ in partitions[0][0]:
    #    i = torch.flatten(i.to("cpu"), 1)
    #    output = m(i)
    #    print(output.shape)
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
