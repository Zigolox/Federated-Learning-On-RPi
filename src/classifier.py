"""
This code is currently adapted from flwr. It will however be rewritten before use
"""


"""
    PyTorch MNIST image classification.

"""


# mypy: ignore-errors
# pylint: disable=W0223

import timeit
from collections import OrderedDict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import datasets, transforms

import flwr as fl


# TODO write function that times the program from start to finish to test effect of parameter changes


def dataset_partitioner(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    cid: int,
    nb_clients: int,
) -> torch.utils.data.DataLoader:
    """Function which partitions datasets"""

    # Set the seed so we are sure to generate the same global batches
    # indices across all clients
    np.random.seed(123)  # TODO check if performance is affected by change of parameter

    # Get the data corresponding to this client
    dataset_size = len(dataset)
    nb_samples_clients = (dataset_size) // nb_clients
    #print((nb_samples_clients / 2) // 2)
    dataset_indices = list(range(dataset_size))
    np.random.shuffle(dataset_indices)

    # Get starting and ending indices w.r.t cid
    start_index = cid * nb_samples_clients
    end_index = start_index + nb_samples_clients
    sampler = SubsetRandomSampler(dataset_indices[start_index:end_index])
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,  # TODO test what shuffle=False does
    )
    return data_loader


def load_data(
    data_root: str,
    train_batch_size: int,
    test_batch_size: int,
    cid: int,
    nb_clients: int,
) -> Tuple[DataLoader, DataLoader]:
    """Loads both training and test datasets for MNIST. Returns tuple containing DataLoaders for training and test set."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    train_dataset = datasets.MNIST(
        data_root, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(data_root, train=False, transform=transform)

    # Create partitioned datasets based on the total number of users and cid
    train_loader = dataset_partitioner(
        dataset=train_dataset,
        batch_size=train_batch_size,
        cid=cid,
        nb_clients=nb_clients,
    )

    test_loader = dataset_partitioner(
        dataset=test_dataset,
        batch_size=test_batch_size,
        cid=cid,
        nb_clients=nb_clients,
    )

    return (train_loader, test_loader)


class MNISTNet(nn.Module):
    """Simple CNN adapted from Pytorch's 'Basic MNIST Example'."""

    def __init__(self) -> None:
        super(MNISTNet, self).__init__()
        self.dropout1 = nn.Dropout2d(0.06)
        self.dropout2 = nn.Dropout2d(0.15)
        self.dropout3 = nn.Dropout2d(0.15)
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)
        self.fc3 = nn.Linear(10, 6)

    # pylint: disable=arguments-differ,invalid-name
    def forward(self, x: Tensor) -> Tensor:
        """Computes forward pass. Takes tensor x as a mini-batch containing images from MNIST dataset. Returns probability density of the output being from a specific class given the input."""
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epochs: int,
    device: torch.device = torch.device("cpu"),
) -> int:
    """Train routine based on Pytorch's 'Basic MNIST Example', returns number of total samples used during training"""
    model.train()
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    print(f"Training {epochs} epoch(s) w/ {len(train_loader)} mini-batches each")
    for epoch in range(epochs):  # loop over the dataset multiple times
        print()
        loss_epoch: float = 0.0
        num_examples_train: int = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # Grab mini-batch and transfer to device
            data, target = data.to(device), target.to(device)
            num_examples_train += len(data)

            # Zero gradients
            optimizer.zero_grad()

            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            if batch_idx % 10 == 8:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}\t\t\t\t".format(
                        epoch,
                        num_examples_train,
                        len(train_loader) * train_loader.batch_size,
                        100.0
                        * num_examples_train
                        / len(train_loader)
                        / train_loader.batch_size,
                        loss.item(),
                    ),
                    end="\r",
                    flush=True,
                )
        scheduler.step()
    return num_examples_train


def test(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, float, float]:
    """Test routine 'Basic MNIST Example'. Returns tuple containing the total number of test samples, the test loss, and the accuracy evaluated on the test set."""
    model.eval()
    test_loss: float = 0
    correct: int = 0
    num_test_samples: int = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            num_test_samples += len(data)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= num_test_samples

    return (num_test_samples, test_loss, correct / num_test_samples)


class PytorchMNISTClient(fl.client.Client):
    """Flower client implementing MNIST handwritten classification using PyTorch."""

    def __init__(
        self,
        cid: int,
        train_loader: datasets,
        test_loader: datasets,
        epochs: int,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = MNISTNet().to(device)
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs

    def get_weights(self) -> fl.common.Weights:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_weights(self, weights: fl.common.Weights) -> None:
        """Sets model weights from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {
                k: torch.tensor(v)
                for k, v in zip(self.model.state_dict().keys(), weights)
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> fl.common.ParametersRes:
        """Encapsulates the weights into Flower Parameters."""
        weights: fl.common.Weights = self.get_weights()
        parameters = fl.common.weights_to_parameters(weights)
        return fl.common.ParametersRes(parameters=parameters)

    def fit(self, ins: fl.common.FitIns) -> fl.common.FitRes:
        """Trains the model on local dataset. Returns set of variables containing the new set of weights and information the client."""

        # Set the seed so we are sure to generate the same global batches
        # indices across all clients
        np.random.seed(123)

        weights: fl.common.Weights = fl.common.parameters_to_weights(ins.parameters)
        fit_begin = timeit.default_timer()

        # Set model parameters/weights
        self.set_weights(weights)

        # Train model
        num_examples_train: int = train(
            self.model, self.train_loader, epochs=self.epochs, device=self.device
        )

        # Return the refined weights and the number of examples used for training
        weights_prime: fl.common.Weights = self.get_weights()
        params_prime = fl.common.weights_to_parameters(weights_prime)
        fit_duration = timeit.default_timer() - fit_begin
        return fl.common.FitRes(
            parameters=params_prime,
            num_examples=num_examples_train,
            num_examples_ceil=num_examples_train,
            fit_duration=fit_duration,
        )

    def evaluate(self, ins: fl.common.EvaluateIns) -> fl.common.EvaluateRes:
        """Evaluates and returns information on the clients testing results."""
        weights = fl.common.parameters_to_weights(ins.parameters)

        # Use provided weights to update the local model
        self.set_weights(weights)

        (
            num_examples_test,
            test_loss,
            accuracy,
        ) = test(self.model, self.test_loader, device=self.device)
        print(
            f"Client {self.cid} - Evaluate on {num_examples_test} samples: Average loss: {test_loss:.4f}, Accuracy: {100*accuracy:.2f}%\n"
        )

        # Return the number of evaluation examples and the evaluation result (loss)
        return fl.common.EvaluateRes(
            loss=float(test_loss),
            num_examples=num_examples_test,
            accuracy=float(accuracy),
        )
