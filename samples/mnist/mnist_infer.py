# Copyright 2024 IBM Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sample PyTorch application with MNIST model trained by
`mnist_training.py`.
"""
import argparse
from mnist import MNIST
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_nnpa


class ImageDataset(Dataset):
    def __init__(self, images, labels, normalize=False):
        self.images = images
        self.labels = labels
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.uint8)
        image = torch.reshape(image, [1, 28, 28])
        if self.normalize is not None:
            normalized = image.float() / 255.0
            image = (normalized - 0.1307) / 0.3081

        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def infer(model, device, infer_loader):
    model.eval()

    loss = 0
    correct = 0
    start_time = time.time()

    with torch.inference_mode():
        for data, target in infer_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    end_time = time.time()
    loss /= len(infer_loader.dataset)

    print('Infer time: {:.4f}'.format(end_time - start_time))
    print('Infer set: Average loss: {:.4f}, accuracy: {}/{} ({:.0f}%)'.format(
        loss, correct, len(infer_loader.dataset),
        100. * correct / len(infer_loader.dataset)))


def main():
    # Inference settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--no-nnpa', action='store_true', default=False,
                        help='disables NNPA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    args = parser.parse_args()

    use_nnpa = not args.no_nnpa and torch.nnpa.is_available()

    torch.manual_seed(args.seed)

    if use_nnpa:
        device = torch.device('nnpa')
    else:
        device = torch.device('cpu')

    mnist_data = MNIST('./data')
    images, labels = mnist_data.load_testing()

    dataset = ImageDataset(images, labels, normalize=True)

    infer_kwargs = {'batch_size': args.batch_size}

    infer_loader = DataLoader(dataset, **infer_kwargs)

    model = Net().to(device)
    model.load_state_dict(torch.load('mnist_cnn.pt', weights_only=True))

    infer(model, device, infer_loader)


if __name__ == '__main__':
    main()
