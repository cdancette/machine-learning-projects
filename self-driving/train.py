import copy 
import os

import torch
from torch.autograd import Variable
from torch import optim
from torch import nn

from model import CustomModel
from data import CustomDataset, get_dataloader
import argparse

def train(model, criterion, train_loader, test_loader, max_epochs=50, 
          learning_rate=0.001):
    
    dataloaders = {
        "train":train_loader, "val": test_loader
    }

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    best_acc = 0
    for epoch in range(max_epochs):
        print('Epoch {}/{}'.format(epoch, max_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['val', 'train']:
            if phase == 'train':
                # scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data
                labels = labels.view(labels.size(0))

                inputs, labels = Variable(inputs), Variable(labels)
                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0] * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, "models2/model-%s.weights" % epoch)

    print('Training complete')
    print('Best val Acc: {:4f}'.format(best_acc))

if __name__=='__main__': 
    num_classes = 3
    model = CustomModel()

    train_path = "dataset2"
    test_path = "test_set"
    train_loader = get_dataloader(train_path, batch_size=8)
    test_loader = get_dataloader(test_path, batch_size=30)

    loss = nn.CrossEntropyLoss()
    train(model, loss, train_loader, test_loader)
