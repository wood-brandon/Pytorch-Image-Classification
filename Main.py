import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.optim as optim
from Models.AlexNet import AlexNet
from Models.VGG import VGG
from Models.ResNet_50 import ResNet_50
from Models.LeNet import LeNet


def train(model, train_loader, optimizer, epoches, device, train_loss, train_acc):
    model.train()
    criterion = nn.BCELoss() # BCELoss used for binary classification evaluation

    for index, (images, labels) in enumerate(train_loader):

        labels = labels.type(torch.FloatTensor)  # bceloss requires float
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad() # reset gradients
        preds = model(images)  # Pass data into network

        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()  # update weights
        # accumulate values for plotting learning
        train_loss.append(loss.item())
        train_acc.append(get_accuracy(torch.round(preds), labels))

        # This code was taken from the MNIST example code given and modified slightly 
        if index % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoches, index * len(images), len(train_loader.dataset),
                         100. * index / len(train_loader), loss.item()))


def test(model, test_loader, device,test_acc,test_loss):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    i = 0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for images, labels in test_loader:
            i += 1
            labels = labels.type(torch.FloatTensor)
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += (criterion(output, labels).item())
            total_accuracy += get_accuracy(torch.round(output), labels)
    total_loss /= i
    total_accuracy /= i

    # accumulate values for plotting learning
    test_acc.append(total_accuracy)
    test_loss.append(total_loss)

    # print statement was taken from the MNIST example code given
    print('\nTest set: Average loss: {:.4f}, Accuracy: ({:.0f}%)\n'.format(
        total_loss, 100. * total_accuracy.__float__()))


def get_all_preds(model, train_loader, device):
    all_preds = torch.tensor([])
    all_preds: Tensor = all_preds.to(device)
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        output = torch.round(model(images))
        all_preds = torch.cat((all_preds, output), dim=0)
    return all_preds


def get_accuracy(preds, target):
    return torch.mean((sum(target == preds)).float() / len(target))


def main():
    if __name__ == '__main__':

        # 1: VGG 2: AlexNet 3: ResNet-50 4: LeNet
        model_num = 2
        epoches = 6
        save = True

        # check if cuda is available
        cuda_available = torch.cuda.is_available()
        # Use cuda if it is available
        device = torch.device("cuda" if cuda_available else "cpu")
        torch.manual_seed(1)
        model = 0

        if model_num == 1:
            model = VGG().to(device)
            train_batch = 20
            test_batch = 100
            lr = 0.001
            gamma = 0.7
        elif model_num == 2:
            model = AlexNet().to(device)
            train_batch = 64
            test_batch = 1000
            lr = 0.001
            gamma = 0.7
        elif model_num == 3:
            model = ResNet_50().to(device)
            train_batch = 2
            test_batch = 10
            lr = 0.0001
            gamma = 0.7
        else:
            model = LeNet().to(device)
            train_batch = 64
            test_batch = 1000
            gamma = 0.7
            lr = 0.001

        optimizer = optim.Adam(model.parameters(), lr=lr) # create optimizer using Adam algorithm

        train_set = datasets.CelebA(
            root='.data/CelebA'
            , split='train'
            , target_type='attr'
            , target_transform=None
            , download=True
            , transform=transforms.Compose([
                transforms.Pad(padding=(0, 0, 46, 6), fill=0, padding_mode='edge'),
                transforms.ToTensor(),
            ])
        )

        test_set = datasets.CelebA(
            root='.data/CelebA'
            , split='test'
            , target_type='attr'
            , target_transform=None
            , transform=transforms.Compose([
                transforms.Pad(padding=(0, 0, 46, 6), fill=0, padding_mode='edge'),
                transforms.ToTensor(),
            ])
        )
        # Index the 40 attributes and extract the male (col = 20), young(39) and eyeglasses(15) categories
        train_set.attr = train_set.attr[:train_set.attr.size(0), (15, 20, 39)]
        test_set.attr = test_set.attr[:test_set.attr.size(0), (15, 20, 39)]

        kwargs = {'num_workers': 1,
                  'pin_memory': True} if cuda_available else {}  # This code was taking from the MNIST example code for use when using CUDA
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=train_batch, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=test_batch, shuffle=True, **kwargs)

        scheduler = StepLR(optimizer, step_size=1, gamma=gamma)  # decreases learning rate every epoch

        train_loss = []
        train_acc = []
        test_loss = [1] # initialise values to plot epoch 0
        test_acc = [0.5]
        print('Training:')
        for epoches in range(1, epoches + 1):
            train(model, train_loader, optimizer, epoches, device, train_loss, train_acc)
            print('Testing:')
            test(model, test_loader, device,test_acc,test_loss)
            scheduler.step()

        if save == True:
            torch.save(model, (".data/Models/{}".format(model_num)))
        print("Getting values for confusion matrix...")
        # get training predictions without updating weights for confusion matrix
        with torch.no_grad():

            pred_train_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch)
            train_preds = get_all_preds(model, pred_train_loader, device)

        # create a 3d confusion matrix, one 2.2 for each class (glasses, young, male)
        # confusion matrix calc inspired by deeplizard tutorial
        # https://deeplizard.com/learn/video/0LhiS6yu2qQ
        cmt = torch.zeros(3, 2, 2)
        stacked = torch.stack(
            (
                test_set.attr.type(torch.IntTensor)
                , train_preds.type(torch.IntTensor)
            )
            , dim=1)
        print("Calculating confusion matrix...")
        for p in stacked: # use clever array indexing to populate confusion matrix
            j, k = p.tolist()
            cmt[0, j[0], k[0]] = cmt[0, j[0], k[0]] + 1
            cmt[1, j[1], k[1]] = cmt[1, j[1], k[1]] + 1
            cmt[2, j[2], k[2]] = cmt[2, j[2], k[2]] + 1
        print("Confusion matrix: \n",cmt.type(torch.IntTensor))

        # plot train/test accuracy per epoch
        plt.subplot(121)
        plt.xticks(range(0, epoches + 1))
        yvals = np.arange(0, epoches, 1 / (- (-train_set.attr.size(0) // train_batch))) # use ceiling calculation to solve number of training iterations, then divide by total batches to organise y axis by epoch
        plt.xlabel('Epoches')
        plt.ylabel('Accuracy')
        plt.plot(yvals, train_acc, 'g', label="Train accuracy")
        plt.plot(test_acc, 'bs', label="Test accuracy")
        plt.plot(test_acc, 'b') # extra plot not needed, just joins the squares and makes it look nicer
        plt.legend(loc="upper left")

        # plot train/test loss per epoch
        plt.subplot(122)
        plt.xticks(range(0, epoches + 1))
        plt.xlabel('Epoches')
        plt.ylabel('Loss')
        plt.plot(yvals, train_loss, 'g', label="Train loss")
        plt.plot(test_loss, 'bs', label="Test loss")
        plt.plot(test_loss, 'b')
        plt.legend(loc="upper left")
        plt.show()


main()
