from random import shuffle

import torch
import torchvision

import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn

import torch.optim as optim
import argparse

from src.Net.Net import Net
from src.Net.NetCustom import NetCustom
from torch.utils.data import SubsetRandomSampler

plt.interactive(False)


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Build neural network.')
    parser.add_argument('-bs', '--batch-size', metavar='BATCH-SIZE', type=int, default=64,
                        help='Batch size')
    parser.add_argument('-vs', '--validation-split', metavar='VALIDATION-SPLIT', type=float, default=0.2,
                        help='Percentage of the validation set (from 0 to 1)')
    parser.add_argument('-nf', '--num-folds', metavar='NUM-FOLDS', type=int, default=2,
                        help='Number of folds. If there is more folds than network type, folds network type will loop')
    parser.add_argument('-ne', '--num-epoch', metavar='NUM-EPOCH', type=int, default=15,
                        help='Number of epoch')
    parser.add_argument('-se', '--stop-epsilon', metavar='STOP-EPSILON', type=float, default=0.75,
                        help='Percentage of loss margin allowed before quiting training (0 = no margin, 1= up to 2 times best loss)')
    parser.add_argument('-be', '--best-epsilon', metavar='BEST-EPSILON', type=float, default=0.1,
                        help='Percentage of best loss before saving model')
    parser.add_argument('-rm', '--refresh-model', metavar='REFRESH-MODEL', type=int, default=1000,
                        help='Validating model rate')
    parser.add_argument('-d', '--device', metavar='[cpu | cuda:0]', type=str, default='cuda:0',
                        help='Device type (cpu or cuda, cpu will be taken if cuda is not available)')
    parser.add_argument('-trf', '--train-folder', metavar='TRAIN-FOLDER', type=str, default='../train_images',
                        help='Train folder')
    parser.add_argument('-tef', '--test-folder', metavar='TEST-FOLDER', type=str, default='../test_images',
                        help='Test folder')
    parser.add_argument('-mn', '--model-name', metavar='MODEL-NAME', type=str, default='model',
                        help='Model name, best model will have "_best" suffix')
    parser.add_argument('-mf', '--model-folder', metavar='MODEL-FOLDER', type=str, default='../models',
                        help='Model folder')
    parser.add_argument('-sa', '--save-all', metavar='[0|1]', type=int, default=0,
                        help='Save all model')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    trainset = torchvision.datasets.ImageFolder(root=args.train_folder, transform=Net.transform)

    dataset_size = len(trainset)
    fold_size = int(dataset_size / args.num_folds)
    fold_split_size = int(np.floor(args.validation_split * fold_size))
    indices = list(range(dataset_size))

    testset = torchvision.datasets.ImageFolder(root=args.test_folder, transform=Net.transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    net_type = [NetCustom().__class__, Net().__class__]  # Add your custom network class here (should be in Net module)
    net = []
    best_model = []
    train_loader = []
    validation_loader = []
    shuffle(indices)
    for i in range(args.num_folds):
        # Init network array
        net.append(net_type[i % len(net_type)]())
        net[i].to(device)
        best_model.append(net_type[i % len(net_type)]())

        # Init train and valid loader
        split_start = dataset_size - (fold_split_size * (i + 1))
        split_end = split_start + fold_split_size
        train_indices = indices[0:split_start - 1] + indices[split_end + 1:dataset_size]
        val_indices = indices[split_start:split_end]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader.append(torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                        sampler=train_sampler))
        validation_loader.append(torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                             sampler=valid_sampler))

    classes = ('Non-Visage', 'Visage    ')

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net[0].parameters(), lr=0.001, momentum=0.9)

    done = False
    print("Start training")
    for i in range(args.num_folds):
        best_loss = float('inf')
        print('Training fold %d' % i)
        for epoch in range(args.num_epoch):
            batch = 0
            for j, data in enumerate(train_loader[i], 0):
                X_train, y_train = data
                X_train, y_train = X_train.to(device), y_train.float().to(device)
                optimizer.zero_grad()

                predicted = net[i](X_train)
                y_train = torch.eye(2, device=device)[y_train.long()]

                loss = criterion(predicted, y_train)
                loss.backward()
                optimizer.step()

                if batch % args.refresh_model == 0:
                    running_loss = 0.0
                    for X_test, y_test in iter(validation_loader[i]):
                        X_test, y_test = X_test.to(device), y_test.float().to(device)
                        optimizer.zero_grad()

                        predicted = net[i](X_test)
                        y_test = torch.eye(2, device=device)[y_test.long()]
                        loss = criterion(predicted, y_test)
                        running_loss += loss.item()

                    if running_loss < best_loss * (1 + args.best_epsilon):
                        if running_loss < best_loss:
                            best_loss = running_loss
                        best_model[i].load_state_dict(net[i].state_dict())
                    elif running_loss > best_loss * (1 + args.stop_epsilon):
                        print("Exit because of over fitting")
                        done = True
                        break

                    print("Epoch {} \t[batch = {}] \tactual {} \tbest {}".format(epoch, batch, running_loss, best_loss))

                batch += 1
            if done:
                break

    print('Finished Training')

    # Compare model
    best_fold = [0, .0]
    for i in range(args.num_folds):
        total = .0
        correct = .0
        for data in testloader:
            images, labels = data
            if device.type != 'cpu':
                images = images.cuda(device=device)
                labels = labels.cuda(device=device)
            _, predicted = torch.max(net[i](images).data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = (correct / total)
        if acc > best_fold[1]:
            best_fold[0] = i
            best_fold[1] = acc
        print('Accuracy of the network %d on the test set : %d %% ' % (i, (100 * acc)))
        if args.save_all:
            torch.save(best_model[i].state_dict(),
                       args.model_folder + "/" + args.model_name + "_" + str(i) + "." + best_model[
                           i].__class__.__name__)

    print('Model %d selected with %d %% of accuracy' % (best_fold[0], (100 * best_fold[1])))
    torch.save(best_model[best_fold[0]].state_dict(),
               args.model_folder + "/" + args.model_name + "_best." + best_model[best_fold[0]].__class__.__name__)

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    # print images
    imshow(torchvision.utils.make_grid(images))
    for i in range(4):
        print('\t'.join('%5s' % classes[labels[i * 8 + j]] for j in range(8)))
