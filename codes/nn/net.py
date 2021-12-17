
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class Net(nn.Module):
    """docstring for Net."""

    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1600),
            nn.BatchNorm1d(1600),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1600, 800),
            nn.BatchNorm1d(800),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(800, 400),
            nn.BatchNorm1d(400),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(100, 32),
            nn.BatchNorm1d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        x = self.classifier(x)
        return x

    def get_available_device(self):
        if torch.cuda.is_available:
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        return device

    def save_model(self, path="../nn_models/nn_model_best.pth"):
        torch.save(self.state_dict(), path)

    def my_train(self, trainset, num_of_inputs):
        self.train(mode=True)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
        trainloader = DataLoader(
            trainset, batch_size=100, shuffle=True, drop_last=True)

        epoch_losses = []
        epoch_accuracies = []
        for epoch in range(30):
            print("epoch: %d " % (epoch + 1))
            # file.write("epoch: %d " % (epoch + 1))
            epoch_loss = 0.0
            correct = 0
            total = 0
            for i, data in enumerate(trainloader, 0):
                x, y = data
                optimizer.zero_grad()
                outputs = self(x)  # forward net(inputs)

                # calculating loss
                loss = criterion(outputs, y)  # loss calculation
                loss.backward()  # backward
                optimizer.step()  # optimize
                epoch_loss += loss.item()
                if i % 50 == 49:
                    print("I: %d, loss: %.5f" %
                          (i + 1, epoch_loss / ((i + 1) * 100)))

                # calculating Accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                if i % 50 == 49:
                    print("I %d, correct: %d, total: %d, accuracy: %.5f" %
                          (i + 1, correct, total, correct / total))

            print("avg epoch loss: %.5f" % (epoch_loss / num_of_inputs))
            epoch_losses.append(epoch_loss / num_of_inputs)
            print('Accuracy on test data: %.5f %%' % ((100 * correct) / total))
            epoch_accuracies.append((100 * correct) / total)
            print(epoch_losses)
            print(epoch_accuracies)

        self.save_model()

    def my_test(self, testset):
        self.eval()
        testloader = DataLoader(testset, batch_size=100,
                                shuffle=True, drop_last=True)
        # dataiter = iter(testloader)
        # data, labels = dataiter.next()
        # classes = ('NO', 'YES')
        # print('GroundTruth: ', ' '.join('%5s' %
        #                                 classes[labels[j]] for j in range(4)))
        # outputs = self(data)
        # _, predicted = torch.max(outputs, 1)
        # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
        #                               for j in range(4)))
        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                x, y = data
                outputs = self(x)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                if i % 50 == 49:
                    print("I %d, correct: %d, total: %d" %
                          (i + 1, correct, total))

        print('Accuracy on test data: %d %%' % (
            100 * correct / total))
