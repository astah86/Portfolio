import os
import torch
import numpy as np
import dill
from torch import nn



def converting (symbol):
    if symbol < 10:
        symbol = symbol + 48
    elif symbol < 36:
        symbol = symbol + 55
    elif symbol < 38:
        symbol = symbol + 61
    elif symbol < 43:
        symbol = symbol + 62
    elif symbol < 44:
        symbol = symbol + 67
    elif symbol < 46:
        symbol = symbol + 69
    else:
        symbol = symbol + 70

    return chr(symbol)


class MMM(nn.Module):

    def __init__(self):
        super(MMM, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=128 * 2 * 2, out_features=6000)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=6000, out_features=1200)
        self.fc3 = nn.Linear(in_features=1200, out_features=400)
        self.fc4 = nn.Linear(in_features=400, out_features=47)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.drop(out)
        out = self.fc3(out)
        out = self.fc4(out)

        return out


class Model:
    def __init__(self):
        self.model = MMM()
        self.model.load_state_dict(torch.load('myapp/cnn.ckpt'))


        


    def predict(self, x):

        '''
        Parameters
        ----------
        x : np.ndarray
            Входное изображение -- массив размера (28, 28)
        Returns
        -------
        pred : str
            Символ-предсказание 
        '''

        # fig, axs = plt.subplots(1, 1, figsize=(20, 4))
        # axs.imshow(x, cmap='gray')


        model_predict = self.model(x)

        
        predicted_class = torch.argmax(model_predict, dim=1).item()

        result = str(converting(predicted_class))

        return result

        # your code here
