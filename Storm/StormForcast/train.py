import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import datetime

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from livelossplot import PlotLosses
from torch.utils.data import Subset

from .datasets import Storm_Dataset
from .models import ImageRegressionCNN
from .utilities import set_seed


class Train_Validate():
    """
    Initial the train class, which is used to train and test both tasks
    Parameters
    ----------
    device: is to decide whether cpu or cuda. passes as string,
            should be used directly
    path: the path to the data folder you'd like to use
          (also used for surprise storm)
    task: (str) either WindSpeed
    batch_size: a hyperparameter
    lr: learning rate
    split_method: the order of loaded dataset, 'random' or 'time'
                if split method is time, the data points is then in time order
                if split method is random, data points is a random
    num_storms: the number of storms will be loaded into the dataset
    surprise storm: set to True when use this package to
                    deal with a surprise storm
    resume: set to True for transfer learning
    resume_path: the path to model to resume from (pth file)

    e.g.
    Load 10 storms, without resume, not for surprise storm:
    int_model = StormForcast.Train_Validate(
        '/content/gdrive/MyDrive/gp2/Selected_Storms_curated/',
        task='WindSpeed', device='cuda',
        batch_size_train=32, batch_size_val=1000, batch_size_test=1000,
        lr=2e-3, epoch=100, split_method='random', num_storms=10)

    Load 1 storm (bc I've trained), with resume, not for surprise storm
    model_resume = StormForcast.Train_Validate(
        '/content/gdrive/MyDrive/gp2/Selected_Storms_curated/',
        task='WindSpeed', device='cuda',
        batch_size_train=32, batch_size_val=1000, batch_size_test=1000,
        lr=2e-3, epoch=100, split_method='random',
        num_storms=1, resume=True, resume_path='CNNGeneral_epoch_16.pth')

    Load surprise storms, with resume:
    surprise_model = StormForcast.Train_Validate(
        './tst/tst', task='WindSpeed', device='cuda',
        batch_size_train=32, batch_size_val=100, batch_size_test=100,
        lr=2e-3, epoch=20, split_method='random', num_storms=3,
        surprise_storm=True,
        resume=True, resume_path='CNNGeneral_epoch_16.pth')
    """

    def __init__(self, path, task, device, batch_size_train,
                 batch_size_val, batch_size_test, lr, epoch,
                 split_method, num_storms, weight_decay=1.5e-5,
                 surprise_storm=False, seed=42, resume=False,
                 resume_path=None):

        # Basic Settings
        print('setting basic parameters ......')
        self.path = path
        self.task = task
        self.device = device
        self.surprise_storm = surprise_storm
        self.resume = resume
        self.resume_path = resume_path
        print('done')

        # Dataset Parameters
        self.num_storms = num_storms
        self.split_method = split_method

        # Hyperparameters Setting
        print('setting hyperparameters ......')
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.lr = lr
        self.epoch = epoch
        self.weight_decay = weight_decay
        self.seed = seed
        print('done')

        if task == 'WindSpeed':
            # model
            print('initializing model, optimier, criterion ......')
            self.model = ImageRegressionCNN().to(self.device)
            self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                              weight_decay=self.weight_decay,
                                              lr=self.lr)
            self.criterion = nn.MSELoss()
            print('done')

            if self.resume:
                print('resuming previous model ......')
                checkpoint = torch.load(self.resume_path,
                                        map_location=torch.device('cpu'))
                self.model.load_state_dict(checkpoint['model_state_dict'])

                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) # noqa
                else:
                    print("Optimizer state not found in checkpoint."
                          "Initializing optimizer without loading state.")
                checkpoint = torch.load(self.resume_path, map_location=torch.device('cpu')) # noqa
                print('done')

            # Datasets
            if self.surprise_storm:
                print('loading data ......')
                self.ds = Storm_Dataset(self.path)
                self.ds_train = Subset(self.ds, range(0, 220))
                self.ds_val = Subset(self.ds, range(220, 242))
                self.ds_predict = Subset(self.ds, range(242, len(self.ds)))
                self.ds_labled = Subset(self.ds, range(0, 242))

                self.data_loader_train = DataLoader(self.ds_train,
                                                    batch_size_train)
                self.data_loader_val = DataLoader(self.ds_val,
                                                  batch_size_val)
                self.data_loader_test = DataLoader(self.ds_predict,
                                                   batch_size_test)
                print('done')

            else:
                print('loading data ......')
                self.ds = Storm_Dataset(self.path, num_storms=self.num_storms,
                                        split_method=self.split_method)
                self.ds_train, self.ds_val = train_test_split(self.ds,
                                                              test_size=0.3,
                                                              random_state=42)
                self.ds_val, self.ds_test = train_test_split(self.ds_val,
                                                             test_size=0.2,
                                                             random_state=42)

                self.data_loader_train = DataLoader(self.ds_train,
                                                    batch_size_train)
                self.data_loader_val = DataLoader(self.ds_val,
                                                  batch_size_val)
                self.data_loader_test = DataLoader(self.ds_test,
                                                   batch_size_test)
                print('done')

    def train(self, model, optimizer, criterion, loader):
        if self.task == 'WindSpeed':
            model.train()
            train_loss = 0
            for X, y in loader:
                X, y = X.to(self.device), y.to(self.device).float()
                optimizer.zero_grad()
                output = model(X.view(-1, 1, 256, 256)).squeeze()
                loss = criterion(output, y)
                loss.backward()
                train_loss += loss*X.size(0)
                optimizer.step()
            return train_loss/len(loader.dataset)

        if self.task == 'ImageGeneration':
            pass

    def validate(self, model, criterion, loader):
        if self.task == 'WindSpeed':
            model.eval()
            validation_loss = 0
            for X, y in loader:
                with torch.no_grad():
                    X, y = X.to(self.device), y.to(self.device).float()
                    output = model(X.view(-1, 1, 256, 256)).squeeze()
                    loss = criterion(output, y)
                    validation_loss += loss*X.size(0)

            return validation_loss/len(loader.dataset)

    def evaluate(self, dataset=None):
        '''
        If dataset is None, use the loaded data's test set
        to evaluate model; and if surprised storm is True, use
        the labled part of surprise storm to evaluate

        If dataset is path a storm (like pkh), then use the entire new
        dataset to evaluate
        '''
        if not dataset:
            if self.surprise_storm:
                dataset = self.ds_labled
            else:
                dataset = self.ds
        else:
            dataset = Storm_Dataset(dataset, num_storms=1,
                                    split_method='time')

        results = []
        reals = []
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x.to(self.device)
            result = self.model(x.view(1, 1, 256, 256)).squeeze().item()
            results.append(result)
            real = y
            reals.append(real)
        return results, reals

    def predict(self):
        '''
        Used when surprise storm is True
        It predicts all 13 images' wind speed
        '''

        dataset = self.ds_predict
        results = []
        for i in range(len(dataset)):
            x = dataset[i][0]
            x = x.to(self.device)
            result = self.model(x.view(1, 1, 256, 256)).squeeze().item()
            results.append(result)
        return results

    def draw_result(self, type='evaluate', dataset=None):
        '''
        draw the result of evaluate or predict function

        type: evaluate or predict
        '''
        if type == 'evaluate':
            results, reals = self.evaluate(dataset=dataset)
            plt.plot(results)
            plt.plot(reals)

        if type == 'predict':
            known_results, known_reals = self.evaluate(dataset=dataset)
            unk_results = self.predict()
            print(unk_results)
            plt.plot(known_results, label='Known Results', color='blue')
            plt.plot(known_reals, label='Known Reals', color='red',
                     linestyle='--')
            start_index_for_unk = len(known_results)
            x_values_for_unk = range(start_index_for_unk,
                                     start_index_for_unk + len(unk_results))
            plt.plot(x_values_for_unk, unk_results, label='Unknown Results',
                     color='green')
            plt.title('Wind speed prediction on surprised storm')
            plt.xlabel('time indice')
            plt.ylabel('wind speed')
            plt.legend()

    def train_whole(self, threshold=10):
        '''
        use this method to train the network

        threshold: when validation loss is less than this, save the model
        '''
        print('start training:')
        if self.surprise_storm:
            threshold = 60
        set_seed(self.seed)
        liveloss = PlotLosses()
        for epoch in range(self.epoch):
            logs = {}
            train_loss = self.train(self.model, self.optimizer, self.criterion,
                                    self.data_loader_train)
            logs['' + 'log loss'] = train_loss.item()

            validation_loss = self.validate(self.model, self.criterion,
                                            self.data_loader_val)
            logs['val_' + 'log loss'] = validation_loss.item()

            liveloss.update(logs)
            liveloss.draw()

            if validation_loss < threshold:
                # Save model and optimizer
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") # noqa
                self.save_model(model_path=f'CNNGeneral_epoch_{epoch+1}__{timestamp}.pth') # noqa
                print(f'Model saved at Epoch {epoch+1} with Validation Loss:{validation_loss:.4f}') # noqa

    def save_model(self, model_path):

        # Save model state and optimizer state
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }, model_path)
