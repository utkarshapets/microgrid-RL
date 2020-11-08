import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import IPython

import time
from lstm import LSTM

"""
References:
[1] https://romanorac.github.io/machine/learning/2019/09/27/time-series-prediction-with-lstm.html

"""

class Optimizer:
    """
    Ditto to reference [1], this is a helper class designed to make training and testing the model easier


    *** Note *** 
     This code is currently using Tensorboard to make the training process easier to track.
     In order to use tensorboard: please refer to https://pytorch.org/docs/stable/tensorboard.html

     Before you run main.py, in another terminal tab/window cd into the lstm_simulation dir
     and run: tensorboard --logdir=runs

     That ^ command tells Tensorboard to start and saves the logs in lstm_simulation/runs 
     (you can change runs to any other name)
     If you wish to save outside of lstm_simulation, insert the path in line 44
     Either way, you will have to call tensorboard from this directory (atleast I believe so)

     Tensorboard allows you to view the training process in real-time. 
     After you run "tensorboard --logdir=runs", go to http://localhost:6006/ on your browser 
    """


    def __init__(self, 
                model: LSTM,
                loss_function = nn.MSELoss(),  ##### standard for regression 
                learning_rate= 1e-3,
                optimizer = torch.optim.Adam, # search other optimizers 
                device = 'cpu',
                output_file_name = None):
        
        self.model = model
        self.loss_fn = loss_function
        self.lr = learning_rate
        self.optimizer = optimizer(model.parameters(), lr=learning_rate)
        self.device = torch.device(device)

        #If you want to save logs somewhere else insert path in SummaryWriter(path)
        self.writer = SummaryWriter(log_dir = "runs/" + output_file_name)
    
    def train(self,
            train_dataloader: DataLoader,
            val_dataloader: DataLoader, 
            num_epochs = 1):
        """
        Args: train_dataloader: (Dataloader) containing only training data samples
              val_dataloader: (DataLoader) containing validation samples 
                        (this is used to plot the validation accuracy and loss throughout training process)
            num_epochs: (Int) think of this as the number of iterations you want to train


        Note: Currently, the train function takes in torch DataLoader (not Datasets),
            keep this in mind if you plan on tweaking the Dataset. To create a Dataloader from
            a Dataset class please refer to lines 26-30 of main.py

        """

        self.model.train()

        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss = 0
            train_correct_ct = 0
            rmse = 0

            #resets hidden state
            self.model.hidden = self.model.init_hidden()
            for batch_number, (x, y) in enumerate(train_dataloader):
                x = x.to(device=self.device)
                y = y.to(device = self.device)

                predicted_y = self.model(x)
                self.optimizer.zero_grad()
                loss = self.loss_fn(predicted_y, y)
                loss.backward(retain_graph = True)
                self.optimizer.step()

                train_loss += loss
                train_correct_ct += predicted_y.eq(y.data).sum().item() # <-- 

                print("predicted_y")
                print(predicted_y)

                print("y data")
                print(y.data)

                ### ^--- ?? should maybe be rounded -- a bit. 

            
            train_loss = train_loss / len(train_dataloader)
            train_acc= train_correct_ct / len(train_dataloader.dataset)
            validation_loss, validation_acc = self.eval_(val_dataloader)

            IPython.embed()

            elapsed = time.time() - start_time
            print(
                "Epoch {:d} Train loss: {:.2f}. Train Accuracy: {:.2f}. Validation loss: {:.2f}. Validation Accuracy: {:.2f}. Elapsed time: {:.2f}ms. \n".format(
                epoch + 1, train_loss, train_acc, validation_loss, validation_acc, elapsed)
                )

            self.writer.add_scalar('Training Loss',train_loss, epoch + 1)
            self.writer.add_scalar('Training Accuracy', train_acc, epoch + 1)
            self.writer.add_scalar('Validation Accuracy', validation_acc, epoch + 1)

            self.optimizer.zero_grad()
    
    def eval_(self,
            dataloader: DataLoader):
        
        """
        Inputs: dataloader: (DataLoader). Assumes Dataloader contains labels!!!

        Note: This function returns the self.model's loss and accuracy on the data in dataloader
        """
            
        self.model.eval()
        total_loss = 0
        correct_ct = 0
        with torch.no_grad():
            for batch_number, (x,y) in enumerate(dataloader):
                x = x.to(device = self.device)
                y = y.to(device = self.device)

                predicted_y = self.model(x)
                correct_ct += predicted_y.eq(y.data).sum().item()
                loss = self.loss_fn(predicted_y,y)
                total_loss += loss

        total_loss = total_loss / len(dataloader)
        acc = correct_ct / len(dataloader.dataset)

        return total_loss, acc
    
    def test(self, test_dataloader: DataLoader):

        """
        Inputs: dataloader: (DataLoader). Assumes Dataloader DOES NOT contain labels!!!

        Note: This function returns self.model's prediction for a test set
        """
        
        self.model.eval()
        
        predictions = []
        with torch.no_grad():
            for batch_number, x_batch in enumerate(test_dataloader):
                x_batch = x_batch.to(device=self.device)
                predictions.append(self.model(x_batch))
        
        return torch.Tensor(predictions)


    