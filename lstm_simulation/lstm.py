import torch
import torch.nn as nn

#Setting seed b/c this involves random processes and setting seed allows reproducibility
torch.manual_seed(420)

"""
References: 
https://romanorac.github.io/machine/learning/2019/09/27/time-series-prediction-with-lstm.html
https://www.jessicayung.com/lstms-for-time-series-in-pytorch/

"""

class LSTM(nn.Module):

    """
    This class is a wrapper for an LSTM.

    Luckily (or to my dismay), PyTorch has a built in LSTM that you can call with nn.lstm(..)
    However, in order to use the results of this LSTM you typically have to do some post-processing

    This class handles that post-processing.

    Note: Currently, this is very barebones. Feel free to experiment and add multiple self.lstm, etc.

    """

    def __init__(self, 
                input_dim: int, 
                hidden_dim: int, 
                batch_size:int,
                num_layers = 1, 
                output_dim = 1):
                
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.output_dim = output_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)
        self.output_layer = nn.Linear(self.hidden_dim, output_dim)
        self.init_hidden()
    
    #resets hidden state
    def init_hidden(self):
        self.hidden =  (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                        torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))
    
    #Reference: https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html
    def forward(self, input):
        batch_sz, seq_length = input.size()
        input_reshaped = input.view(seq_length, batch_sz, -1)
        output, self.hidden = self.lstm(input_reshaped)

        #By torch default, the output we want is the last slice of output
        #This reshapes allow us to pass into our output layer
        output_reshaped = output[-1].view(batch_sz,-1)

        output = self.output_layer(output_reshaped)
        return output
