import torch
from torch.utils.data import Dataset
import pandas as pd
import math
import numpy as np
import os

class Action_Energy_Dataset(Dataset):

    """
    This is a torch dataset class for the new_action_energy_data directory in the repo.
    Note the new_action_energy_data directory contains a reformatted version of the data
    from before in order to make it easier to parse. 
    
    """

    def __init__(self, file_path, split = 'train'):
        """
        Args: File_path: Path to one of the csv files in the new_action_energy_data directory
            Split: String either saying 'train', 'valid', or 'dummy test' (this is b/c I don't have a true testset)
                    
                    Background:
                    In ML, ppl typically break their data into 3 categories: 
                    1. Training Data: The Data used for learning
                    2. Validation Data: The Data used to test how well you're learning
                    3. Test Data: The final dataset where you output your predictions 
                        (here you may not be able to tell if your predictions are right/wrong
                        whereas in the categories above you can tell if you're right/wrong).
                    

        """
        if(split not in ['train','valid','dummy_test']):
            raise NotImplementedError

        if(not os.path.exists(file_path)):
            raise ValueError('File not found at "{0}".'.format(file_path))

        self.df = pd.read_csv(file_path)
        self.data = torch.tensor(self.df['Point'].values).float()
        self.target = torch.tensor(self.df['Energy'].values).float()

        if(split == 'train'):
            num_train_samples = math.floor(0.8*self.target.shape[0])
            self.data = self.data[0:num_train_samples]
            self.target = self.target[0:num_train_samples]
        
        elif(split == 'valid'):
            num_valid_samples = math.floor(0.2*self.target.shape[0])
            self.data = self.data[0:num_valid_samples]
            self.target = self.target[0:num_valid_samples]



    def __len__(self):
        """
        Every ten rows in the csv file is formatted (Hour, Point Value, Energy Use)
        with Hour going from 0 to 10. 

        This function tells us how many days of data are in our current dataset. 
        """
        return math.floor(self.target.shape[0] / 10)

    def __getitem__(self, idx):
        """
        Returns 2 arrays: 1) An array of point values for that day
                        2) An array of corresp. energy usages values for that day
        
        Note: These are vectors with 10 entries (1 for each hr)
        """
        start_idx = idx * 10
        end_idx = start_idx + 10
        x = self.data[start_idx:end_idx]
        y = self.target[start_idx : end_idx]
        return x, y
