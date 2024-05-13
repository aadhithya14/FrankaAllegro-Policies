import torch
import torchvision
import torch.nn as nn
from .learner import Learner
import os 


class DynamicsLearner(Learner):
    def __init__ (self,dynamics_learner,optimizer):
        self.optimizer = optimizer
        self.dynamics_learner = dynamics_learner
        
    def to(self,device):
        self.device=device
        self.dynamics_learner.to(device)

    def train(self):
        self.dynamics_learner.train()

    def eval(self):
        self.dynamics_learner.eval()

    def save(self,checkpoint_dir,model_type='best'):
        torch.save(self.dynamics_learner.state_dict(),os.path.join(checkpoint_dir,f'dynamics_learner_{model_type}.pt'),_use_new_zipfile_serialization=False)
