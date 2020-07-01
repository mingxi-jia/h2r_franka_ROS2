from abc import abstractmethod

import torch

class HisBase:
    def __init__(self):
        self.his = None

    def initHis(self, num_processes):
        self.his = torch.zeros((num_processes, 1, self.patch_size, self.patch_size))

    @abstractmethod
    def getCurrentObs(self, *args):
        pass

    @abstractmethod
    def getNextObs(self, *args):
        pass

    @abstractmethod
    def updateHis(self, *args):
        pass


