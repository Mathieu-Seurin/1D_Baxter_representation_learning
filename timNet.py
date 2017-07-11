#!/bin/python
#coding: utf-8

import numpy as np
import torch

import time

from torch.utils.serialization import load_lua
import torch.legacy.nn as nnl

from baxter_interface import Head, Limb

import const
import matplotlib.pyplot as plt

from os.path import isfile

from copy import deepcopy

class DummyTimNet(object):

    def __init__(self):
        pass
    def forward(self,x):
        img = deepcopy(x[0])
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
        return x

    def cuda(self):
        pass

    def setMinMaxRbf(self, img1,img2):
        raise const.DrunkProgrammer("Cannot use Dummy timnet and rbf")

    def calcRBFvalue(self, x):
        numRbf = const.NUM_RBF
        means = np.linspace(self.minRepr,self.maxRepr,numRbf+2)
        means = means[1:-1] #Getting rid of first and last elements, because you don't want gaussians on the extremum
        std = np.sqrt(np.abs(means[0]-means[1])/2)
        values = [norm.pdf(x,means[i],std) for i in range(numRbf)]

        return torch.Tensor(np.array(values)).unsqueeze(0)

    def __call__(self,x):
        return self.forward(x)

class TrueNet(DummyTimNet):
    def __init__(self):
        super(TrueNet,self).__init__()
        ready = False
        while not ready:
            try:
                self.head = Head()
            except OSError:
                print "Waiting for Baxter to be ready, come"
                time.sleep(1)
                continue
            else:
                ready=True

    def forward(self,*args):
        x = self.head.pan()
        if const.RBF:
            x = self.calcRBFvalue(x)
            return x
        
        return torch.Tensor(np.array([x])).unsqueeze(0)

    def setMinMaxRbf(self, img1,img2):
        self.minRepr = -1.3
        self.maxRepr = 1.3

    def __call__(self,x):
        return self.forward(x)

class TrueNet3D(DummyTimNet):
    def __init__(self):
        super(TrueNet3D,self).__init__()
        self.state = None #This variable is changed in env.py, when doing an action,
                          #the env switch this variable to the absolute position of the gripper
        self.posButton = np.array(const.DEFAULT_BUTTON_POS)
        self.posButton[2] = 0.20
        #Needed because the network take the relative position
        
        self.stringFileMean = const.MODEL_PATH+'meanStdTrueState3D.npy'
        self.mean = None
        self.std = None

        if isfile(self.stringFileMean):
            self.logState = np.load(self.stringFileMean)
            self.mean = self.logState.mean(axis=0)
            self.std = self.logState.std(axis=0)
        else:
            self.logState = []
                         

    def forward(self,*args):
        assert not(self.state is None), "Problem, this variable should have changed"
        x = np.array(self.state) - self.posButton
        
        if type(self.logState) is list:
            self.logState.append(x)
            np.save(self.stringFileMean,np.array(self.logState))
            
        
        if not self.mean is None:
            x -= self.mean
            x /= self.std

        if const.RBF:
            raise const.DrunkProgrammer("Not available for 3D model")
        x = torch.Tensor(x).unsqueeze(0)
        return x
        
    def __call__(self,x):
        return self.forward(x)
        
class LuaModel(DummyTimNet):
    def __init__(self,modelName,batchSize=1):
        """Loading tim model from torch (in .t7 format)
        batchSize is needed because nn.view from torch doesn't seem to handle
        well the batch dimension"""
        super(LuaModel,self).__init__()
        self.useRBF = const.RBF
        self.load_preprocess_model(modelName)

    def load_preprocess_model(self,modelName):
        self.net = load_lua(const.TIM_PATH+modelName)
        self.meanRepr, self.stdRepr = load_lua(const.TIM_PATH+'meanStdRepr.t7')
        if const.MODEL=='repr':
            assert len(self.net.modules)==23
            #If you don't do that, you get error on dimension etc ...
            self.net.modules[19] = nnl.View(batchSize,100)
        elif const.MODEL in ['auto1','auto2'] :
            self.net.modules[19] = nnl.View(batchSize,100)
            self.net.modules[23:] = []
        
    def forward(self,x):
        """
        - Compute forward for lua model
        - Then normalize the representation using mean and std calculated during the representation learning
        """
        # img = x.cpu().numpy()
        # print "np.shape", img.shape
        # img = np.swapaxes(img,1,3)
        # img = np.swapaxes(img,2,1)

        # print "img",img 
        # plt.imshow(img[0], interpolation='nearest')
        # plt.show()

        x = self.net.forward(x)
        x = (x-self.meanRepr)/self.stdRepr

        if self.useRBF:
            x = self.calcRBFvalue(x.cpu()[0,0])
            
        return x

    def cuda(self):
        self.net.cuda()
        self.meanRepr = self.meanRepr.cuda()
        self.stdRepr = self.stdRepr.cuda()

    def setMinMaxRbf(self, img1,img2):
        repr1 = self.forward(img1).cpu()[0,0]
        repr2 = self.forward(img2).cpu()[0,0]

        if repr1>repr2:
            self.minRepr = repr2
            self.maxRepr = repr1
        else:
            self.minRepr = repr1
            self.maxRepr = repr2

        self.useRBF = True


class LuaModel3D(LuaModel):
    def __init__(self,modelName):
        super(LuaModel3D,self).__init__(modelName)

    def load_preprocess_model(self, modelName):
        self.net = load_lua(const.MODEL_3D+modelName)

    def forward(self,x):
        x = self.net.forward(x)
        if self.useRBF:
            x = self.calcRBFvalue(x.cpu()[0,0])
        return x
    def cuda(self):
        self.net.cuda()


def loadModel(modelName):
    if const.TASK > 2:
        model = LuaModel3D(modelName)
    else:
        model = LuaModel(modelName)
    return model

