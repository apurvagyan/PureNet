import numpy as np

from tensor import Tensor

class Layer:
    def __init__(self) -> None:
        pass

    def forward(self, inputs: Tensor) -> Tensor:
        '''
        Produces outputs corresponding to the inputs for this layer
        '''
        raise NotImplementedError
    
    def backprop(self, grad: Tensor) -> Tensor:
        '''
        Backpropagates this gradient through the layer
        '''
        raise NotImplementedError
