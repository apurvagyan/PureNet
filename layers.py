import numpy as np
from typing import Dict

from tensor import Tensor

'''
Base Layer Implementation
'''
class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

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

'''
Linear Layer Implementation
'''
class Linear(Layer):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()

        # inputs are (batch_size, input_size)
        # outputs are (batch_size, output_size)
        self.params["w"] = np.random.randn(input_size, output_size)
        self.params["b"] = np.random.randn(output_size)

    def forward(self, inputs: Tensor) -> Tensor:
        self.inputs = inputs

        return inputs @ self.params["w"] + self.params["b"]
    
    def backprop(self, grad: Tensor) -> Tensor:
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad

        # return gradient with respect to inputs
        return grad @ self.params["w"].T

'''
Activation Layer Implementation
'''


