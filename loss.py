import numpy as np

from tensor import Tensor

'''
Base Class
'''
class Loss:
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
    def grad(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError
    
'''
MSE Loss Function
'''
class MSE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        '''
        Calculate the Mean Squared Error (MSE) loss.
        '''
        return np.mean((predicted.data - actual.data) ** 2)
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        '''
        Calculate the gradient of the MSE loss.
        '''
        return Tensor(2 * (predicted.data - actual.data) / actual.data.size)
    
'''
MAE Loss Function
'''
class MAE(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        '''
        Calculate the Mean Absolute Error (MAE) loss.
        '''
        return np.mean(np.abs(predicted.data - actual.data))
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        '''
        Calculate the gradient of the MAE loss.
        '''
        grad = np.where(predicted.data > actual.data, 1, -1)
        return Tensor(grad / actual.data.size)
    