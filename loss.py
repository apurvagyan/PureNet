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
    
class CrossEntropy(Loss):
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        '''
        Calculate the Cross-Entropy loss.
        '''
        epsilon = 1e-12
        predicted_data = np.clip(predicted.data, epsilon, 1. - epsilon)

        N = predicted_data.shape[0]
        ce_loss = -np.sum(actual.data * np.log(predicted_data + 1e-9)) / N

        return ce_loss
    
    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor:
        '''
        Calculate the gradient of the Cross-Entropy loss.
        '''
        epsilon = 1e-12
        predicted_data = np.clip(predicted.data, epsilon, 1. - epsilon)

        return Tensor(-actual.data / predicted_data)
    