import numpy as np

class Tensor:
    ''' 
    Initialization: Convert input data to a NumPy array if it isn't one already 
    '''
    def __init__(self, data):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        self.data = data

    ''' 
    String Representation: Provides a string representation for debugging and display 
    '''
    def __repr__(self):
        return f"Tensor({self.data})"

    ''' 
    Addition: Element-wise addition with another Tensor or a scalar 
    '''
    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    ''' 
    Reverse Addition: Handles addition when Tensor is on the right side of the operator 
    '''
    def __radd__(self, other):
        return self.__add__(other)

    ''' 
    Subtraction: Element-wise subtraction with another Tensor or a scalar 
    '''
    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    ''' 
    Reverse Subtraction: Handles subtraction when Tensor is on the right side of the operator 
    '''
    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        else:
            return Tensor(other - self.data)

    ''' 
    Multiplication: Element-wise multiplication with another Tensor or a scalar 
    '''
    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    ''' 
    Reverse Multiplication: Handles multiplication when Tensor is on the right side of the operator 
    '''
    def __rmul__(self, other):
        return self.__mul__(other)

    ''' 
    Dot Product: Matrix multiplication (dot product) with another Tensor 
    '''
    def dot(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.dot(self.data, other.data))
        else:
            raise ValueError("Dot product requires another Tensor")

    ''' 
    Shape: Returns the shape of the Tensor 
    '''
    def shape(self):
        return self.data.shape

    ''' 
    Transpose: Returns the transposed Tensor 
    '''
    def transpose(self):
        return Tensor(self.data.T)

    ''' 
    Sum: Computes the sum of elements along the specified axis 
    '''
    def sum(self, axis=None):
        return Tensor(self.data.sum(axis=axis))

    ''' 
    Mean: Computes the mean of elements along the specified axis 
    '''
    def mean(self, axis=None):
        return Tensor(self.data.mean(axis=axis))

    ''' 
    Max: Computes the maximum of elements along the specified axis 
    '''
    def max(self, axis=None):
        return Tensor(self.data.max(axis=axis))

    ''' 
    Min: Computes the minimum of elements along the specified axis 
    '''
    def min(self, axis=None):
        return Tensor(self.data.min(axis=axis))
    