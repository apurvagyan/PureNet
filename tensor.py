import numpy as np

class Tensor:
    def __init__(self, data):
        if isinstance(data, (list, tuple)):
            data = np.array(data)
        self.data = data

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data + other.data)
        else:
            return Tensor(self.data + other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data - other.data)
        else:
            return Tensor(self.data - other)

    def __rsub__(self, other):
        if isinstance(other, Tensor):
            return Tensor(other.data - self.data)
        else:
            return Tensor(other - self.data)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return Tensor(self.data * other.data)
        else:
            return Tensor(self.data * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def dot(self, other):
        if isinstance(other, Tensor):
            return Tensor(np.dot(self.data, other.data))
        else:
            raise ValueError("Dot product requires another Tensor")

    def shape(self):
        return self.data.shape

    def transpose(self):
        return Tensor(self.data.T)

    def sum(self, axis=None):
        return Tensor(self.data.sum(axis=axis))

    def mean(self, axis=None):
        return Tensor(self.data.mean(axis=axis))

    def max(self, axis=None):
        return Tensor(self.data.max(axis=axis))

    def min(self, axis=None):
        return Tensor(self.data.min(axis=axis))