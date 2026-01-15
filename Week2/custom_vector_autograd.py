import numpy as np
import numpy.typing as npt



class CustomTensorVariable:
    def __init__(self, value:npt.NDArray[np.float16], require_grad:bool=True) -> None:
        self.value : npt.NDArray[np.float16] = value
        self.require_grad : bool = require_grad
        self.grad = np.zeros_like(value)
        self._backward = lambda : None
        self._prev = set()

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.value, dtype=np.float16)
        for node in reversed(topo):
            node._backward()


    def zero_grad(self):
        self.grad = np.zeros_like(self.value)


    def __add__(self, other):
        other = other if isinstance(other, CustomTensorVariable) else CustomTensorVariable(np.array(other, dtype=np.float16))
        out = CustomTensorVariable(self.value + other.value, require_grad=self.require_grad or other.require_grad)

        def _backward():
            if self.require_grad:
                self.grad += out.grad
            if other.require_grad:
                other.grad += out.grad

        out._prev = {self, other}
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, CustomTensorVariable) else CustomTensorVariable(np.array(other, dtype=np.float16))
        out = CustomTensorVariable(self.value - other.value, require_grad=self.require_grad or other.require_grad)

        def _backward():
            if self.require_grad:
                self.grad += out.grad
            if other.require_grad:
                other.grad -= out.grad

        out._prev = {self, other}
        out._backward = _backward
        return out

    def ReLU(self):
        out = CustomTensorVariable(np.maximum(self.value, 0), require_grad=self.require_grad)
        out._prev = {self}
        def _backward():
            if self.require_grad:
                self.grad += out.grad * (self.value > 0).astype(np.float16)
        out._backward = _backward
        return out

    def matmul(self, other):
        other = other if isinstance(other, CustomTensorVariable) else CustomTensorVariable(np.array(other, dtype=np.float16), require_grad=False)
        out = CustomTensorVariable(np.matmul(self.value, other.value), require_grad = self.require_grad or other.require_grad)

        def _backward():
            if self.require_grad:
                self.grad += np.matmul(out.grad, other.value.T)
            if other.require_grad:
                other.grad += np.matmul(self.value.T, out.grad)

        out._prev = {self, other}
        out._backward = _backward
        return out

    def __pow__(self, p):
        out = CustomTensorVariable(np.power(self.value, p))
        out._prev = {self}
        def _backward():
            if self.require_grad:
                self.grad += (p * np.power(self.value, p-1) * out.grad).astype(np.float16)

        out._backward = _backward
        return out
