import numpy as np

class Variable:
    def __init__(self, value:np.float16, require_grad:bool = True) -> None:
        self.value:np.float16 = value
        self.grad:np.float16 = np.float16(0.0)
        self._backward = lambda: None
        self._prev = set()
        self.require_grad = require_grad

    def zero_grad(self):
        self.grad = np.float16(0.0)

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

        self.grad = np.float16(1.0)
        for node in reversed(topo):
            node._backward()


    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(np.float16(other), require_grad=False)
        out = Variable(self.value + other.value, require_grad=self.require_grad or other.require_grad)

        def _backward():
            if self.require_grad:
                self.grad += out.grad
            if other.require_grad:
                other.grad += out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(np.float16(other), require_grad=False)
        out = Variable(self.value * other.value, require_grad=self.require_grad or other.require_grad)

        def _backward():
            if self.require_grad:
                self.grad += other.value * out.grad
            if other.require_grad:
                other.grad += self.value * out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(np.float16(other), require_grad=False)
        out = Variable(self.value - other.value, require_grad=self.require_grad or other.require_grad)

        def _backward():
            if self.require_grad:
                self.grad += out.grad
            if other.require_grad:
                other.grad -= out.grad

        out._backward = _backward
        out._prev = {self, other}
        return out

    def abs(self):
        out = Variable(np.abs(self.value), require_grad=self.require_grad)
        out._prev = {self}
        def _backward():
            if self.require_grad:
                self.grad += out.grad if self.value >= 0 else -out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(np.float16(other), require_grad=False)
        out = Variable((self.value/(other.value)), require_grad=(self.require_grad or other.require_grad))

        out._prev = {self, other}
        def _backward():
            if self.require_grad:
                self.grad += out.grad / other.value
            if other.require_grad:
                other.grad -= out.grad *(self.value / (other.value)**2)

        out._backward = _backward
        return out

    def sin(self):
        out = Variable(np.sin(self.value), require_grad=self.require_grad)
        out._prev = {self}
        def _backward():
            if self.require_grad:
                self.grad += out.grad * np.cos(self.value)
        out._backward = _backward
        return out

    def cos(self):
        out = Variable(np.cos(self.value), require_grad=self.require_grad)
        out._prev = {self}
        def _backward():
            if self.require_grad:
                self.grad -= out.grad * np.sin(self.value)
        out._backward = _backward
        return out


    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad}, require_grad={self.require_grad})"

