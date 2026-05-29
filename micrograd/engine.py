import math
import numpy as np

class Value:
    def __init__(self, data, children=(), opr="", label=""):
        self.data = data
        self.grad = 0
        self._prev = set(children)
        self.opr = opr
        self._backward = lambda: None
        self.label = label
        
    def __repr__(self):
        return f"Value(data= {self.data}| grad= {self.grad}| label={self.label})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), "*")
        def _backward():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other
    
    def sigmoid(self):
        n = self.data
        t = 1 / (1 + math.exp(-n))
        out = Value(t, (self), "sigmoid")
        def _backward():
            self.grad += (t*(1-t))*out.grad
        out._backward = _backward
        return out
    
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
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        