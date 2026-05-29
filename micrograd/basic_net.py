import math
import random
from engine import Value

class neuron:
    def __init__(self, input_size):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(input_size)]
        self.bias = Value(random.uniform(-1, 1))
        
    def __call__(self, x):
        sumation = Value(0.0)
        for xi, wi in zip(x, self.weights):
            sumation += (xi*wi)
        raw = sumation + self.bias
        res = raw.sigmoid()
        return res
    
    def parameters(self):
        return self.weights + [self.bias]

class layer:
    def __init__(self, input_size, output_size):
        self.neurons = [neuron(input_size) for _ in range(output_size)]
    
    def __call__(self, x):
        l = [n(x) for n in self.neurons]
        return l
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class basic_MLP:
    def __init__(self, input_size, layers, output_size):
        lz = [input_size] + layers + [output_size]
        self.layers = [layer(lz[i], lz[i+1]) for i in range(len(lz))]
        
    def __call__(self, x):
        for lay in self.layers:
            x = lay(x)
        return x
    
    def parameters(self):
        return [p for lay in self.layers for p in lay.parameters()]    
        