import math
import os
import random

random.seed(42)

if not os.path.exists("data.txt"):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "data.txt")

with open("data.txt") as file:
    docs = [line.strip() for line in file if line.strip()]

random.shuffle(docs)
print(f"num docs: {len(docs)}")


# Tokenizer: strings -> integers and back
#   - for this example, it will be a char-level tokenizer
class Tokenizer:
    __slots__ = ("__decoder", "__encoder", "vocab")

    def __init__(self, vocab: set[str]) -> None:
        if set(vocab) != vocab:
            raise ValueError("Each element in the vocabulary must appear only once")

        if "<BOS>" in vocab or "<EOS>" in vocab:
            raise ValueError("Special tokens cannot be included in the vocabulary")

        self.vocab = vocab
        self.vocab |= {"<BOS>", "<EOS>"}

        self.__encoder = {}
        self.__decoder = {}
        for i, word in enumerate(vocab):
            self.__encoder[word] = i
            self.__decoder[i] = word

    def encode(self, s: str) -> list[int]:
        return [self.__encoder[char] for char in s]

    def decode(self, seq: list[int]) -> str:
        return "".join([self.__decoder[i] for i in seq])


# Node of the comp graph, represented as a tree
#   - I don't need to learn how to write an Autograd for now, so I will just mostly copy Karpathy's implementation
# fmt: off
class Value:
    __slots__ = ("_children", "_local_grads", "data", "grad")

    def __init__(self, data, children=(), local_grads=()) -> None:
        self.data = data                    # scalar value of this node, updated on forward pass
        self.grad = 0                       # derivative of the loss w.r.t. this node, updated in backward pass
        self._children = children           # direct children on this node
        self._local_grads = local_grads     # derivatives of this node w.r.t. its children, for the chain rule

    def __add__(self, other):
        # Transform to "Value" if it's just a plain number
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        # first, we build the sequence in reversed order using DFS
        # then, we compute the grad by applying the chain rule in order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad
# fmt: on

