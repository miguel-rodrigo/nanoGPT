# SelfGPT

Minimal representation of an LLM model in diferent languages, purely for learning purposes.

## Main ingredients in every implementation

Every implementation will have the same components and structure:

- A dataset
- A vocabulary and tokenizer adapted to the dataset, plus BOS and EOS tokens
- An autograd class for backprop, with a proper implementation of the derivative for the necesary operations
- Model parameters (size and such)
- Functions like activations, softmax and rmsnorm
- A state representation for all of the model weights in a hash table
- Adam optimizer
- Training loop
- Inference code
