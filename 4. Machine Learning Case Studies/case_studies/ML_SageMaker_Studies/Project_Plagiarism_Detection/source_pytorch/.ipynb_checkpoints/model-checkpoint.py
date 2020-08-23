# torch imports
import torch.nn.functional as F
import torch.nn as nn


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
    """
    Define a neural network that performs binary classification.
    The network should accept your number of features as input, and produce 
    a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
    
    Notes on training:
    To train a binary classifier in PyTorch, use BCELoss.
    BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
    """

    ## TODO: Define the init function, the input params are required (for loading code in train.py to work)
    def __init__(self, input_features, hidden_dim, n_hidden):
        """
        Initialize the model by setting up linear layers.
        Use the input parameters to help define the layers of your model.
        :param input_features: the number of input features in your training/test data
        :param hidden_dim: helps define the number of nodes in the hidden layer(s)
        :param output_dim: the number of outputs you want to produce
        """
        super(BinaryClassifier, self).__init__()

        # define any initial layers, here
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.n_hidden = n_hidden
        
        # sigmoid layer
        self.sig = nn.Sigmoid()
    
        prev_layer = input_features
        
        layers = []

        # Dense layers
        for h in range(n_hidden):
            # Linear and the ReLU
            layer = nn.Linear(prev_layer, hidden_dim)
            activation = nn.ReLU()
            drop = nn.Dropout()
            layers.extend([layer, activation, drop])
            prev_layer = hidden_dim
            
        self.module_list = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, 1)
        self.sig = nn.Sigmoid()
        # Output dimension will only be 1 for binary classifier
        # So removing the output_dim variable as a feature
    
    ## TODO: Define the feedforward behavior of the network
    def forward(self, x):
        """
        Perform a forward pass of our model on input features, x.
        :param x: A batch of input features of size (batch_size, input_features)
        :return: A single, sigmoid-activated value as output
        """

        # define the feedforward behavior
        for f in self.module_list:
            x = f(x)
            
        x = self.output(x)
        x = self.sig(x)

        return x
    