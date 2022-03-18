import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random


class FeedForward(nn.Module):
    
    """Initializes a Multiple layer Perceptron also known as the Feed Forward Network"""
    def __init__(self, input_dim, hidden, outputs):
        
        super(FeedForward, self).__init__()
        
        """Initializes ReLU as the activation function"""
        self.a = nn.ReLU() 
        """Dropout layer"""
        self.e = nn.Dropout(0.2)
        """Initializes 2 linear layers to transform input into output through hidden layers"""
        self.b = nn.Linear(input_dim, hidden) 
        self.c = nn.Linear(hidden, outputs)
        """Generates a log probability distribution for the outputs"""
        self.d = nn.LogSoftmax(dim = 0) 
        
        """Initializes weights for linear layers"""
        nn.init.xavier_uniform_(self.b.weight) 
        nn.init.xavier_uniform_(self.c.weight)

        """ Generates output log propabilities for the given input"""
    def forward(self, x):
        return self.d(self.c(self.e(self.a(self.b(x))))) 
    
class ThreatLevelClassifier():
    
    """Initializes the number of hidden layers, output classes (4 in this case due to the 4 ratings each case can be assigned), and dimension of the input vector"""
    def __init__(self, X_train):
        self.inputs = X_train 
        self.hidden = 60 
        self.outputs = 4 
        self.input_dim = 28 
        """Creates an instance of the Feed Forward network"""
        self.FeedForward = FeedForward(self.input_dim, self.hidden, self.outputs) 
        
        """Converts the input from a vector to a tensor"""        
    def input_converter(self, x) -> torch.FloatTensor:
        return torch.FloatTensor(x)    
        
    def train(self):
        
        epochs = 5
        
        """Step size"""
        learning_rate = 0.001
        
        """Instatiates the optimizer that uses the learning rate to find the optimal low point of the loss function"""
        optimizer = optim.Adam(self.FeedForward.parameters(), lr = learning_rate) 
        
        for epoch in range(0, epochs):
            
            """Overall loss initialized to zero"""
            overall_loss = 0.0 
            
            """Creates a list of index numbers for each case"""
            ip_index = [i for i in range(0, len(self.inputs))] 
            
            """Shuffles index number of cases randomly in order to obtain a random distribution of high, low and medium cases"""
            random.shuffle(ip_index) 

            for i in ip_index:
                
                """Creates input tensor"""
                x = self.input_converter(self.inputs.iloc[i,3:31])
                
                """Creates gold output label"""
                y = self.inputs.iloc[i,32]             

                """One hot encoding of gold label"""
                y_vector = torch.zeros(self.outputs)
                y_vector.scatter_(0, torch.from_numpy(np.asarray(y, dtype=np.int64)), 1)
                
                """Invokes network instance to get output log probability"""
                self.FeedForward.zero_grad()
                log_probability = self.FeedForward.forward(x)
                
                """Computes loss"""
                steps = torch.neg(log_probability).dot(y_vector)
                overall_loss += steps
                
                """Update rule"""
                steps.backward()
                optimizer.step()
            
            """Display total loss at end of each epoch"""
            print('Epoch :',epoch,'loss: ',overall_loss.item())
            
    def predict(self, case) -> int:
        
        """Initialize list for predicted classfification values"""
        y_pred=[]
        
        """Predicts threat level for each case"""
        for i in range(0,len(case)):
            
            """Converts input vector into a float tensor"""
            x = self.input_converter(case.iloc[i,3:31])  
            
            """Invokes network to predict output's log_probaility"""
            log_probability = self.FeedForward.forward(x)
            
            """Returns index number of maximum log probability from output tensor"""
            pred = torch.argmax(log_probability)
            
            """Appends predicted value to y_pred list"""
            y_pred.append(pred)
            
        return y_pred

