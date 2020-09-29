import torch
import torch.nn as nn
import torch.nn.functional as F

#simple neural classifier
class Net(nn.Module):
   def __init__(self, n1, n2, num_features, num_classes):
       super(Net, self).__init__()
       self.layer1 = nn.Linear(num_features, n1) #rethink structure
       self.layer2 = nn.Linear(n1, n2)
       self.layer3 = nn.Linear(n2, num_classes)

   def forward(self, x):
       x = self.layer1(x) #can add non-linearity here to
       x = F.relu(self.layer2(x)) #maybe add a different non-linearity
       x = self.layer3(x)
       return x

   def predict(self, x):
      self.eval()
      with torch.no_grad():
         x = self.forward(x)
         x = F.softmax(x, dim = 1)
         prob = torch.mean(x, dim = 0)
         arg_max = torch.argmax(prob)
      return (arg_max, prob[arg_max], x)
