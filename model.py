import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
  def __init__(self, n_class=0):
    super().__init__()
    self.batch_norm1 = nn.BatchNorm3d(3)
    self.conv1 = nn.Conv3d(3, 16, 3, padding = (1, 1, 1), stride=(1, 1, 1))
    self.pool1 = nn.MaxPool3d((3, 3, 8), stride=(2, 2, 8), padding=(1, 1, 0))
    #(16, 32, 32, 2)
    
    self.batch_norm2 = nn.BatchNorm3d(16)
    self.conv2 = nn.Conv3d(16, 32, 3, padding = (1, 1, 1), stride=(1, 1, 1))
    self.pool2 = nn.MaxPool3d((3, 3, 2), stride=(2, 2, 1), padding=(1, 1, 0))
    #(32, 16, 16, 1)

    self.batch_norm3 = nn.BatchNorm3d(32)
    self.conv3 = nn.Conv3d(32, 64, 3, padding = (1, 1, 1), stride=(1, 1, 1))
    self.pool3 = nn.MaxPool3d((3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
    #(32, 16, 16, 1)

    self.batch_norm4 = nn.BatchNorm3d(64)
    self.conv4 = nn.Conv3d(64, 128, 3, padding = (1, 1, 1), stride=(1, 1, 1))
    self.pool4 = nn.MaxPool3d((3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
    #(64, 8, 8, 1)
    self.relu = nn.ReLU()

    self.n_class = n_class
    if self.n_class != 0:
      self.l1 = nn.Linear(128, n_class)

  def forward(self, x):
    x = self.batch_norm1(x)
    x = self.conv1(x)
    x = self.relu(x)
    x = self.pool1(x)
    #print(x.size())
   
    x = self.batch_norm2(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.pool2(x)
    #print(x.size())

    x = self.batch_norm3(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.pool3(x)
    #print(x.size())

    x = self.batch_norm4(x)
    x = self.conv4(x)
    x = self.relu(x)
    x = self.pool4(x)
    #print(x.size())
    
    #m = nn.AvgPool3d((8, 8, 1), stride=(1, 1, 1))
    #print(m(x).size())
    # global pooling
    x = F.adaptive_avg_pool3d(x, (1, 1, 1))
    x = x.view(x.shape[0], -1)
    if self.n_class != 0: 
      x = self.l1(x)

    return x

def test():
    model = SimpleModel(n_class=10)
    input = torch.randn(1, 3, 224, 224, 16)
    output = model(input)
    #c = nn.Flatten(start_dim=1)(output)
    c = output.view(output.shape[0], -1)
    print(c.size())
    #print(model(input).size())
    #print(model)
    #for parameter in model.parameters():
    #    print(parameter.size())

if __name__ == "__main__":
    test()