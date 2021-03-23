
## Datahandlers
from PIL import Image
from torch.utils.data import Dataset

class DataHandler_For_Arrays(Dataset):
  """
  base class for mnist / fashion_mnist
  """
  def __init__(self, X, Y, transform=None, num_classes=10):
    self.X = np.expand_dims(X, axis=1)#X[np.newaxis,...] # x[:, np.newaxis]:
    self.Y = Y
    # self.Y = torch.as_tensor(self.Y)
    # self.Y = torch.nn.functional.one_hot(self.Y, num_classes=10)
    self.transform = transform
    self.num_classes = num_classes
  
  def __getitem__(self,index):
    x,y = self.X[index], self.Y[index]

    x = x / 255.0
    if self.transform is not None:
      x = Image.fromarray(x.numpy(), model=="L")
      x = self.transform(x)
    return x,y
    
  def __len__(self):
    return len(self.X)
