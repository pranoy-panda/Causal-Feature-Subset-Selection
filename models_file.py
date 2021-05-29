from config import *

'''
* This network takes input X and returns a function of X, g_{\theta}(X) which are 
the logits for sample_concrete() function.
* In genral, we need a logit(or importance score) for each of the M x M patches. 
So output shape could be M x M, or 1 x M^2
* In case of MNIST dataset, we need a logit(or importance score) for each of the
 7 x 7 patches. So output shape could be 7 x 7, or 1 x 49. Let it be 1 x 49 for now. 
'''
class gumbel_selector(nn.Module): # this class is obviously specific to the problem statement that we have
    #takes input 1x28x28 (single channel)
    def __init__(self, k=5):
        super(gumbel_selector, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1,8,k, padding=2), nn.MaxPool2d(2), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(8,16,k, padding=2), nn.MaxPool2d(2), nn.ReLU())        
        self.c3 = nn.Conv2d(16,1,1)
    
    def forward(self, x):
        bs = x.size(0)
        
        o1 = self.c1(x)
        o2 = self.c2(o1)
        logits = self.c3(o2)
        
        return logits.view(bs,-1) #shape(bs, 49)

class BaseModel(nn.Module):
    def __init__(self,k=5):
        super(BaseModel, self).__init__()
        self.c1 = nn.Sequential(nn.Conv2d(1, 8, k, bias=True), nn.MaxPool2d(2), nn.ReLU())
        self.c2 = nn.Sequential(nn.Conv2d(8, 16, k, bias=True), nn.MaxPool2d(2), nn.ReLU())
        self.fc = nn.Linear(4*4*16, 2) #conv without padding
        
    def forward(self, x):
        bs = x.size(0)
        
        o1 = self.c1(x)
        o2 = self.c2(o1)
        out = self.fc(o2.view(bs,-1))
        return out