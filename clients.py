from collections import OrderedDict
import torch 
from typing import Dict
import flwr as fl
from flwr.common import NDArray, Scalar
from model import Net, train, test
class FlowerClient(fl.client.NumPyClient):
    def __init__(self,trainloader, valoader, num_classes:int):
        super().__init__()
        self.trainloader = trainloader
        self.valoader = valoader
        self.model = Net(num_classes)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k:torch.Tensor(v) for k,v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def get_parameters(self, config: Dict[str,Scalar]):
        return [val.cpu().numpy() for _,val in self.model.state_dict().items()]
    
        
    
    def fit(self,parameters, config):
         self.set_parameters(parameters)
         lr = config['lr']
         momentum = config['momentum']
         epochs = config['local_epochs']

         optim = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)

         train(self.model, self.trainloader, optim, epochs, self.device)

         return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters: NDArray, config: Dict[str,Scalar]):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.valoader, self.device)
        return float(loss), len(self.valoader), {'accuracy':accuracy}
def generate_client_fn(trainloaders, valoader, numclass):
    def client_fn(cid:str):
        return FlowerClient(trainloader=trainloaders[int(cid)],valoader=valoader[int(cid)],num_classes=numclass).to_client()
    return client_fn
        


