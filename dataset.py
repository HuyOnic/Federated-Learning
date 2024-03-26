from torch.utils.data import random_split, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
import torch 
def get_mnist(datapath: str = './data'):
    tr = Compose([ToTensor(), Normalize((0.1307,),(0.3081,))])
    trainset = MNIST(datapath, train=True, download=True, transform=tr)
    testset = MNIST(datapath, train=False, download=True, transform=tr)
    return trainset, testset

def prepare_datasets(num_partition: int, batch_size:int, val_ratio:float = 0.1):
    trainset, testset = get_mnist()

    num_sample = len(trainset)//num_partition
    partition_len = [num_sample]*num_partition
    
    
    trainsets = random_split(trainset, partition_len, torch.Generator().manual_seed(2023))
    train_loader = []
    val_loader = []
#   create validate set
    for trainset in trainsets:
        num_total = len(trainset)
        num_val = int(num_total*val_ratio)
        num_train = num_total-num_val
        for_train, for_val = random_split(trainset,(num_train,num_val),torch.Generator().manual_seed(2023))
        train_loader.append(DataLoader(for_train,batch_size=batch_size,shuffle=True))
        val_loader.append(DataLoader(for_val,batch_size=batch_size,shuffle=True))

    test_loader = DataLoader(testset,batch_size=64)
    return train_loader,val_loader,test_loader
