import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import torch.utils.data as data 

from model import PlantXMamba
from plan_mamba.dataset import RiceDataset
from plan_mamba.utils import get_transform
from plan_mamba.train import train_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(args):
    dataset_train = RiceDataset(image_paths=args.train, transform=get_transform('train'))
    dataloader_train = data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=int(args.threads))
    
    dataset_test = RiceDataset(image_paths=args.test, transform=get_transform('test'))
    dataloader_test = data.DataLoader(dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=int(args.threads))
    
    return dataloader_train, dataloader_test

if __name__ == "__main__":
    parsers = argparse.ArgumentParser(description='Base Fine-Grained SBIR model')
    parsers.add_argument('--train', type=str, default='/kaggle/input/rice-dataset/Rice_dataset/train_smp')
    parsers.add_argument('--test', type=str, default='/kaggle/input/rice-dataset/Rice_dataset/pred_img')
    
    args = parsers.parse_args()
    dataloader_train, dataloader_test = get_dataloader(args=args)
    
    num_classes = 5
    model = PlantXMamba(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    history = train_model(model=model, train_loader=dataloader_train, val_loader=dataloader_test,
                criterion=criterion, optimizer=optimizer, num_epochs=10,
                device=device)