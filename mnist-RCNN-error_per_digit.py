from __future__ import print_function
import os
import sys
import time
from os import walk
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import struct
import pandas as pd

from model import Encoder
from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader



def main():

    parser = argparse.ArgumentParser(description='Estimate average error and std for each MNIST dataset')
    parser.add_argument('--model-name',type=str, required=True,
                        help= 'filepath of model to use')
    parser.add_argument('--output-name', type=str, required=True,
                        help='name of output files')
    parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                        help='batch-size for evaluation')

    args = parser.parse_args()

    #Load model
    path='/home/ubuntu/Saved_Models/'
    filename=os.path.join(path,args.model_name,'checkpoint.pt')

    use_cuda= torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    model=Encoder(device)
    model.load_state_dict(torch.load(filename))
    model=model.cuda()

    data_root_file='/home/ubuntu/mnist-interpretable-tranformations/data'
    data_loaders={digit:DataLoader (MNISTDadataset(data_root_file,digit), 
        batch_size=args.batch_size, shuffle=False, **kwargs) for digit in range(0,10)}

    step=5 #degrees step
    mean_error = pd.DataFrame()
    mean_abs_error=pd.DataFrame()
    error_std=pd.DataFrame()

    for digit, data_loader in data_loaders.items():
            sys.stdout.write('Processing digit {} \n'.format(digit))
            sys.stdout.flush()
            results=get_metrics(model,data_loader,device,step)
            mean_error[digit]=pd.Series(results[0])
            mean_abs_error[digit]= pd.Series(results[1])
            error_std[digit]= pd.Series(results[2])
            
    mean_error.index=mean_error.index*step
    mean_abs_error.index=mean_abs_error.index*step
    error_std.index=error_std.index*step

    mean_error.to_csv(args.output_name+'_mean_error.csv')
    mean_abs_error.to_csv(args.output_name+'_mean_abs_error.csv')
    error_std.to_csv(args.output_name+'_error_std.csv')

    ##Plottin just absolute error
    with plt.style.context('ggplot'):
        mean_abs_error.plot(figsize=(9, 8))
        plt.xlabel('Degrees')
        plt.ylabel('Average error in degrees')
        plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
                   ncol=2, shadow=True, title="Digits", fancybox=True)
        
        plt.tick_params(colors='gray', direction='out')
        plt.savefig(args.output_name+'_abs_mean_curves.png')
        plt.close()

    ##Plotting absoltue error and std
    with plt.style.context('ggplot'):
        fig =plt.figure(figsize=(9, 8))
        ax = fig.add_subplot(111)
        x=mean_abs_error.index
        for digit in mean_abs_error.columns:
            mean=mean_abs_error[digit]
            std=error_std[digit]
            line,= ax.plot(x,mean)
            ax.fill_between(x,mean-std,mean+std,alpha=0.2,facecolor=line.get_color(),edgecolor=line.get_color())
        
        ax.set_xlabel('Degrees')
        ax.set_ylabel('Average error in degrees')
        ax.legend(loc="upper left", bbox_to_anchor=[0, 1],
                   ncol=2, shadow=True, title="Digits", fancybox=True)
        ax.tick_params(colors='gray', direction='out')
        fig.savefig(args.output_name+'_mean_&_std_curves.png')
        fig.clf()

def read_idx(filename):
    import struct
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

class MNISTDadataset(Dataset):
    def __init__(self,root_dir, digit,transform=None):
        """
        Args:
            digit(int):        MNIST digit
            root_dir (string): Directory where the ubyte lies
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        file_path =os.path.join(root_dir,'train-images-'+str(digit)+'-ubyte')
        self.data = read_idx(file_path)/255
        
        self.data = torch.Tensor(self.data)
        self.data =  (self.data).unsqueeze(1)
        
        self.transform=transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        sample=self.data[idx]
        

        if self.transform:
            sample = self.transform(sample)

        return sample


# Get overall accuracy for all models
def rotate_tensor(input,angles):
    """
    Rotates each image by angles and concatinates them in one tenosr
    Args:
        input: [N,c,h,w] **numpy** tensor
        angles: [D,1]
    Returns:
        output [N*D,c,h,w] **numpy** tensor
    """
    outputs = []
    for i in range(input.shape[0]):
        for angle in angles:
            output = rotate(input[i,...], 180*angle/np.pi, axes=(1,2), reshape=False)
            outputs.append(output)
    return np.stack(outputs, 0)
    

def get_metrics(model, data_loader,device, step=5):
    """ 
    Returns the average error per step of degrees in the range [0,np.pi]
    Args:
        model : pytorch Net_Reg model
        data_loader
        step (scalar) in degrees
    
    """
    #turn step to radians
    step=np.pi*step/180
    entries=int(np.pi/step)
    model.eval()
    errors=np.zeros((entries,len(data_loader.dataset)))
    
    
    with torch.no_grad():

        start_index=0
        for batch_idx,data in enumerate(data_loader):
            
            batch_size=data.shape[0]
            angles = torch.arange(0, np.pi, step=step)
            target = rotate_tensor(data.numpy(),angles.numpy())
            data=data.to(device)

            
            
            target = torch.from_numpy(target).to(device)
            
            #Get Feature vector for original and tranformed image
            cosine_similarity=nn.CosineSimilarity(dim=2)

            x=model(data) #Feature vector of data
            y=model(target) #Feature vector of targets

            #Compare Angles            
            x=x.view(x.shape[0],1,-1)
            x=x.repeat(1,entries,1)# Repeat each vector "entries" times
            x=x.view(-1,1,2)# rearrange vector
            
            y=y.view(y.shape[0],1,2) # collapse 3D tensor to 2D tensor
            
            ndims=x.shape[1]        # get dimensionality of feature space
            new_batch_size=x.shape[0]   # get augmented batch_size
            
            #Loop every 2 dimensions
            
            sys.stdout.write("\r%d%% complete" % ((batch_idx * 100)/len(data_loader)))
            sys.stdout.flush()

            predicted_cosine=cosine_similarity(x,y)
            predicted_angle=(torch.acos(predicted_cosine)).cpu()  
            error=predicted_angle.numpy()-(angles.view(-1,1).repeat(batch_size,1).numpy()*180/np.pi)
            #Get the tota
            for i in range(entries):
                index=np.arange(i,new_batch_size,step=entries)
                errors[i,start_index:start_index+batch_size]=error[index].reshape(-1,)

            start_index+=batch_size
    
    mean_error=errors.mean(axis=1)
    mean_abs_error=(abs(errors)).mean(axis=1)
    error_std=errors.std(axis=1, ddof=1)
   
    return mean_error, mean_abs_error, error_std


if __name__ == '__main__':
    main()