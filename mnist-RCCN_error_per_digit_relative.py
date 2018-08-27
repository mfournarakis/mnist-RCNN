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

from model import Encoder, Encoder_Tanh
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
    parser.add_argument('--rotation-range', type=float, default=120, metavar='theta',
                        help='rotation range for evaluation [-theta, +theta)')
    parser.add_argument('--step', type=float, default=10, metavar='theta',
                        help='rotation step in degrees for evaluation of curves')
    parser.add_argument('--tanh', action= 'store_true', default=False, 
                        help='Model with or without tanh (Default=False')

    args = parser.parse_args()

    filename=os.path.join(args.model_name,'checkpoint.pt')

    use_cuda= torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.tanh: 
        model=Encoder_Tanh(device)
        model.load_state_dict(torch.load(filename))
        model=model.cuda()

        sys.stdout.write("model WITH tanh nonlinearity")
        sys.stdout.flush()
    else:
        model=Encoder(device)
        model.load_state_dict(torch.load(filename))
        model=model.cuda()

        sys.stdout.write("model WITHOUT tanh nonlinearity")
        sys.stdout.flush()

    data_root_file='../data'
    data_loaders={digit:DataLoader (MNISTDadataset(data_root_file,digit), 
        batch_size=args.batch_size, shuffle=False, **kwargs) for digit in range(0,10)}

    mean_error=pd.DataFrame()
    error_std=pd.DataFrame()


    for digit, data_loader in data_loaders.items():
            sys.stdout.write('Processing digit {} \n'.format(digit))
            sys.stdout.flush()
            mean, std=get_metrics(model,data_loader,device,0,args.rotation_range,args.step)
            mean_error[digit]= mean
            error_std[digit]= std

    mean_error.index=np.arange(-args.rotation_range,+args.rotation_range+args.step, args.step)
    error_std.index=np.arange(-args.rotation_range,+args.rotation_range+args.step, args.step)

    mean_error.to_csv(args.output_name+'_train_mean.csv')
    error_std.to_csv(args.output_name+'train_std.csv')
   
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
            output = rotate(input[i,...], angle, axes=(1,2), reshape=False)
            outputs.append(output)
    return np.stack(outputs, 0)


def convert_to_convetion(input):
    """
    Coverts all anlges to convecntion used by atan2
    """

    input[input<180]=input[input<180]+360
    input[input>180]=input[input>180]-360
    
    return input
    

def get_metrics(model, data_loader,device,starting_angle,rot_range,step):
    """ 
    Returns the average error per step of degrees in the range [0,np.pi]
    Args:
        model : pytorch Net_Reg model
        data_loader
        step (scalar) in degrees
        rotation range (scalar) in degrees
    
    """
    #Number of entries
    entries=len(np.arange(-rot_range, +rot_range+step,  step=step))

    #turn step to radians

    model.eval()
    errors=np.zeros((entries,len(data_loader.dataset)))
    start_index=0
    
    with torch.no_grad():

        start_index=0
        for batch_idx,data in enumerate(data_loader):
            
            batch_size=data.shape[0]
            angles = np.arange(-rot_range+starting_angle, +rot_range+starting_angle+step,  step=step)
            relative_angles=np.arange(-rot_range, +rot_range+step,  step=step)
        
            #Convert angels to [0,360]
            target = rotate_tensor(data.numpy(),angles)
            
            data  = rotate_tensor(data.numpy(),[starting_angle])
        
            
            data= torch.from_numpy(data).to(device)
            target = torch.from_numpy(target).to(device)

            
            # Forward passes
            f_data=model(data) # [N,2,1,1]
            f_targets=model(target) #[N,2,1,1]

            #Repeat original data
            f_data=f_data.view(-1,1,2)  # [N,1,2]
            f_data=f_data.repeat(1,entries,1) # [N,entries,2]
            f_data=f_data.view(-1,2) # [N,2*entries]

            new_batch_size=f_data.shape[0] 

            f_targets=f_targets.view(-1,2) #[N,2]

            f_data_y= f_data[:,1] #Extract y coordinates
            f_data_x= f_data[:,0] #Extract x coordinates

            f_targets_y= f_targets[:,1] #Extract y coordinates
            f_targets_x= f_targets[:,0] #Extract x coordinates

            theta_data=torch.atan2(f_data_y,f_data_x).cpu().numpy()*180/np.pi #Calculate absotulue angel of vector
            theta_targets=torch.atan2(f_targets_y,f_targets_x).cpu().numpy()*180/np.pi #Calculate absotulue angel of vector

            estimated_angle=theta_targets-theta_data
            
            estimated_angle=convert_to_convetion(estimated_angle)

            error=estimated_angle-np.tile(relative_angles,batch_size)

            for i in range(entries):
                index=np.arange(i,new_batch_size,step=entries)
                errors[i,start_index:start_index+batch_size]=error[index]
            
            start_index+=data.shape[0]

            sys.stdout.write("\r%d%% complete" % ((batch_idx * 100)/len(data_loader)))
            sys.stdout.flush()

    
    mean_abs_error=np.nanmean((abs(errors)),axis=1)
    error_std=np.nanstd(errors, axis=1)
   
    return mean_abs_error, error_std


if __name__ == '__main__':
    main()