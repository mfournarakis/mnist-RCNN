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
    parser.add_argument('--rotation-range', type=float, default=120, metavar='theta',
                        help='rotation range for evaluation [-theta, +theta)')
    parser.add_argument('--step', type=float, default=10, metavar='theta',
                        help='rotation step in degrees for evaluation of curves')

    args = parser.parse_args()

    #Load model
    # path='./../Saved_Models/mnist-RNCC/'
    path='/home/ubuntu/Saved_Models/'
    filename=os.path.join(path,args.model_name,'checkpoint.pt')

    use_cuda= torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    model=Encoder(device)
    model.load_state_dict(torch.load(filename))
    model=model.cuda()

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=False, **{})

    mean_abs_train_error=pd.DataFrame()
    train_error_std=pd.DataFrame()

    mean_abs_test_error=pd.DataFrame()
    test_error_std=pd.DataFrame()

    starting_agles=[0 ,90 ,180, 270]

    for starting_angle in starting_agles:
            mean, std=get_metrics(model,train_loader,device,starting_angle,args.rotation_range,args.step)
            mean_abs_train_error[starting_angle]= pd.Series(mean)
            train_error_std[starting_angle]= pd.Series(std)

            mean, std=get_metrics(model,test_loader,device,starting_angle,args.rotation_range,args.step)
            mean_abs_test_error[starting_angle]= pd.Series(mean)
            test_error_std[starting_angle]= pd.Series(std)


    mean_abs_train_error.index=np.arange(-args.rotation_range,+args.rotation_range+args.step, args.step)
    train_error_std.index=np.arange(-args.rotation_range,+args.rotation_range+args.step, args.step)
    mean_abs_test_error.index=np.arange(-args.rotation_range,+args.rotation_range+args.step, args.step)
    test_error_std.index=np.arange(-args.rotation_range,+args.rotation_range+args.step, args.step)

    mean_abs_train_error.to_csv(args.output_name+'_train_mean.csv')
    train_error_std.to_csv(args.output_name+'train_std.csv')
    mean_abs_test_error.to_csv(args.output_name+'test_mean.csv')
    test_error_std.to_csv(args.output_name+'test_std.csv')


    # ##Plottin just absolute error
    # with plt.style.context('ggplot'):
    #     mean_abs_error.plot(figsize=(9, 8))
    #     plt.xlabel('Degrees')
    #     plt.ylabel('Average error in degrees')
    #     plt.legend(loc="upper left", bbox_to_anchor=[0, 1],
    #                ncol=2, shadow=True, title="Digits", fancybox=True)
        
    #     plt.tick_params(colors='gray', direction='out')
    #     plt.savefig(args.output_name+'_abs_mean_curves.png')
    #     plt.close()

    # ##Plotting absoltue error and std
    # with plt.style.context('ggplot'):
    #     fig =plt.figure(figsize=(9, 8))
    #     ax = fig.add_subplot(111)
    #     x=mean_abs_error.index
    #     for digit in mean_abs_error.columns:
    #         mean=mean_abs_error[digit]
    #         std=error_std[digit]
    #         line,= ax.plot(x,mean)
    #         ax.fill_between(x,mean-std,mean+std,alpha=0.2,facecolor=line.get_color(),edgecolor=line.get_color())
        
    #     ax.set_xlabel('Degrees')
    #     ax.set_ylabel('Average error in degrees')
    #     ax.legend(loc="upper left", bbox_to_anchor=[0, 1],
    #                ncol=2, shadow=True, title="Digits", fancybox=True)
    #     ax.tick_params(colors='gray', direction='out')
    #     fig.savefig(args.output_name+'_mean_&_std_curves.png')
    #     fig.clf()

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
        for batch_idx, (data,_) in enumerate(data_loader):
            
            batch_size=data.shape[0]
            angles = convert_to_convetion(np.arange(-rot_range+starting_angle, +rot_range+starting_angle+step,  step=step))
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

            error=estimated_angle-np.tile(angles,batch_size)

            for i in range(entries):
                index=np.arange(i,new_batch_size,step=entries)
                errors[i,start_index:start_index+batch_size]=error[index]
            
            start_index+=data.shape[0]

            sys.stdout.write("\r%d%% complete" % ((batch_idx * 100)/len(data_loader)))
            sys.stdout.flush()

            break
    
    mean_abs_error=np.nanmean((abs(errors)),axis=1)
    error_std=np.nanstd(errors, axis=1)
   
    return mean_abs_error, error_std


if __name__ == '__main__':
    main()