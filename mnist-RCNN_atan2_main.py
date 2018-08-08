from __future__ import print_function
import os
import sys
import time

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import struct
import pandas as pd
from tensorboardX import SummaryWriter

import matplotlib
from scipy.ndimage.interpolation import rotate
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import rotate
from torchvision import datasets, transforms

from model import Encoder,feature_transformer
from torch.utils.data import Dataset, DataLoader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
         nn.init.xavier_normal_(m.weight)

def get_error_per_digit(args, model,device):

    #data_root_file='/home/ubuntu/mnist-interpretable-tranformations/data'
    data_root_file='/Users/marios/Google Drive/MSc CSML/MSc Project/\
01_mnist-interpretable-tranformations/02_Autoencoder_rotation/data'
    data_loaders={digit:DataLoader (MNISTDadataset(data_root_file,digit), 
        batch_size=args.batch_size, shuffle=False) for digit in range(0,10)}

    step=5 #degrees step
    mean_error = pd.DataFrame()
    mean_abs_error=pd.DataFrame()
    error_std=pd.DataFrame()

    for digit, data_loader in data_loaders.items():
            sys.stdout.write('Processing digit {} \n'.format(digit))
            sys.stdout.flush()
            results=get_metrics(args,model,data_loader,device,step)
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
    

def get_metrics(args,model, data_loader,device, step=5):
    """ 
    Returns the average error per step of degrees in the range [-args.relative_rotation_range, args.relative_rotation_range]
    Args:
        model : pytorch Net_Reg model
        data_loader
        step (scalar) in degrees
    
    """
    #turn step to radians
    total_range=2*args.relative_rot_range #(in radians)
    step=np.pi*step/180 #(in radians)
    entries=int(total_range/step)
    model.eval()
    errors=np.zeros((entries,len(data_loader.dataset)))

    with torch.no_grad():

        start_index=0
        for batch_idx,data in enumerate(data_loader):
            
            batch_size=data.shape[0]
            angles = torch.arange(-args.relative_rot_range, args.relative_rot_range, step=step)
            target = rotate_tensor_given_angle(data.numpy(),angles.numpy())
            data=data.to(device)

            
            
            target = torch.from_numpy(target).to(device)
            
            #Get Feature vector for original and tranformed image

            f_data=model(data) #Feature vector of data
            f_targets=model(target) #Feature vector of targets

            #Compare Angles            
            f_data=f_data.view(f_data.shape[0],1,-1)
            f_data=f_data.repeat(1,entries,1)# Repeat each vector "entries" times
            f_data=f_data.view(-1,2)# rearrange vector to [entries*N,2]
            
            f_targets=f_targets.view(-1,2) # collapse 3D tensor to 2D tensor
        
            new_batch_size=f_data.shape[0] # get augmented batch_size

            f_data_y= f_data[:,1] #Extract y coordinates
            f_data_x= f_data[:,0] #Extract x coordinates

            f_targets_y= f_targets[:,1] #Extract y coordinates
            f_targets_x= f_targets[:,0] #Extract x coordinates

            theta_data=torch.atan2(f_data_y,f_data_x).numpy()*180/np.pi #Calculate absotulue angel of vector
            theta_targets=torch.atan2(f_targets_y,f_targets_x).numpy()*180/np.pi #Calculate absotulue angel of vector

            estimated_angle=theta_targets-theta_data
            estimated_angle=estimated_angle.reshape(-1,1)

            error=estimated_angle-(angles.view(-1,1).repeat(batch_size,1).numpy()*180/np.pi)

          
            #Get the tota
            for i in range(entries):
                index=np.arange(i,new_batch_size,step=entries)
                errors[i,start_index:start_index+batch_size]=error[index].reshape(-1,)

            start_index+=batch_size

            sys.stdout.write("\r%d%% complete" % ((batch_idx * 100)/len(data_loader)))
            sys.stdout.flush()
    
    mean_error=np.nanmean(errors,axis=1)
    mean_abs_error=np.nanmean(abs(errors),axis=1)
    error_std=np.nanstd(errors, axis=1, ddof=1)
   
    return mean_error, mean_abs_error, error_std




# Get overall accuracy for all models
def rotate_tensor_given_angle(input,angles):
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


def rotate_tensor(input,init_rot_range,relative_rot_range, plot=False):
    """
    Rotate the image
    Args:
        input: [N,c,h,w]    **numpy** tensor
        init_rot_range:     (scalar) the range of ground truth rotation
        relative_rot_range: (scalar) the range of relative rotations
        plot: (flag)         plot the original and rotated digits
    Returns:
        outputs1: [N,c,h,w]  input rotated by offset angle
        outputs2: [N,c,h,w]  input rotated by offset angle + relative angle [0, rot_range]
        relative angele [N,1] relative angle between outputs1 and outputs 2 in radians
    """
    #Define offest angle of input
    offset_angles=init_rot_range*np.random.rand(input.shape[0])
    offset_angles=offset_angles.astype(np.float32)

    #Define relative angle
    relative_angles=relative_rot_range*np.random.rand(input.shape[0])
    relative_angles=relative_angles.astype(np.float32)


    outputs1=[]
    outputs2=[]
    for i in range(input.shape[0]):
        output1 = rotate(input[i,...], 180*offset_angles[i]/np.pi, axes=(1,2), reshape=False)
        output2 = rotate(input[i,...], 180*(offset_angles[i]+relative_angles[i])/np.pi, axes=(1,2), reshape=False)
        outputs1.append(output1)
        outputs2.append(output2)

    outputs1=np.stack(outputs1, 0)
    outputs2=np.stack(outputs2, 0)

    if plot:
        #Create of output1 and outputs1
        N=input.shape[0]
        rows=int(np.floor(N**0.5))
        cols=N//rows
        plt.figure()
        for j in range(N):
            plt.subplot(rows,cols,j+1)
            if outputs1.shape[1]>1:
                image=outputs1[j].transpose(1,2,0)
            else:
                image=outputs1[j,0]

            plt.imshow(image, cmap='gray')
            plt.grid(False)
            plt.title(r'$\theta$={:.1f}'.format(offset_angles[j]*180/np.pi), fontsize=6)
            plt.axis('off')
        #Create new figure with rotated
        plt.figure(figsize=(7,7))
        for j in range(N):
            plt.subplot(rows,cols,j+1)
            if input.shape[1]>1:
                image=outputs2[j].transpose(1,2,0)
            else:
                image=outputs2[j,0]
            plt.imshow(image, cmap='gray')
            plt.axis('off')
            plt.title(r'$\theta$={:.1f}'.format( (offset_angles[i]+relative_angles[i])*180/np.pi), fontsize=6)
            plt.grid(False)
        plt.tight_layout()      
        plt.show()

    return outputs1, outputs2, relative_angles


def save_model(args,model):
    """
    saves a checkpoint so that model weight can later be used for inference
    Args:
    model:  pytorch model
    """

    path='./model_'+args.name
    import os
    if not os.path.exists(path):
      os.mkdir(path)
    torch.save(model.state_dict(), path+'/checkpoint.pt')



def evaluate_model(args,model, device, data_loader):
    """
    Evaluate loss in subsample of data_loader
    """
    model.eval()
    with torch.no_grad():
        for data, targets in data_loader:
            # Reshape data
            data,targets,angles = rotate_tensor(data.numpy(),args.init_rot_range, args.relative_rot_range)
            targets = torch.from_numpy(targets).to(device)
            angles = torch.from_numpy(angles).to(device)
            angles = angles.view(angles.size(0), 1)
            data = torch.from_numpy(data).to(device)

            # Forward passes
            f_data=model(data) # [N,2,1,1]
            f_targets=model(targets) #[N,2,1,1]

            #Apply rotation matrix to f_data with feature transformer
            f_data_trasformed= feature_transformer(f_data,angles,device)

            #Define loss
            loss=define_loss(args,f_data_trasformed,f_targets)
            break

    return loss.cpu()


def rotation_test(args,model, device, test_loader):
    """
    Test how well the eoncoder discrimates angles
    return the average error in degrees
    """
    model.eval()
    with torch.no_grad():
        for data,_ in test_loader:
            ## Reshape data
            data,targets,angles = rotate_tensor(data.numpy(),args.init_rot_range, args.relative_rot_range)
            data = torch.from_numpy(data).to(device)
            targets = torch.from_numpy(targets).to(device)
        
            # Forward passes
            f_data=model(data) # [N,2,1,1]
            f_targets=model(targets) #[N,2,1,1]

            f_data=f_data.view(-1,2)  # [N,2]
            f_targets=f_targets.view(-1,2) #[N,2]

            f_data_y= f_data[:,1] #Extract y coordinates
            f_data_x= f_data[:,0] #Extract x coordinates

            f_targets_y= f_targets[:,1] #Extract y coordinates
            f_targets_x= f_targets[:,0] #Extract x coordinates

            theta_data=torch.atan2(f_data_y,f_data_x).numpy()*180/np.pi #Calculate absotulue angel of vector
            theta_targets=torch.atan2(f_targets_y,f_targets_x).numpy()*180/np.pi #Calculate absotulue angel of vector

            estimated_angle=theta_targets-theta_data

            error=estimated_angle-angles*180/np.pi

            abs_mean_error=np.nanmean(abs(error))

            error_std=np.nanstd(error,ddof=1)
            break

    return abs_mean_error,error_std


def define_loss(args, x,y):
    """
    Return the loss based on the user's arguments

    Args:
        x:  [N,2,1,1]    output of encoder model
        y:  [N,2,1,1]    output of encode model
    """
    if args.loss=='forbenius':
        forb_distance=torch.nn.PairwiseDistance()
        x_polar=x.view(-1,2)
        x_polar=x/x.norm(p=2,dim=1,keepdim=True)
        y_polar=y.view(-1,2)
        y_polar=y/y.norm(p=2,dim=1,keepdim=True)
        loss=(forb_distance(x_polar,y_polar)**2).sum()

    elif args.loss=='cosine_squared':

        cosine_similarity=nn.CosineSimilarity(dim=2)

        loss=((cosine_similarity(x.view(x.size(0),1,2),y.view(y.size(0),1,2))-1.0)**2).sum()

    elif args.loss=='cosine_abs':

        cosine_similarity=nn.CosineSimilarity(dim=2)
        loss=torch.abs(cosine_similarity(x.view(x.size(0),1,2),y.view(y.size(0),1,2))-1.0).sum()

    return loss

def main():
    # Training settings
    list_of_choices=['forbenius', 'cosine_squared','cosine_abs']

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch  rotation test (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--store-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before storing training loss')
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument("--loss",dest='loss',default='forbenius',
    choices=list_of_choices, help='Decide type of loss, (forbenius) norm, difference of (cosine), (default=forbenius)')
    parser.add_argument('--init-rot-range',type=float, default=360,
                        help='Upper bound of range in degrees of initial random rotation of digits, (Default=360)')
    parser.add_argument('--relative-rot-range',type=float, default=90, metavar='theta',
                        help='Relative rotation range (-theta, theta)')
    parser.add_argument('--eval-batch-size', type=int, default=200, metavar='N',
                        help='batch-size for evaluation')
    
    args = parser.parse_args()

    #Print arguments
    for arg in vars(args):
        sys.stdout.write('{} = {} \n'.format(arg,  getattr(args, arg)))
        sys.stdout.flush()

    sys.stdout.write('Random torch seed:{}\n'.format( torch.initial_seed()))
    sys.stdout.flush()

    args.init_rot_range=args.init_rot_range*np.pi/180
    args.relative_rot_range= args.relative_rot_range*np.pi/180
    # Create save path

    path = "./output_"+args.name
    if not os.path.exists(path):
        os.makedirs(path)

    sys.stdout.write('Start training\n')
    sys.stdout.flush()

    use_cuda = torch.cuda.is_available()


    device = torch.device("cuda" if use_cuda else "cpu")

    writer = SummaryWriter(path, comment='Encoder atan2 MNIST')
    # Set up dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    train_loader_eval = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # Init model and optimizer
    model = Encoder(device).to(device)

    #Initialise weights
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Init losses log
    
    prediction_mean_error=[] #Average  rotation prediction error in degrees
    prediction_error_std=[] #Std of error for rotation prediciton
    train_loss=[]

    #Train
    n_iter=0
    for epoch in range(1, args.epochs + 1):
        sys.stdout.write('Epoch {}/{} \n '.format(epoch,args.epochs))
        sys.stdout.flush()

        for batch_idx, (data, targets) in enumerate(train_loader):
            model.train()
            # Reshape data
            data, targets, angles = rotate_tensor(data.numpy(),args.init_rot_range, args.relative_rot_range)
            data = torch.from_numpy(data).to(device)
            targets = torch.from_numpy(targets).to(device)
            angles = torch.from_numpy(angles).to(device)
            angles = angles.view(angles.size(0), 1)

            # Forward passes
            optimizer.zero_grad()
            f_data=model(data) # [N,2,1,1]
            f_targets=model(targets) #[N,2,1,1]

            #Apply rotatin matrix to f_data with feature transformer
            f_data_trasformed= feature_transformer(f_data,angles,device)

            #Define loss

            loss=define_loss(args,f_data_trasformed,f_targets)

            # Backprop
            loss.backward()
            optimizer.step()

            #Log progress
            if batch_idx % args.log_interval == 0:
                sys.stdout.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\r'
                    .format(epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss))
                sys.stdout.flush()

                writer.add_scalar('Training Loss',loss,n_iter)

            #Store training and test loss
            if batch_idx % args.store_interval==0:
                #Train Loss
                train_loss.append(evaluate_model(args,model, device, train_loader_eval))


                #Rotation loss in trainign set
                mean, std=rotation_test(args,model, device, train_loader_eval)
                prediction_mean_error.append(mean)
                writer.add_scalar('Mean test error',mean,n_iter)

                prediction_error_std.append(std)

            n_iter+=1


        save_model(args,model)



    #Save model

    #Save losses
    train_loss=np.array(train_loss)
    prediction_mean_error=np.array(prediction_mean_error)
    prediction_error_std=np.array(prediction_error_std)

    np.save(path+'/training_loss',train_loss)
    np.save(path+'/prediction_mean_error',prediction_mean_error)
    np.save(path+'/prediction_error_std',prediction_error_std)


    plot_learning_curve(args,train_loss,prediction_mean_error,prediction_error_std,path)

    #Get diagnostics per digit 
    get_error_per_digit(args, model,device)


def plot_learning_curve(args,training_loss,average_error,error_std,path):

    x_ticks=np.arange(len(training_loss))*args.store_interval*args.batch_size
    with plt.style.context('ggplot'):
        fig, (ax1,ax2)=plt.subplots(2,1,sharex=True,figsize=(5,5))
        # #Set gray background
        # ax1.set_facecolor('#E6E6E6')
        # ax2.set_facecolor('#E6E6E6')

        #Plot loss
        ax1.plot(x_ticks,training_loss,label='Training Loss',linewidth=1.25)
        loss_type=args.loss+' Loss'
        ax1.set_ylabel(loss_type,fontsize=10)
        
        ax1.legend()

        # #Grid lines
        # ax2.grid()
        # ax1.grid()

       
        line,=ax2.plot(x_ticks,average_error,label='Average Abs training error',linewidth=1.25,color='g')
        ax2.fill_between(x_ticks,average_error-error_std,average_error+error_std,
            alpha=0.2,facecolor=line.get_color(),edgecolor=line.get_color())
        ax2.set_ylabel('Degrees',fontsize=10)
        ax2.set_xlabel('Training Examples',fontsize=10)
        ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax2.legend()


        #Control colour of ticks
        ax1.tick_params(colors='gray', direction='out')
        for tick in ax1.get_xticklabels():
            tick.set_color('gray')
        for tick in ax1.get_yticklabels():
            tick.set_color('gray')

        ax2.tick_params(colors='gray', direction='out')
        for tick in ax2.get_xticklabels():
            tick.set_color('gray')
        for tick in ax2.get_yticklabels():
            tick.set_color('gray')

        fig.suptitle('Learning Curves')
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])
        fig.savefig(path+'/learning_curves')
        fig.clf()
  
if __name__ == '__main__':
    main()
