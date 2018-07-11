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

from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import rotate
from torchvision import datasets, transforms

from model import Encoder,feature_transformer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
         nn.init.xavier_normal_(m.weight)


def rotate_tensor(input,rot_range=np.pi, plot=False):
    """
    Rotate the image
    Args:
        input: [N,c,h,w] **numpy** tensor
        rot_range: (scalar) the range of relative rotations
        plot: (flag)    plot the original and rotated digits
    Returns:
        outputs1: [N,c,h,w]  input rotated by offset angle
        outputs2: [N,c,h,w]  input rotated by offset angle + relative angle [0, rot_range]
        relative angel [N,1] relative angle between outputs1 and outputs 2 in radians
    """
    #Define offest angle of input
    offset_angles=np.pi*np.random.rand(input.shape[0])
    offset_angles=offset_angles.astype(np.float32)

    #Define relative angle
    relative_angles=rot_range*np.random.rand(input.shape[0])
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
            data,targets,angles = rotate_tensor(data.numpy()) #angles are relative between data and target (targets =  R * data)
            data = torch.from_numpy(data).to(device)
            targets = torch.from_numpy(targets).to(device)
            angles = torch.from_numpy(angles).to(device)
            angles = angles.view(angles.size(0), 1)

            # Forward passes
            f_data=model(data) # [N,2,1,1]
            f_targets=model(targets) #[N,2,1,1]

            #Apply rotation matrix to f_data with feature transformer
            f_data_trasformed= feature_transformer(f_data,angles,device)

            import ipdb; ipdb.set_trace()

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
        for data, targets in test_loader:
            ## Reshape data
            data,targets,angles = rotate_tensor(data.numpy()) #angles are relative between data and target (targets =  R * data)
            data = torch.from_numpy(data).to(device)
            targets = torch.from_numpy(targets).to(device)
            angles = torch.from_numpy(angles).to(device)
            angles = angles.view(angles.size(0), 1)

            # Forward passes
            f_data=model(data) # [N,2,1,1]
            f_targets=model(targets) #[N,2,1,1]

            #Get cosine similarity
            f_data=f_data.view(f_data.size(0),1,2)
            f_targets=f_targets.view(f_targets.size(0),1,2)

            cosine_similarity=nn.CosineSimilarity(dim=2)

            predicted_cosine=cosine_similarity(f_data,f_targets)

            predicted_angle=(torch.acos(predicted_cosine)).cpu()

            error=abs(predicted_angle-angles).mean()*180/np.pi
            break

    return error


def define_loss(args, x,y):
    """
    Return the loss based on the user's arguments

    Args:
        x:  [N,2,1,1]    output of encoder model
        y:  [N,2,1,1]    output of encode model
    """

    if args.loss=='frobenius':
        forb_distance=torch.nn.PairwiseDistance()
        loss=(forb_distance(x.view(-1,2),y.view(-1,2))**2).sum()

    elif args.loss=='cosine_squared':

        cosine_similarity=nn.CosineSimilarity(dim=2)
        loss=((cosine_similarity(x.view(x.size(0),1,2),y.view(y.size(0),1,2))-1.0)**2).sum()

    elif args.loss=='cosine_abs':

        cosine_similarity=nn.CosineSimilarity(dim=2)
        loss=torch.abs(cosine_similarity(x.view(x.size(0),1,2),y.view(y.size(0),1,2))-1.0).sum()

    return loss

def main():
    # Training settings
    list_of_choices=['frobenius', 'cosine_squared','cosine_abs']

    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for reconstruction testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--store-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before storing training loss')
    parser.add_argument('--name', type=str, default='',
                        help='name of the run that is added to the output directory')
    parser.add_argument("--loss",dest='loss',default='frobenius',
    choices=list_of_choices, help='Decide type of loss, (frobenius) norm, difference of (cosine), (default=forbenius)')

    
    args = parser.parse_args()

    # Create save path
    path = "./output_"+args.name
    if not os.path.exists(path):
        os.makedirs(path)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Set up dataloaders
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    train_loader_eval = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **{})

    # Init model and optimizer
    model = Encoder(device).to(device)
    print(model)
  
    #Initialise weights
    model.apply(weights_init)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #Init losses log
    rotation_test_loss=[]
    train_loss=[]
    test_loss=[]

    # Where the magic happens
    for epoch in range(1, args.epochs + 1):
        for batch_idx, (data, targets) in enumerate(train_loader):
            model.train()
            # Reshape data
            data,targets,angles = rotate_tensor(data.numpy())
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

            #Store training and test loss
            if batch_idx % args.store_interval==0:
                #Train Lossq
                train_loss.append(evaluate_model(args,model, device, train_loader_eval))

                #Test Loss
                test_loss.append(evaluate_model(args,model, device, test_loader))

                #Rotation loss in trainign set
                rotation_test_loss.append(rotation_test(args,model, device, train_loader_eval))


    #Save model
    save_model(args,model)
    #Save losses
    train_loss=np.array(train_loss)
    test_loss=np.array(test_loss)
    rotation_test_loss=np.array(rotation_test_loss)

    np.save(path+'/training_loss',train_loss)
    np.save(path+'/test_loss',test_loss)
    np.save(path+'/rotation_test_loss',rotation_test_loss)

    plot_learning_curve(args,train_loss,test_loss,rotation_test_loss,path)


def plot_learning_curve(args,training_loss,test_loss,rotation_test_loss,path):

    x_ticks=np.arange(len(training_loss))*args.store_interval*args.batch_size

    plt.subplot(121)
    plt.plot(x_ticks,training_loss,label='Training Loss')
    plt.plot(x_ticks,test_loss,label='Test Loss')
    loss_type=args.loss+' Loss'
    plt.ylabel(loss_type)
    plt.xlabel('Training Examples')
    plt.title('Learning Curves')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.legend()

    plt.subplot(122)
    plt.plot(x_ticks,rotation_test_loss,label='Test Cosine Loss')
    plt.title('Average error in degrees over {} trainign examples'.format(args.test_batch_size))
    plt.xlabel('Training Examples')
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    plt.ylabel('Degrees')
    plt.tight_layout()

    path = path+"/learning_curves"
    plt.savefig(path)
    plt.close()
  
if __name__ == '__main__':
    main()