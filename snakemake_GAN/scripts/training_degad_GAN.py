import monai
import shutil
from monai.transforms import (
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    Spacingd,
    RandWeightedCrop,
    RandRotate,
    RandFlip,
    Rand3DElasticd,
    Rand3DElastic,
    RandRotated,
    LoadImage,
    EnsureChannelFirstd,
    Orientationd,
    EnsureChannelFirst,
    ScaleIntensityd,
    RandFlip,
    ToTensor,
    SpatialPadd,
    ToTensord,
    ScaleIntensity,
    RandFlipd)
import nibabel
import shutil
import tqdm
from torchmetrics import MeanSquaredError
import time
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, Dataset ,nifti_saver, PatchDataset, PersistentDataset, SmartCacheDataset, ThreadDataLoader
import torch
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import nibabel
from glob import glob
from monai.networks.blocks import Convolution
from monai.networks.nets import Discriminator, Generator
from monai.utils import progress_bar
import torch.nn as nn
import torchmetrics 
from pytorchtools import EarlyStopping
import numpy 
import torchvision.transforms as transforms
import random
from functools import reduce
from operator import mul
from torch.utils.data import DataLoader

from torchmetrics import StructuralSimilarityIndexMeasure as SSIM
import argparse

def train_model(input_files, patch_size,batch_size, lr,filter_num_g,filter_num_d,depth_g,num_conv_g,num_conv_d,train_steps_d):
    print(f"Training degad GAN with input files: {input_files[0]} {input_files[1]}\npatch_size: {patch_size}\nbatch size: {batch_size}\nlearning rate: {lr}\nnumber of initial filters (generator): {filter_num_g}\nnumber of initial filters (discriminator): {filter_num_d}\nGenerator depth: {depth_g}\nnumber of convolutions per block (generator): {num_conv_g}\nnumber of convolutions per block (discriminator): {num_conv_d}\nnumber of discriminator training steps per generator step:{train_steps_d}")


    output_dir = f"/project/6050199/akhanf/cfmm-bids/data/Lau/degad/snakemake/snakemake_GAN/output/patch-{patch_size}_batch-{batch_size}_LR-{lr}_filterG-{filter_num_g}_filterD-{filter_num_d}_depthG-{depth_g}_convsG-{num_conv_g}_convsD-{num_conv_d}_stepsD-{train_steps_d}/"


    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    fname_tr=input_files[0]# training file
    radius_actual = [int(patch_size/2-1)]*3 # getting c3d patch radius ie. if 32 ^3 patch size, it is 15
    patch_radius= numpy.array(radius_actual) # Patch dimensions
    dims = 1+2*patch_radius # numpyt
    dims_tuple = (patch_size,)*3
    k = 2  # Number of channels
    bps = (4 * k * numpy.prod(dims)) # Bytes per sample
    np_tr = os.path.getsize(fname_tr) // bps  # Number of samples
    arr_shape_tr= (int(np_tr),dims[0],dims[1],dims[2], k)
    arr_train = numpy.memmap(fname_tr,'float32','r+',shape=arr_shape_tr)
    
    fname_va=input_files[1] # validation file   
    np_va = os.path.getsize(fname_va) // bps      # Number of samples
    arr_shape_va= (int(np_va),dims[0],dims[1],dims[2], k)
    arr_val= numpy.memmap(fname_va,'float32','r+',shape=arr_shape_va)

    arr_train = numpy.swapaxes(arr_train,1,4)
    arr_val = numpy.swapaxes(arr_val,1,4)

    train_size=int(arr_train.shape[0])
    val_size=int(arr_val.shape[0])
    arr_train_image = arr_train[0:train_size,0,:,:,:].reshape(train_size,1,arr_train.shape[2],arr_train.shape[3],arr_train.shape[4])
    arr_train_label = arr_train[0:train_size,1,:,:,:].reshape(train_size,1,arr_train.shape[2],arr_train.shape[3],arr_train.shape[4])

    arr_val_image = arr_val[0:val_size,0,:,:,:].reshape(val_size,1, arr_val.shape[2],arr_val.shape[3],arr_val.shape[4])
    arr_val_label = arr_val[0:val_size,1,:,:,:].reshape(val_size,1, arr_val.shape[2],arr_val.shape[3],arr_val.shape[4])
    arr_train_dict= [{"image": gad_name, "label": nongad_name} for gad_name, nongad_name in zip(arr_train_image,arr_train_label)]
    arr_val_dict= [{"image": gad_name, "label": nongad_name} for gad_name, nongad_name in zip(arr_val_image,arr_val_label)]
    
    train_transforms = Compose([SpatialPadd(keys = ("image","label"), spatial_size = dims_tuple), Rand3DElasticd(keys = ("image","label"), sigma_range = (0.5,1), magnitude_range = (0.1, 0.4), prob=0.4, shear_range=(0.1, -0.05, 0.0, 0.0, 0.0, 0.0), scale_range=0.5, padding_mode= "zeros"),
              RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=1),RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=0),RandFlipd(keys = ("image","label"), prob = 0.5, spatial_axis=2)])
    val_transforms = Compose([SpatialPadd(keys = ("image","label"),spatial_size = dims_tuple)])
    train_patches_dataset = CacheDataset(data=arr_train_dict ,transform = train_transforms, cache_rate =0.25, copy_cache=False, progress=True) # dataset with cache mechanism that can load data and cache deterministic transforms’ result during training.
    validate_patches_dataset = CacheDataset(data=arr_val_dict ,transform = val_transforms, cache_rate = 0.25, copy_cache=False,progress=True)

    """
    
    filter = filter_num_g
    cnn_depth = depth_g
    layer_per_block = num_conv_g
    """
    
    bottleneck = depth_g # set num convs in bottleneck of generator equal to depth of generator
    
    channels = ()
    for i in range(depth_g):
        channels += num_conv_g*(filter_num_g,)
        filter_num_g*=2
    channels+=  bottleneck*(filter_num_g,) 

    strides = ()
    for i in range(depth_g):
        strides += (2,) + (1,)*(num_conv_g -1)
    strides += (bottleneck-1) * (1,)
    
    gen = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=channels, 
        strides=strides,
        dropout=0.2,
        norm='BATCH'
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    gen.apply(monai.networks.normal_init)
    gen_model = gen.to(device)
    disc = GANDiscriminator(filter_num_d)
    disc.apply(monai.networks.normal_init)
    disc_model = disc.to(device)


    import time
    learning_rate = lr
    betas = (0.5, 0.999)
    gen_opt = torch.optim.Adam(gen_model.parameters(), lr = learning_rate, betas=betas)
    disc_opt = torch.optim.Adam(disc_model.parameters(), lr = learning_rate, betas=betas)

    epoch_loss_values = [float('inf')] # list of generator  loss calculated at the end of each epoch
    disc_loss_values = [float('inf')] # list of discriminator loss values calculated at end of each epoch
    disc_train_steps = train_steps_d# number of times to loop thru discriminator for each batch

    gen_training_steps = int(np_tr / batch_size) # batch_size is a tunable param
    disc_training_steps = disc_train_steps * gen_training_steps # number of validation steps per epoch

    max_epochs = 100
    start = time.time()

    train_loader = DataLoader(train_patches_dataset, batch_size=batch_size, shuffle=True, num_workers=32)

    for epoch in range(max_epochs):
        gen_model.train()
        disc_model.train()# setting models to training mode
        epoch_loss = 0 # initializing epoch loss for generator
        disc_epoch_loss = 0 # initializing  epoch loss for discriminator
        progress_bar(
                index = epoch +1,
                count = max_epochs, 
                desc = f"epoch {epoch + 1}, avg gen loss: {epoch_loss_values[-1]:.4f}, avg disc loss: {disc_loss_values[-1]:.4f}",
            )
        for i,train_batch in enumerate(train_loader):# iterating through dataloader

            gad_images =train_batch["image"].cuda()
            nongad_images = train_batch["label"].cuda() 
            gen_opt.zero_grad()
            degad_images = gen_model(gad_images) # feeding CNN with gad images
            #disc_real_pred = disc_model(torch.cat([gad_images, nongad_images], dim=1))
            disc_fake_pred = disc_model(torch.cat([gad_images, degad_images], dim=1)) # getting disc losses when fed fake images
            gen_loss = GeneratorLoss(nongad_images, degad_images, disc_fake_pred,device) # getting generator losses
            gen_loss.backward()# computes gradient(derivative) of current tensor, automatically frees part of greaph that creates loss
            gen_opt.step() # updates parameters to minimize loss
            epoch_loss += gen_loss.item() # adding generator loss for this batch to total gen loss for this epoch
            for _ in range(disc_train_steps):
                gad_images = gad_images.clone().detach() # 
                nongad_images = nongad_images.clone().detach()#need to recall it for each iteration to avoid error message of backpropagation through a graph a second time after gradients have been freed
                degad_images = gen_model(gad_images) # feeding CNN with gad images
                disc_opt.zero_grad() # resetting gradient for discrminator to 0
                disc_real_pred = disc_model(torch.cat([gad_images, nongad_images], dim=1))
                disc_fake_pred = disc_model(torch.cat([gad_images, degad_images], dim=1)) 
                disc_loss = DiscriminatorLoss(disc_real_pred,disc_fake_pred,device)
                disc_loss.backward() #initializes back propagation to compute gradient of current tensors 
                disc_opt.step() # updates parameters to minimize loss
                disc_epoch_loss += disc_loss.item() # taking sum of disc loss for the number of steps for this batch
        epoch_loss = epoch_loss / gen_training_steps # epoch loss is the total loss by the end of that epoch divided by the number of steps
        epoch_loss_values.append(epoch_loss) #updates the loss value for that epoch
        disc_epoch_loss= disc_epoch_loss / disc_training_steps# average disc epoch loss is the total loss divided by the number of discriminator steps
        disc_loss_values.append(disc_epoch_loss) # avg disc loss is the total loss divided by the total disc steps in the epoch
    end = time.time()
    time = end - start




    with open (f'{output_dir}/model_stats.txt', 'w') as file:  
        file.write(f'Training time: {time}\n') 
        file.write(f'Number of trainable parameters: {trainable_params}\n')
        file.write(f'generator loss: {epoch_loss_values[-1]} discriminator loss: {disc_loss_values[-1]}')



    torch.save(gen_model.state_dict(), f"{output_dir}/trained_generator.pt")
    torch.save(disc_model.state_dict(), f"{output_dir}/trained_discriminator.pt")


    plt.figure(figsize=(12,5))
    plt.plot(list(range(len(epoch_loss_values))), epoch_loss_values, label="Generator Loss")
    plt.plot(list(range(len(disc_loss_values))), disc_loss_values , label="Discriminator Loss")
    plt.grid(True, "both", "both")
    plt.legend()
    plt.savefig(f'{output_dir}/lossfunction.png')


    gen_model.load_state_dict(torch.load(f'{output_dir}/trained_generator.pt'))
    gen_model.eval()

    # running inference 
    test_gad_t1= sorted(glob('/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/sub-P*/*rescaled_T1w.nii.gz'))[-1:] #TODO: MOVE PASSINGDATASET INSIDE SNAKEMAKE
    # gad images who's corresponding nongad images underwent a rigid transform
    test_nongad_t1= sorted(glob('/project/6050199/akhanf/cfmm-bids/data/Lau/degad/derivatives/passing_dataset/sub-P*/*nongad_normalized_fcm.nii.gz'))[-1:] # nongad images which underwent a rigid transform and underwent fcm normalization
    test_files = [{"image": gad_name, "label": nongad_name} for gad_name, nongad_name in zip(test_gad_t1,test_nongad_t1)] #creates list of dictionaries, with gad and nongad images labelled
    inference_transforms = Compose( #loading full image
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"])])

    infer_ds = Dataset(data=test_files, transform=inference_transforms) 
    infer_loader = DataLoader(infer_ds, batch_size=1, shuffle=True) #using pytorch's dataloader

    degad_imgs = []
    gad_infer_imgs = []
    nongad_infer_imgs = []
    for infer_imgs in infer_loader:
        gad_infer_imgs.append(infer_imgs["image"])
        nongad_infer_imgs.append(infer_imgs["label"])
        output_degad_img = sliding_window_inference(inputs = infer_imgs["image"].to('cpu'), roi_size = dims_tuple, sw_batch_size= 5, predictor = gen_model.to('cpu'), overlap = 0.25, mode = "gaussian", sw_device= 'cpu', device = 'cpu', progress = True )
        degad_imgs.append(output_degad_img)


    for i in range(len(degad_imgs)): #looping thru number of output files
        degad_img =degad_imgs[i][0][0] # reshaping to exclude batch and channels (only one channel)
        gad_image= nibabel.load(test_files[i]["image"]) # getting original gad image back to compare to 
        gad_image_file = test_files[i]["image"]
        print(gad_image_file)
        sub = os.path.basename(gad_image_file).split("_")[0]
        degad_name = f'{sub}_acq-degad_T1w.nii.gz'
        degad_file = nibabel.Nifti1Image(degad_img.detach().numpy()*100, affine= gad_image.affine,header= gad_image.header) # with same header as inference gad 
        output_dir_test = f'{output_dir}/test'
        os.makedirs(f'{output_dir_test}/bids/{sub}/ses-pre/anat', exist_ok=True)# save in bids format
        output_path = f'{output_dir_test}/bids/{sub}/ses-pre/anat/{degad_name}'
        nibabel.save(degad_file,output_path)


    import random
    ## generating random whole brain slices
    fig, axes = plt.subplots(8, 4,figsize=(10,25))
    plt.suptitle('Whole brain slices: Gad, NonGad, Degad, Subtraction map')

    for i in range (1,33,4):
        plt.subplot(8, 4, i)
        x = random.randint(80, 190)
        plt.imshow(gad_infer_imgs[0][0, 0,40:210 ,40:150, x].cpu().data.numpy(), cmap ="gray")

        plt.subplot(8, 4, i+1)
        plt.imshow(nongad_infer_imgs[0][0, 0, 40:210 , 40:150, x].cpu().data.numpy(), "gray")

        plt.subplot(8, 4, i+2)
        plt.imshow(degad_imgs[0][0, 0, 40:210,40:150, x].cpu().data.numpy(), "gray")

        plt.subplot(8, 4, i+3)
        noise_vector = degad_imgs[0][0,0,:,:,:] - nongad_infer_imgs[0][0,0,:,:,:] 
        #pos values are where model overestimated intensities and neg values are where the model underestimated
        plt.imshow(noise_vector[40:210,40:150, x].cpu().data.numpy(), "seismic",vmin=-1,vmax=1)
        plt.colorbar()
    plt.savefig(f'{output_dir}/test/figure_whole_brain.png')


    #generating random 32x32 slices
    fig, axes = plt.subplots(8, 4,figsize=(10,30))
    plt.suptitle('Patches: Gad, NonGad, Degad, subtraction map')

    for i in range (1,33,4):

        x = random.randint(40, 190)
        y = random.randint(40, 190)
        z = random.randint(40, 190)
        plt.subplot(8, 4, i)
        plt.imshow(gad_infer_imgs[0][0, 0, x:x+32,y:y+32 ,50].cpu().data.numpy(), cmap ="gray")
        plt.subplot(8, 4, i+1)
        plt.imshow(nongad_infer_imgs[0][0, 0, x:x+32,y:y+32 ,50].cpu().data.numpy(), "gray")
        plt.subplot(8, 4, i+2)
        plt.imshow(degad_imgs[0][0, 0, x:x+32,y:y+32,50].cpu().data.numpy(), "gray")

        plt.subplot(8, 4, i+3)
        noise_vector = degad_imgs[0][0,0,:,:,:] - nongad_infer_imgs[0][0,0,:,:,:] 
        #pos values are where model overestimated intensities and neg values are where the model underestimated
        plt.imshow(noise_vector[x:x+32,y:y+32,50].cpu().data.numpy(), "seismic",vmin=-1,vmax=1)
        plt.colorbar()

    plt.savefig(f'{output_dir}/test/figure_32_patches.png')  

class GANDiscriminator(nn.Module):
    def __init__(self, ini_filters):
        super().__init__()
        in_channels=2
        kernel_size=3
       
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, ini_filters, kernel_size, stride=2, padding=1),
            nn.InstanceNorm3d(ini_filters),
            nn.PReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(ini_filters, ini_filters*2, kernel_size, stride=2, padding=1),
            nn.InstanceNorm3d(ini_filters*2),
            nn.PReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(ini_filters*2, ini_filters*4, kernel_size, stride=2, padding=1),
            nn.InstanceNorm3d(ini_filters*4),
            nn.PReLU()
        )
        
        self.conv_out = nn.Conv3d(ini_filters*4, 1, kernel_size, stride=1, padding=0)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.tanh(x)
        return x



def GeneratorLoss(nongad_images,degad_images, fake_preds, device):
    """
    Loss function is the sum of the binary cross entropy between the error of the discriminator output btwn gad and degad (fake prediction) and the root mean square error betwn as nongad and degad images multiplies by scalar weight coefficient
    nongad_image= real nongad images from the sample
    degad_images= generated nongad images from the generator
    fake_preds: output of discriminator when fed fake data
    """
    
    coeff = 0.01
    
    BCE_loss= torch.nn.BCELoss() 
    real_target = torch.ones((fake_preds.shape[0], fake_preds.shape[1], fake_preds.shape[2], fake_preds.shape[3], fake_preds.shape[4])) #new_full returns a tensor filled with 1 with the same shape as the discrminator prediction 
    fake_preds = torch.sigmoid(fake_preds) # applying sigmmoid function to output of the discriminator to map probability between 0 and 1
    BCE_fake = BCE_loss(fake_preds.to(device), real_target.to(device)) # BCE loss btwn the output of discrim when fed fake data and 1 <- generator wants to minimize this
    L1_loss = torch.nn.L1Loss()
    loss = L1_loss(degad_images, nongad_images)  # producing RMSE between ground truth nongad and degad
    generator_loss = loss*coeff + BCE_fake
    return generator_loss


# In[25]:


def DiscriminatorLoss(real_preds, fake_preds,device):
    """
    Loss function for the discriminator: The discriminator loss is calculated by taking the sum of the L2 error of the discriminator output btwn gad and nongad( real prediction ) and the L2 error of the output btwn gad and degad( fake predition)
    
    real_preds: output of discriminator when fed real data
    fake_preds: output of discriminator when fed fake data
    """
    
    real_target = torch.ones((real_preds.shape[0], real_preds.shape[1], real_preds.shape[2],real_preds.shape[3], real_preds.shape[4])) #new_full returns a tensor filled with 1 with the same shape as the discrminator prediction 
    
    fake_target = torch.zeros((fake_preds.shape[0], fake_preds.shape[1], fake_preds.shape[2], fake_preds.shape[3], fake_preds.shape[4])) #new_full returns a tensor filled with 0 w/ the same shape as the generator prediction
    BCE_loss =  torch.nn.BCELoss().to(device)  # creates a losss value for each batch, averaging the value across all elements
    # Apply sigmoid to discriminator outputs, to fit between 0 and 1
    real_preds = torch.sigmoid(real_preds).cuda()
    fake_preds = torch.sigmoid(fake_preds).cuda()
    
    BCE_fake = BCE_loss(fake_preds.cuda(), fake_target.cuda()) # BCE loss btwn the output of discrim when fed fake data and 0 <- generator wants to minimize this
    BCE_real = BCE_loss(real_preds.cuda(), real_target.cuda()) # BCE loss btwn the output of discrim when fed real data and 1 <- generator wants to minimize this
    
    return BCE_real + BCE_fake

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN degad model with specified parameters.")
    parser.add_argument("--input", nargs='+', required=True, help="Path to the training and validation data, in that order")
    parser.add_argument("--patch_size", type=int, required=True, help="Patch size for training.")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training.")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for training.")
    parser.add_argument("--ini_filter_G", type=int, required=True, help="Number of filters in initial layer of generator.")
    parser.add_argument("--ini_filter_D", type=int, required=True, help="Number of filters in initial layer of discriminator.")
    parser.add_argument("--depth_G", type=int, required=True, help="Depth of generator.")
    parser.add_argument("--num_conv_G", type=int, required=True, help="Number of convolutions in each layer of generator.")
    parser.add_argument("--num_conv_D", type=int, required=True, help="Number of convolutions in each layer of discriminator.")
    parser.add_argument("--train_steps_D", required=True, help="Number of steps being applied to discriminator.")

    args = parser.parse_args()
    input_files = args.input
    patch_size = args.patch_size
    batch_size = args.batch_size
    lr = args.lr
    filter_num_G=args.ini_filter_G
    filter_num_D=args.ini_filter_D
    depth_G= args.depth_G
    num_conv_G=args.num_conv_G
    num_conv_D=args.num_conv_D
    train_steps_D=args.train_steps_D

    train_model(input_files,patch_size, batch_size,lr,filter_num_G,filter_num_D,depth_G,num_conv_G,num_conv_D, train_steps_D)