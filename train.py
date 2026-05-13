
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss, DiceCELoss

import torch
from torch.optim import Adam

from preprocess import prepare
from utilities import train

data_dir = r'C:\Users\Dell\Downloads\T_Segmentation\Task05_Prostate'
model_dir = r'C:\Users\Dell\Downloads\T_Segmentation\Results' 
data_in = prepare(data_dir, cache=True)

device = torch.device("cpu")
model = UNet(
    spatial_dims=3,
    in_channels=2,
    out_channels=2,
    channels=(16, 32, 64, 128, 256), 
    strides=(2, 2, 2, 2),
    num_res_units=2,
    norm=Norm.BATCH,
)

# loss_function = DiceCELoss(to_onehot_y=True, sigmoid=True, squared_pred=True, ce_weight=calculate_weights(1792651250,2510860).to(device))
loss_function = DiceLoss(to_onehot_y=True, sigmoid=True, squared_pred=True)

optimizer = Adam(model.parameters(), 1e-4, weight_decay=1e-5, amsgrad=True)

if __name__ == '__main__':
    train(model, data_in, loss_function, optimizer, 10, model_dir, device=device)