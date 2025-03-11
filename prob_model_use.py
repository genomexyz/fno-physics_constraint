import numpy as np
import torch
import torch.nn as nn
from neuralop.models import SFNO, TFNO
from neuralop import Trainer
from neuralop.training import AdamW
from neuralop.data.datasets import load_spherical_swe
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.basemap import Basemap

import matplotlib.colors as mcolors

def predict_multiple(model, x, num_samples=10):
    model.train()  # Keep dropout active
    outputs = torch.stack([model(x) for _ in range(num_samples)])  # Run multiple stochastic passes
    return outputs

def predict_multiple(model, x, num_samples=10):
    model.train()  # Keep dropout active
    
    with torch.no_grad():  # Prevent autograd from using memory
        outputs = torch.empty((num_samples, *x.shape), dtype=x.dtype, device=x.device)  # Preallocate tensor

        for i in range(num_samples):
            outputs[i] = model(x)  # Store predictions one at a time to avoid unnecessary tensor stacking

    return outputs

def normalize(data, mean, std):
    return (data - mean) / std

def unnormalize(data, mean, std):
    return (data * std) + mean

class ExtendedFNO(nn.Module):
    def __init__(self, new_fno, dropout_rate):
        super(ExtendedFNO, self).__init__()

        #self.fno = trained_fno  # Load the trained FNO model
        #for param in self.fno.parameters():
        #    param.requires_grad = False  # Freezing all FNO layers
        self.dropout = nn.Dropout2d(p=dropout_rate)  # Spatial dropout
        self.fno_out = new_fno

    def forward(self, x):
        x = self.dropout(x)  # Apply dropout
        x = self.fno_out(x)  # Pass through output layer
        return x

color_to_value = {
    (7, 254, 246): 1,
    (0, 150, 255): 5,
    (0, 2, 254): 10,
    (1, 254, 3): 20,
}

value_mapping = np.array(list(color_to_value.values()))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_data = np.load('dataset_test.npy')
test_data = torch.tensor(test_data, dtype=torch.float32)

lat2d = np.load('lat.npy')
lon2d = np.load('lon.npy')


model_pre = SFNO(n_modes=(120, 120), in_channels=7, out_channels=7, hidden_channels=32, projection_channels=64, factorization='dense').to(device)
# Load the checkpoint
checkpoint = torch.load('trained_model_with_stats.pth', map_location=device)
model_pre.load_state_dict(checkpoint['model_state_dict'])

model2nd = SFNO(n_modes=(120, 120), in_channels=7, out_channels=7, hidden_channels=32, projection_channels=64, factorization='dense').to(device)
model = ExtendedFNO(model2nd, 0.2)
checkpoint2nd = torch.load('trained_model_prob_with_stats.pth', map_location=device)
model.load_state_dict(checkpoint2nd['model_state_dict'])

# Load the mean and std
mean_train_torch = checkpoint['mean']
std_train_torch = checkpoint['std']

colors = ['#00FD02', '#FDFC00', '#FC7506', '#C90002', '#9600FB', '#9600FB']  # Custom colors
cmap = mcolors.ListedColormap(colors)

bounds = [1, 5, 10, 20, 100]  # Rainfall categories
norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='max')

colors_prob = ['#CFFD51', '#67FD3D', '#FFFD74', '#FE9942', '#F84438', '#CA0F98']  # Custom colors
cmap_prob = mcolors.ListedColormap(colors)

bounds_prob = [0.2, 0.4, 0.6, 0.8, 1]  # Rainfall categories
norm_prob = mcolors.BoundaryNorm(bounds_prob, cmap_prob.N, extend='max')

threshold_heavy_rainfall = 10
for iter_d in range(len(test_data)-1):
    if iter_d == 0:
        features = test_data[iter_d, :, :, :].to(device)
        features = torch.unsqueeze(features, 0)
    
    labels = test_data[iter_d+1, :, :, :].to(device)

    print('shape feature', features.shape)
    print('shape labels', labels.shape)
    features_norm = normalize(features, mean_train_torch[None, :, None, None], std_train_torch[None, :, None, None]).float()

    y_pred_norm = model_pre(features_norm)
    y_pred_norm = predict_multiple(model, y_pred_norm, num_samples=7)
    y_pred_norm = torch.squeeze(y_pred_norm, 1)
    y_pred = unnormalize(y_pred_norm, mean_train_torch[None, :, None, None], std_train_torch[None, :, None, None])

    y_pred_rain = y_pred[:, 6, :, :]
    y_pred_rain[y_pred_rain < 0] = 0

    heavy_rainfall_prob = torch.zeros_like(y_pred_rain)
    heavy_rainfall_prob[y_pred_rain >= threshold_heavy_rainfall] = 1
    heavy_rainfall_prob = heavy_rainfall_prob.mean(dim=0)

    print('shape data', heavy_rainfall_prob, heavy_rainfall_prob.size())

    #heavy_rainfall_prob = heavy_rainfall_prob.mean(dim)

    #check stat
    #for iter_p in range(len(y_pred_norm)):
    #    single_prob = y_pred_norm[iter_p]
    #    single_mean_prob = single_prob.mean(dim=(1,2))
    #    single_std_prob = single_prob.std(dim=(1,2))
    #    print('iter %s mean %s'%(iter_p, single_mean_prob))
    #    print('iter %s std %s'%(iter_p, single_std_prob))
    #print('cek shape', y_pred_norm.size())

    y_pred_rain = y_pred_rain.cpu().detach().numpy()
    y_truth_rain = labels[6, :, :].cpu().detach().numpy()

    y_truth_rain[y_truth_rain == 0] = np.nan 
    y_pred_rain[y_pred_rain == 0] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.set_title(f'Heavy Rainfall Probability step-{iter_d+1}')
    m1 = Basemap(projection='merc', resolution='h',
                llcrnrlat=np.min(lat2d), urcrnrlat=np.max(lat2d),
                llcrnrlon=np.min(lon2d), urcrnrlon=np.max(lon2d),
                ax=ax1)
    m1.drawcoastlines()
    m1.drawcountries()
    m1.drawmapboundary()

    x, y = m1(lon2d, lat2d)
    #im1 = m1.contourf(x, y, y_pred_rain, cmap=cmap, norm=norm, extend='both')
    im1 = m1.contourf(x, y, heavy_rainfall_prob, levels=bounds_prob, cmap=cmap_prob, norm=norm_prob)
    cbar1 = m1.colorbar(im1, ax=ax1, boundaries=bounds, ticks=[0.2, 0.4, 0.6, 0.8])
    cbar1.ax.set_yticklabels(['20%', '40%', '60%', '80%'])

    # --- Plot Rain Ground Truth ---
    ax2 = axes[1]
    ax2.set_title(f'Rain Ground Truth step-{iter_d+1}')
    m2 = Basemap(projection='merc', resolution='h',
                llcrnrlat=np.min(lat2d), urcrnrlat=np.max(lat2d),
                llcrnrlon=np.min(lon2d), urcrnrlon=np.max(lon2d),
                ax=ax2)
    m2.drawcoastlines()
    m2.drawcountries()
    m2.drawmapboundary()

    x, y = m2(lon2d, lat2d)
    #im2 = m2.contourf(x, y, y_truth_rain, cmap=cmap, norm=norm, extend='both')
    im2 = m2.contourf(x, y, y_truth_rain, levels=bounds, cmap=cmap, norm=norm)
    cbar2 = m2.colorbar(im2, ax=ax2, boundaries=bounds, ticks=[1, 5, 10, 20])
    cbar2.ax.set_yticklabels(['slight', 'mod', 'heavy', 'very heavy'])

    plt.tight_layout()

    plt.subplots_adjust(right=0.9, wspace=0.2)  # Ensure spacing between subplots
    plt.savefig(f'rain_prediction_prob_vs_truth_{iter_d}.png')
    plt.close()

    #prepare for next iteration
    y_pred_rain = torch.from_numpy(y_pred_rain)
    y_pred_rain = torch.nan_to_num(y_pred_rain, nan=0.0)
    y_pred[:, 6, :, :] = y_pred_rain
    y_pred = y_pred.mean(dim=0)
    y_pred = torch.unsqueeze(y_pred, 0)
    features = y_pred
    features = torch.nan_to_num(features, nan=0.0)
    
    #if iter_d >= 1:
    #    break