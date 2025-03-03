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

color_to_value = {
    (7, 254, 246): 1,
    (0, 150, 255): 5,
    (0, 2, 254): 10,
    (1, 254, 3): 20,
}

value_mapping = np.array(list(color_to_value.values()))

def normalize(data, mean, std):
    return (data - mean) / std

def unnormalize(data, mean, std):
    return (data * std) + mean

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

test_data = np.load('dataset_test.npy')
test_data = torch.tensor(test_data, dtype=torch.float32)

print(test_data)
print(test_data.shape)

lat2d = np.load('lat.npy')
lon2d = np.load('lon.npy')

model = SFNO(n_modes=(120, 120), in_channels=7, out_channels=7, hidden_channels=32, projection_channels=64, factorization='dense').to(device)

# Load the checkpoint
checkpoint = torch.load('trained_model_with_stats.pth', map_location=device)

# Load the model state dictionary
model.load_state_dict(checkpoint['model_state_dict'])

# Load the mean and std
mean_train_torch = checkpoint['mean']
std_train_torch = checkpoint['std']



model.eval()  # Set the model to evaluation mode
print("Model and statistics loaded successfully.")

# Define the colormap and boundaries for the categories
#cmap = plt.get_cmap('viridis')
#bounds = [0, 1, 5, 10, 20]
#norm = BoundaryNorm(bounds, cmap.N)

# Define fixed colormap categories and evenly spaced colorbar
colors = ['#00FD02', '#FDFC00', '#FC7506', '#C90002', '#9600FB', '#9600FB']  # Custom colors
cmap = mcolors.ListedColormap(colors)

bounds = [1, 5, 10, 20, 100]  # Rainfall categories
norm = mcolors.BoundaryNorm(bounds, cmap.N, extend='max')



for iter_d in range(len(test_data)-1):
    if iter_d == 0:
        features = test_data[iter_d, :, :, :].to(device)
        features = torch.unsqueeze(features, 0)

    labels = test_data[iter_d+1, :, :, :].to(device)

    print('shape feature', features.shape)
    print('shape labels', labels.shape)
    features_norm = normalize(features, mean_train_torch[None, :, None, None], std_train_torch[None, :, None, None]).float()

    y_pred_norm = model(features_norm)
    y_pred = unnormalize(y_pred_norm, mean_train_torch[None, :, None, None], std_train_torch[None, :, None, None])
    y_pred_rain = y_pred[0, 6, :, :]
    y_pred_rain[y_pred_rain < 0] = 0
    #y_pred[0, 6, :, :] = y_pred_rain

    y_pred_rain = y_pred_rain.cpu().detach().numpy()
    y_truth_rain = labels[6, :, :].cpu().detach().numpy()
    
    y_truth_rain[y_truth_rain == 0] = np.nan 
    y_pred_rain[y_pred_rain == 0] = np.nan

    #print(y_pred_rain.shape)
    #print(y_pred_rain)

#    # Plot the rain prediction and ground truth side by side
#    plt.figure(figsize=(14, 6))
#
#    plt.subplot(1, 2, 1)
#    plt.title('Rain Prediction')
#    m = Basemap(projection='merc', resolution='h',
#                llcrnrlat=np.min(lat2d), urcrnrlat=np.max(lat2d),
#                llcrnrlon=np.min(lon2d), urcrnrlon=np.max(lon2d))
#    m.drawcoastlines()
#    m.drawcountries()
#    m.drawmapboundary()
#
#    x, y = m(lon2d, lat2d)
#    im1 = m.contourf(x, y, y_pred_rain, cmap=cmap, norm=norm)
#    cbar1 = m.colorbar(im1, ax=plt.gca(), boundaries=bounds, ticks=[1, 5, 10, 20])
#    cbar1.ax.set_yticklabels(['slight', 'mod', 'heavy', 'very heavy'])
#
#    plt.subplot(1, 2, 2)
#    plt.title('Rain Ground Truth')
#
#    m = Basemap(projection='merc', resolution='h',
#                llcrnrlat=np.min(lat2d), urcrnrlat=np.max(lat2d),
#                llcrnrlon=np.min(lon2d), urcrnrlon=np.max(lon2d))
#    m.drawcoastlines()
#    m.drawcountries()
#    m.drawmapboundary()
#
#    x, y = m(lon2d, lat2d)
#    im1 = m.contourf(x, y, y_truth_rain, cmap=cmap, norm=norm)
#    cbar2 = m.colorbar(im1, ax=plt.gca(), boundaries=bounds, ticks=[1, 5, 10, 20])
#    cbar2.ax.set_yticklabels(['slight', 'mod', 'heavy', 'very heavy'])
#
#    # Adjust layout
#    plt.tight_layout()
#
#    # Save the plot
#    plt.savefig(f'rain_prediction_vs_truth_{iter_d}.png')
#    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.set_title(f'Rain Prediction step-{iter_d+1}')
    m1 = Basemap(projection='merc', resolution='h',
                llcrnrlat=np.min(lat2d), urcrnrlat=np.max(lat2d),
                llcrnrlon=np.min(lon2d), urcrnrlon=np.max(lon2d),
                ax=ax1)
    m1.drawcoastlines()
    m1.drawcountries()
    m1.drawmapboundary()

    x, y = m1(lon2d, lat2d)
    #im1 = m1.contourf(x, y, y_pred_rain, cmap=cmap, norm=norm, extend='both')
    im1 = m1.contourf(x, y, y_pred_rain, levels=bounds, cmap=cmap, norm=norm)

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

    # --- Create Shared Evenly Spaced Colorbar ---
    #cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Adjust position for alignment
    #cbar = fig.colorbar(im1, cax=cbar_ax, boundaries=bounds, spacing='proportional', ticks=bounds)
    #cbar.ax.set_yticklabels(['slight', 'mod', 'heavy', 'very heavy'])
    #cbar.set_label("Rainfall Intensity (mm)")
    #plt.colorbar(label='Rainfall Intensity')
    plt.tight_layout()

    plt.subplots_adjust(right=0.9, wspace=0.2)  # Ensure spacing between subplots
    plt.savefig(f'rain_prediction_vs_truth_{iter_d}.png')
    plt.close()

    #prepare for next iteration
    y_pred_rain = torch.from_numpy(y_pred_rain)
    y_pred_rain = torch.nan_to_num(y_pred_rain, nan=0.0)
    y_pred[0, 6, :, :] = y_pred_rain
    features = y_pred
    features = torch.nan_to_num(features, nan=0.0)
    #print('cek nan in features', torch.isnan(features).sum())
    ##features = torch.unsqueeze(features, 0)
    #print('cek features shape', features.shape)
    #if iter_d == 1:
    #    break