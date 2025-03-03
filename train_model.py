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

def normalize(data, mean, std):
    return (data - mean) / std

def unnormalize(data, mean, std):
    return (data * std) + mean

class physics_loss(nn.Module):
    def __init__(self):
        super(physics_loss, self).__init__()
    
    def compute_spatial_derivatives(self, q, res):
        degree2meter = 111111
        dlon = res * degree2meter
        dlat = res * degree2meter

        # Compute dq/dx (longitude gradient) using central difference
        dq_dx = torch.zeros_like(q)
        dq_dx[:, 1:-1] = (q[:, 2:] - q[:, :-2]) / (2 * dlon)
        dq_dx[:, 0] = dq_dx[:, 1]  # Handle left boundary
        dq_dx[:, -1] = dq_dx[:, -2]  # Handle right boundary

        # Compute dq/dy (latitude gradient) using central difference
        dq_dy = torch.zeros_like(q)
        dq_dy[1:-1, :] = (q[2:, :] - q[:-2, :]) / (2 * dlat)
        dq_dy[0, :] = dq_dy[1, :]  # Handle bottom boundary
        dq_dy[-1, :] = dq_dy[-2, :]  # Handle top boundary

        return dq_dx, dq_dy
    
    def calculate_gamma2(self, pressure):
        Cp = 1.0035
        Lambda = 2.45
        MW = 0.622
        gamma = (Cp * pressure) / (Lambda * MW)
        return gamma
    
    def calculate_saturation_vapor(self, temperature):
        saturation_vapor = torch.zeros_like(temperature)

        # Convert temperature from Kelvin to Celsius
        T_C = temperature - 273.15  # Convert K to °C

        saturation_vapor[T_C > 0] = 0.61078 * torch.exp((17.27 * T_C[T_C > 0]) / (T_C[T_C > 0] + 237.3))
        saturation_vapor[T_C <= 0] = 0.61078 * torch.exp((21.875 * T_C[T_C <= 0]) / (T_C[T_C <= 0] + 265.5))

        #convert from kPa to Pa
        saturation_vapor = saturation_vapor * 1000
        return saturation_vapor

    def calculate_slope_saturation_vapor(self, temperature):
        es = self.calculate_saturation_vapor(temperature)
        # Convert temperature from Kelvin to Celsius
        T_C = temperature - 273.15  # Convert K to °C
        ses = (4098 * es) / ((T_C + 237.3) ** 2)
        return ses
    
    #def calculate_ET(self, delta, gamma, cloud_cover, pressure):
    def calculate_ET(self, delta, gamma):
        ET = 0.65 * (delta / (delta + gamma)) #* (Rs / gamma)
        #print('cek RS gamma', Rs, gamma)
        #print('cek delta', delta)
        #exit()
        return ET
    
    def forward(self, Q_past, pred, true):
        Q_now_pred = pred[0, 5, :, :]

        Temperature_pred = pred[0, 0, :, :]
        U1000_pred = pred[0, 1, :, :]
        V1000_pred = pred[0, 2, :, :]
        U850_pred = pred[0, 3, :, :]
        V850_pred = pred[0, 4, :, :]
        #Cloud_pred = pred[0, 5, :, :]
        precip_pred = pred[0, 6, :, :]

        dq_dt_pred = (Q_now_pred - Q_past)
        dq_dx_pred, dq_dy_pred = self.compute_spatial_derivatives(Q_now_pred, 0.25)

        delta_pred = self.calculate_slope_saturation_vapor(Temperature_pred)
        gamma_pred = self.calculate_gamma2(100)
        ET_pred = self.calculate_ET(delta_pred, gamma_pred)

        minET_pred = precip_pred + dq_dt_pred + (U1000_pred * dq_dx_pred) + (V1000_pred * dq_dy_pred) + (U850_pred * dq_dx_pred) + (V850_pred * dq_dy_pred)
        loss_pred = ET_pred - minET_pred

        #print('cek delta', delta_pred)
        #print('cek temperature', Temperature_pred)
        #print('cek min mean max delta', torch.min(delta_pred), torch.mean(delta_pred), torch.max(delta_pred))
        #print('cek delta gamma', torch.isnan(delta_pred).any(), gamma_pred)
        #print('cek dqdx dqdy pred', torch.isnan(dq_dx_pred).any(), torch.isnan(dq_dy_pred).any())
        #print('cek ET pred', torch.isnan(ET_pred).any())
        #print('cek minET pred', torch.isnan(minET_pred).any())

        Q_now_true = true[0, 5, :, :]

        Temperature_true = true[0, 0, :, :]
        U1000_true = true[0, 1, :, :]
        V1000_true = true[0, 2, :, :]
        U850_true = true[0, 3, :, :]
        V850_true = true[0, 4, :, :]
        #Cloud_true = true[0, 5, :, :]
        precip_true = true[0, 6, :, :]

        dq_dt_true = (Q_now_true - Q_past)
        dq_dx_true, dq_dy_true = self.compute_spatial_derivatives(Q_now_true, 0.25)

        delta_true = self.calculate_slope_saturation_vapor(Temperature_true)
        gamma_true = self.calculate_gamma2(100)
        ET_true = self.calculate_ET(delta_true, gamma_true)

        minET_true = precip_true + dq_dt_true + (U1000_true * dq_dx_true) + (V1000_true * dq_dy_true) + (U850_true * dq_dx_true) + (V850_true * dq_dy_true)
        loss_true = ET_true - minET_true

        #consequence of central difference
        loss_pred = loss_pred[1:-1, 1:-1]
        loss_true = loss_true[1:-1, 1:-1]

        #print('cek loss pred true', torch.isnan(loss_pred).any(), torch.isnan(loss_true).any())

        the_loss = loss_pred - loss_true
        return torch.nanmean(the_loss**2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

precips = np.load('precips.npy')
Temps = np.load('Temps.npy')
Us = np.load('Us.npy')
Vs = np.load('Vs.npy')
Qs = np.load('Qs.npy')
#Cloudsurface = np.load('Cloudsurfaces.npy')


dataset_train = []
dataset_test = []
train_period = 30
for iter_d in range(1, len(Temps)-1):
    precip = precips[iter_d, :, :]
    U1000 = Us[iter_d, -1, :, :]
    V1000 = Vs[iter_d, -1, :, :]

    U850 = Us[iter_d, 0, :, :]
    V850 = Vs[iter_d, 0, :, :]

    Temperature = Temps[iter_d, -1, :, :]
    #Cloud = Cloudsurface[iter_d, :, :]

    Q = Qs[iter_d, -1, :, :]
    #Q_past = Qs[iter_d-1, -1, :, :]
    #dq_dt = (Q - Q_past)
    
    if iter_d < train_period:
        dataset_train.append([Temperature, U1000, V1000, U850, V850, Q, precip])
    else:
        dataset_test.append([Temperature, U1000, V1000, U850, V850, Q, precip])

dataset_train = np.array(dataset_train)
dataset_test = np.array(dataset_test)

#get mean and std
mean_train = np.mean(dataset_train, axis=(0,2,3))
std_train = np.std(dataset_train, axis=(0,2,3))

mean_train_torch = torch.tensor(mean_train, dtype=torch.float32)
std_train_torch = torch.tensor(std_train, dtype=torch.float32)

mean_train_torch = mean_train_torch.to(device).float()
std_train_torch = std_train_torch.to(device).float()

#print('cek mean', mean_train)
#exit()
#print(dataset_train.shape, dataset_test.shape)

#np.save('dataset_train.npy', dataset_train)
#np.save('dataset_test.npy', dataset_test)



model = SFNO(n_modes=(120, 120), in_channels=7, out_channels=7, hidden_channels=32, projection_channels=64, factorization='dense').to(device)
print(model)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')

optimizer = AdamW(model.parameters(),
                                lr=8e-4,
                                weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

#l2loss = LpLoss(d=2, p=2, reduce_dims=(0,1))
l2loss = LpLoss(d=2, p=2, reduce_dims=(0,1))
phylos = physics_loss()
train_loss = l2loss

epoch = 100
total_train = 50
indexes_choice = np.arange(dataset_train.shape[0]-1)

dataset_train_torch = torch.tensor(dataset_train, dtype=torch.float32).to(device)
total_real_loss_threshold = 100
break_nan = False
for iter_ep in range(epoch):
    total_real_loss = 0
    total_ploss = 0
    total_eval_loss = 0
    for iter_b in range(total_train):
        idx_choice = np.random.choice(indexes_choice)
        features = dataset_train_torch[idx_choice, :, :, :]
        features = torch.unsqueeze(features, 0)
        labels = dataset_train_torch[idx_choice+1, :, :, :]
        labels = torch.unsqueeze(labels, 0)

        labels_norm = normalize(labels, mean_train_torch[:, None, None], std_train_torch[:, None, None]).float()
        features_norm = normalize(features, mean_train_torch[:, None, None], std_train_torch[:, None, None]).float()

        labels_rain_norm = labels_norm[:, 6, :, :]
        labels_rain_norm = torch.unsqueeze(labels_rain_norm, 1)

        optimizer.zero_grad()  # Reset gradients before backpropagation

        #print('cek dtype feature norm', features_norm.dtype)
        y_pred_norm = model(features_norm)
        y_pred_rain_norm = y_pred_norm[:, 5, :, :]
        y_pred_rain_norm = torch.unsqueeze(y_pred_rain_norm, 1)
        #print(y_pred_rain_norm.shape)

        #vanilla loss
        eval_loss = l2loss(y_pred_rain_norm, labels_rain_norm)

        qpast = features[0, 5, :, :]
        y_pred = unnormalize(y_pred_norm, mean_train_torch[None, :, None, None], std_train_torch[None, :, None, None])

        #physics loss
        ploss = phylos(qpast, y_pred, labels)

        real_loss = eval_loss + ploss
        eval_loss_val = eval_loss.item()
        ploss_val = ploss.item()

        total_real_loss += real_loss.item()
        total_ploss += ploss_val
        total_eval_loss += eval_loss_val
        print(f'Epoch {iter_ep+1} - Batch {iter_b+1} - Real Loss: {real_loss.item()} - Eval Loss: {eval_loss_val} - Physics Loss: {ploss_val}')

        if total_real_loss is torch.nan:
            break_nan = True
            break

        real_loss.backward()
        optimizer.step()

    if break_nan:
        break

    scheduler.step()

    if total_real_loss < total_real_loss_threshold:
        # Save the model along with mean and std
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'mean': mean_train_torch,
            'std': std_train_torch,
            'epoch': iter_ep,
        }
        torch.save(checkpoint, 'trained_model_with_stats_best.pth')
        total_real_loss_threshold = total_real_loss
        #total_eval_loss += eval_loss.item()
        #print(total_eval_loss)

# Save the model along with mean and std
checkpoint = {
    'model_state_dict': model.state_dict(),
    'mean': mean_train_torch,
    'std': std_train_torch
}
torch.save(checkpoint, 'trained_model_with_stats.pth')
print("Model and statistics saved successfully.")