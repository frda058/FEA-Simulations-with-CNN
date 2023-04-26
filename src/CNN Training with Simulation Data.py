import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import skorch
from skorch.regressor import NeuralNetRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler, WeightedRandomSampler


# Set a seed
seed = 126
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

random_seed = 126 # or any of your favorite number 
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ## Import excel files and save as parquet
# **Convert to parquet to improve file read times**
X_df = pd.read_excel('features_all.xlsx', header=None)
X_df = X_df1.T
X_df = X_df1.iloc[:, 1:]
X_df.columns = X_df1.columns.astype(str)
X_df.to_parquet('simulation data/features_all.parquet')

y_df = pd.read_excel('labels_all.xlsx', header=None)
y_df = y_df_1.rename(index={0: 'E', 1: 'size', 2: 'p', 3: 'v', 4:'c', 5:'G', 6:'K', 7:'G/K'})
y_df = y_df_1.T
y_df.columns = y_df_1.columns.astype(str)
y_df.to_parquet('simulation data/labels_all.parquet')


# ## Read in Parquet Files

X_df = pd.read_parquet('simulation data/features_all.parquet')
y_df = pd.read_parquet('simulation data/labels_all.parquet')


# ## Check for null values in the features and labels
X_null = X_df[np.isnan(X_df).any(axis=1)].index.values
y_null = y_df[np.isnan(y_df).any(axis=1)].index.values

print(len(X_null), len(y_null))

union = pd.Series(np.union1d(ser1, ser2))
  
# intersection of the series
intersect = pd.Series(np.intersect1d(ser1, ser2))
  
# uncommon elements in both the series 
notcommonseries = union[~union.isin(intersect)]
  
# displaying the result
print(notcommonseries)


# ## Drop rows with null values in both features and labels

X_df = X_df.drop(index=list(notcommonseries))
y_df = y_df.drop(index=list(notcommonseries))

ser1 = X_df[np.isnan(X_df).any(axis=1)].index.values
ser2 = y_df[np.isnan(y_df).any(axis=1)].index.values
print(len(ser1), len(ser2))

X_df = X_df[~np.isnan(X_df).any(axis=1)]
y_df = y_df[~np.isnan(y_df).any(axis=1)]

print(X_df.shape, y_df.shape)
print(X_df.isnull().values.any())
print(y_df.isnull().values.any())


# ## Take every other signal data point to reduce size of the features

drop_idx = list(range(2,X_df.shape[1],2)) #Indexes to drop

drop_cols = [j for i,j in enumerate(X_df.columns) if i in drop_idx]

X_df2 = X_df.drop(drop_cols, axis=1) 


# ## Remove initial pulse

total_time = 3e-06
time_increment = 2e-09

sampling_num = int(total_time/time_increment)

X_df_ = X_df2.iloc[:, sampling_num:27000]


# ## Create a dataframe with the subset of target variables to predict
# 1) Elastic Modulus  
# 2) Size  
# 3) Acoustic Impedance  
# 4) Speed of Sound  

y_df_sub = y_df[['E', 'size', 'Z', 'c']]


# ## Scale data with MinMaxScaler
scaler = MinMaxScaler()    # Scaler for y
scaler2 = MinMaxScaler()   # Scaler for x

# Scale x
X_df_Scaled = (pd.DataFrame(scaler2.fit_transform(X_df_.T))).T

# Scale y
y_df_Scaled = pd.DataFrame(columns=[y_df_sub.columns])
y_df_Scaled = scaler.fit_transform(y_df_sub)


# CNN model
class MyNet(nn.Module):
    def __init__(self, fc_out):
        super(MyNet, self).__init__()

        # Convolution block
        self.conv_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5, stride=3, padding=2), #out_channels: number of filters/kernel
            nn.ReLU(),
            nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=3, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(5, stride=3, padding=0)
        )

        # Fully connected block
        self.fc_layer = nn.Sequential(
            nn.Linear(in_features = 16*197, out_features = 200),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(200, fc_out)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        #x = x.view(x.size(0), -1)
        x = x.view(-1, 16*197)
        x = self.fc_layer(x)
        return x
    


# ## Run CNN to predict elastic modulus, size, impedance, and speed of sound

# **Initialize Parameters**
instances = 5
lr = 0.0001
fc_out = 4
num_epochs = 2000

validation_size_RMSE = []
validation_size_MAPE = []
validation_size_MAE = []

validation_EM_RMSE = []
validation_EM_MAPE = []
validation_EM_MAE = []

validation_c_RMSE = []
validation_c_MAPE = []
validation_c_MAE = []

validation_Z_RMSE = []
validation_Z_MAPE = []
validation_Z_MAE = []



for i in range(instances):
    
    print("Split #", i+1)
    
    other_x, test_x, other_y, test_y = train_test_split(X_df_Scaled.values, y_df_Scaled, 
                                                    test_size = 0.10, random_state = seed*i)

    x_train_true = torch.unsqueeze(torch.from_numpy(other_x).float(), 1)

    x_val_true = torch.unsqueeze(torch.from_numpy(test_x).float(), 1)

    y_train_true = torch.from_numpy(other_y).float()

    y_val_true = torch.from_numpy(test_y).float()


    # CNN model
    
    model = MyNet(fc_out)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.MSELoss()
    

    # Variable to store epoch and loss for plotting
    epoch_list = []
    loss_train = []
    loss_val = []

    # Train the model
    for epoch in range(num_epochs):
        model.train()
        
        # forward pass: compute predicted outputs by passing inputs to the model
        output_t = model(x_train_true)
        
        # calculate the loss
        loss_t = criterion(output_t, y_train_true)


        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # backward pass: compute gradient of the loss with respect to model parameters
        loss_t.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()


        # prep model for evaluation
        model.eval() 
        with torch.no_grad():
            # forward pass: compute predicted outputs by passing inputs to the model
            output_v = model(x_val_true)

        # calculate the loss
        loss_v = criterion(output_v, y_val_true)
        

        if (epoch + 1) % 5 == 0:
            epoch_list.append(epoch+1)
            loss_train.append(loss_t.item())
            loss_val.append(loss_v.item())

        if (epoch + 1) % 10 == 0:
            print('Epoch [{}/{}], Train Loss: {:.10f}'.format(epoch + 1, num_epochs, loss_t))
            print('Epoch [{}/{}], Validation Loss: {:.10f}'.format(epoch + 1, num_epochs, loss_v))
            print()
            #print(loss_v.item())



    plt.clf()
    plt.figure(figsize=(8,4))
    plt.plot(epoch_list, loss_train, '-o', label='Training loss')
    plt.plot(epoch_list, loss_val, '-o', label='Validation loss')
    plt.legend()
    plt.title(f' CNN Model (lr = {lr})', fontweight='bold', fontsize=20)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(f'Loss_{i}.png', dpi=1200, format = 'png')
    plt.show()


    #y_other_true = y_other_true.detach().numpy()
    #y_other_pred = y_other_pred.detach().numpy()

    y_val_true = y_val_true.detach().numpy()
    y_val_pred = output_v.detach().numpy()

    unscaled_pred = scaler.inverse_transform(y_val_pred)
    unscaled_true = scaler.inverse_transform(y_val_true)
    

    unscaled_E_pred = unscaled_pred[:, 0]
    unscaled_E_true = unscaled_true[:, 0]
    
    unscaled_size_pred = unscaled_pred[:, 1]
    unscaled_size_true = unscaled_true[:, 1]
    
    unscaled_Z_pred = unscaled_pred[:, 2]
    unscaled_Z_true = unscaled_true[:, 2]
    
    unscaled_c_pred = unscaled_pred[:, 3]
    unscaled_c_true = unscaled_true[:, 3]
     

    # Calculate RMSE for val data
    RMSE_EM = np.sqrt(mean_squared_error(unscaled_E_true, unscaled_E_pred))
    print('Elastic Modulus RMSE = ', RMSE_EM)

    RMSE_Size = np.sqrt(mean_squared_error(unscaled_size_true, unscaled_size_pred))
    print('Size RMSE = ', RMSE_Size)
    print()
    
    RMSE_Z = np.sqrt(mean_squared_error(unscaled_Z_true, unscaled_Z_pred))
    print('Impedance RMSE = ', RMSE_Z)
    print()
    
    RMSE_c = np.sqrt(mean_squared_error(unscaled_c_true, unscaled_c_pred))
    print('SoS RMSE = ', RMSE_c)
    print()
    
    
    # Calculate MAPE for val data

    MAPE_EM = mean_absolute_percentage_error(unscaled_E_true, unscaled_E_pred)
    print('Elastic Modulus MAPE = ', MAPE_EM)

    MAPE_Size = mean_absolute_percentage_error(unscaled_size_true, unscaled_size_pred)
    print('Size MAPE = ', MAPE_Size)
    print()
      
    MAPE_Z = np.sqrt(mean_absolute_percentage_error(unscaled_Z_true, unscaled_Z_pred))
    print('Impedance RMSE = ', MAPE_Z)
    print()
    
    MAPE_c = np.sqrt(mean_absolute_percentage_error(unscaled_c_true, unscaled_c_pred))
    print('SoS RMSE = ', MAPE_c)
    print()
  
    

    # Calculate MAE for val data
    errors_EM = mean_absolute_error(unscaled_E_true, unscaled_E_pred)
    print('Elastic Modulus MAE = ', errors_EM)
    print()

    errors_Size = mean_absolute_error(unscaled_size_true, unscaled_size_pred)
    print('Size MAE = ', errors_Size)
    print()
    
    errors_Z = np.sqrt(mean_absolute_error(unscaled_Z_true, unscaled_Z_pred))
    print('Impedance RMSE = ', errors_Z)
    print()
        
    errors_c = np.sqrt(mean_absolute_error(unscaled_c_true, unscaled_c_pred))
    print('SoS RMSE = ', errors_c)
    print()
    
    
    validation_size_RMSE.append(RMSE_Size)
    validation_size_MAPE.append(MAPE_Size)
    validation_size_MAE.append(errors_Size)

    validation_EM_RMSE.append(RMSE_EM)
    validation_EM_MAPE.append(MAPE_EM)
    validation_EM_MAE.append(errors_EM)
    
    validation_c_RMSE.append(RMSE_c)
    validation_c_MAPE.append(MAPE_c)
    validation_c_MAE.append(errors_c)
    
    validation_Z_RMSE.append(RMSE_Z)
    validation_Z_MAPE.append(MAPE_Z)
    validation_Z_MAE.append(errors_Z)


    
    plt.clf()
    plt.figure(figsize=(5,5))
    plt.scatter(unscaled_E_true, unscaled_E_pred, c = 'darkviolet', s=100, alpha=0.5, marker = "o")
    p1 = max(max(unscaled_E_pred), max(unscaled_E_true))
    p2 = min(min(unscaled_E_pred), min(unscaled_E_true))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual Values (kPa)', fontsize=15)
    plt.ylabel('Predictions (kPa)', fontsize=15)
    plt.suptitle('Elastic Modulus', fontweight = 'bold', fontsize=20)
    plt.title(f'MAPE = {MAPE_EM:.3f}, MAE = {errors_EM:.3f}, RMSE = {RMSE_EM:.3f}')
    #plt.xticks(rotation=45)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f'EM_{i}.png', dpi=1200, format = 'png')
    plt.show()

    plt.clf()
    plt.figure(figsize=(5,5))
    plt.scatter(unscaled_size_true*1000, unscaled_size_pred*1000, c = 'crimson', s=100, alpha=0.5, marker = "o")
    p1 = max(max(unscaled_size_pred), max(unscaled_size_true))*1000
    p2 = min(min(unscaled_size_pred), min(unscaled_size_true))*1000
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual Values (mm)', fontsize=15)
    plt.ylabel('Predictions (mm)', fontsize=15)
    plt.suptitle('Size', fontweight = 'bold', fontsize=20)
    plt.axis('equal')
    plt.title(f'MAPE = {MAPE_Size:.3f}, MAE = {errors_Size*1000:.3f}, RMSE = {RMSE_Size*1000:.3f}')
    plt.tight_layout()
    plt.savefig(f'Size_{i}.png', dpi=1200, format = 'png')
    plt.show()
    
    plt.clf()
    plt.figure(figsize=(5,5))
    plt.scatter(unscaled_c_true, unscaled_c_pred, c = 'green', s=100, alpha=0.5, marker = "o")
    p1 = max(max(unscaled_c_pred), max(unscaled_c_true))
    p2 = min(min(unscaled_c_pred), min(unscaled_c_true))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual Values (m/s)', fontsize=15)
    plt.ylabel('Predictions (m/s)', fontsize=15)
    plt.suptitle('Speed of Sound', fontweight = 'bold', fontsize=20)
    plt.axis('equal')
    plt.title(f'MAPE = {MAPE_c:.3f}, MAE = {errors_c:.3f}, RMSE = {RMSE_c:.3f}')
    plt.tight_layout()
    plt.savefig(f'SoS_{i}.png', dpi=1200, format = 'png')
    plt.show()
    
    plt.clf()
    plt.figure(figsize=(5,5))
    plt.scatter(unscaled_Z_true/1000000, unscaled_Z_pred/1000000, c ='orange', s=100, alpha=0.5, marker = "o")
    p1 = max(max(unscaled_Z_pred), max(unscaled_Z_true))/1000000
    p2 = min(min(unscaled_Z_pred), min(unscaled_Z_true))/1000000
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('Actual Values (MRayl)', fontsize=15)
    plt.ylabel('Predictions (MRayl)', fontsize=15)
    plt.suptitle('Acoustic Impedance', fontweight = 'bold', fontsize=20)
    plt.axis('equal')
    plt.title(f'MAPE = {MAPE_Z:.3f}, MAE = {errors_Z/1000000:.3f}, RMSE = {RMSE_Z/1000000:.3f}')
    plt.tight_layout()
    plt.savefig(f'Z_{i}.png', dpi=1200, format = 'png')
    plt.show()


torch.save(model.state_dict(), 'final_model.pth')

print(np.mean(validation_EM_MAPE), '+/-', np.std(validation_EM_MAPE))
print(np.mean(validation_EM_MAE), '+/-', np.std(validation_EM_MAE))
print(np.mean(validation_EM_RMSE), '+/-', np.std(validation_EM_RMSE))

print(np.mean(validation_size_MAPE), '+/-', np.std(validation_size_MAPE))
print(np.mean(validation_size_MAE), '+/-', np.std(validation_size_MAE))
print(np.mean(validation_size_RMSE),  '+/-', np.std(validation_size_RMSE))

print(np.mean(validation_c_MAPE),  '+/-', np.std(validation_c_MAPE))
print(np.mean(validation_c_MAE),  '+/-', np.std(validation_c_MAE))
print(np.mean(validation_c_RMSE),  '+/-', np.std(validation_c_RMSE))

print(np.mean(validation_Z_MAPE),  '+/-', np.std(validation_Z_MAPE))
print(np.mean(validation_Z_MAE)/1000000,  '+/-', np.std(validation_Z_MAE)/1000000)
print(np.mean(validation_Z_RMSE)/1000000,  '+/-', np.std(validation_Z_RMSE)/1000000)

