exp_x_df = pd.read_excel('Experimental_Validation_features.xlsx', header=None)
exp_x_df = exp_x_df.T
exp_x_df.columns = exp_x_df.columns.astype(str)
exp_x_df.to_parquet('Experimental_Validation_features.parquet')

exp_y_df = pd.read_excel('Experimental_Validation_labels.xlsx', header=None)
exp_y_df = exp_y_df.rename(index={0: 'Concentration', 1: 'size' })
exp_y_df = exp_y_df.T
exp_y_df.columns = exp_y_df.columns.astype(str)
exp_y_df.to_parquet('Experimental_Validation_labels.parquet')


exp_x_df = pd.read_parquet('Experimental_Validation_features.parquet')
exp_y_df = pd.read_parquet('Experimental_Validation_labels.parquet')
conc = sorted(set(exp_y_df['Concentration']))
EM_Means = [
 0.08841666666666669,
 0.2313333333333333,
 0.32449124999999995,
 0.536375,
 0.7979938888888888]

EM_Means = [x * 1000 for x in EM_Means]

res = {conc[i]: EM_Means[i] for i in range(len(conc))}
 
exp_y_df['E'] = exp_y_df['Concentration'].map(res)

c_Means = [1460, 1472.5, 1485.5, 1491. , 1497.5]
res2 = {conc[i]: c_Means[i] for i in range(len(conc))}
exp_y_df['c'] = exp_y_df['Concentration'].map(res2)

scaler3 = MinMaxScaler()
scaler4 = MinMaxScaler()


velocity = 1450
window = 0.094
tot_points = exp_x_df.shape[1]

tot_time = window/velocity

sampling_rate = tot_time/tot_points

print(sampling_rate)

d = 0.040
end_point = int(((2*d)/velocity)/sampling_rate)

init_pulse = int(0.000003/sampling_rate)


exp_x_df_sub = exp_x_df.iloc[:, init_pulse:end_point-4000]

init_size = exp_x_df_sub.shape[1]
final_size = X_df_.shape[1]
remove_n = init_size-final_size
print(remove_n)

drop_indices = np.random.choice(exp_x_df_sub.T.index, remove_n, replace=False)
exp_x_df_sub2 = (exp_x_df_sub.T.drop(drop_indices)).T
print(exp_x_df_sub2.shape)

exp_x_df_Scaled = (pd.DataFrame(scaler3.fit_transform(exp_x_df_sub2.T))).T


exp_y_df['p'] = 1030
exp_y_df['Z'] = exp_y_df['c'] *1030
exp_y_df_sub = exp_y_df[['E', 'size', 'Z', 'c']]

exp_y_df_Scaled = pd.DataFrame(columns=[exp_y_df_sub.columns])

exp_y_df_Scaled = scaler4.fit_transform(exp_y_df_sub)

#turn data into tensor
x_test_true = torch.unsqueeze(torch.from_numpy(exp_x_df_Scaled.values).float(), 1)

y_test_true = torch.from_numpy(exp_y_df_Scaled).float()


test_size_RMSE = []
test_size_MAPE = []
test_size_MAE = []

test_EM_RMSE = []
test_EM_MAPE = []
test_EM_MAE = []

test_c_RMSE = []
test_c_MAPE = []
test_c_MAE = []

test_Z_RMSE = []
test_Z_MAPE = []
test_Z_MAE = []

model.eval()

with torch.no_grad():
# forward pass: compute predicted outputs by passing inputs to the model
    output_v = model(x_test_true)

y_test_true = y_test_true.detach().numpy()
y_test_pred = output_v.detach().numpy()    
    
unscaled_pred = scaler4.inverse_transform(y_test_pred)
unscaled_true = scaler4.inverse_transform(y_test_true)

#[['E', 'size', 'Z', 'c']]

unscaled_E_pred = unscaled_pred[:, 0]
unscaled_E_true = unscaled_true[:, 0]

unscaled_size_pred = unscaled_pred[:, 1]
unscaled_size_true = unscaled_true[:, 1]

unscaled_Z_pred = unscaled_pred[:, 2]
unscaled_Z_true = unscaled_true[:, 2]

unscaled_c_pred = unscaled_pred[:, 3]
unscaled_c_true = unscaled_true[:, 3]





# Calculate RMSE for val data
print("******* RMSE *******")
RMSE_EM = np.sqrt(mean_squared_error(unscaled_E_true, unscaled_E_pred))
print('Elastic Modulus RMSE = ', RMSE_EM)

RMSE_Size = np.sqrt(mean_squared_error(unscaled_size_true, unscaled_size_pred))
print('Size RMSE = ', RMSE_Size)

RMSE_Z = np.sqrt(mean_squared_error(unscaled_Z_true, unscaled_Z_pred))
print('Impedance RMSE = ', RMSE_Z)

RMSE_c = np.sqrt(mean_squared_error(unscaled_c_true, unscaled_c_pred))
print('SoS RMSE = ', RMSE_c)
print()


# Calculate MAPE for val data
print("******* MAPE *******")
MAPE_EM = mean_absolute_percentage_error(unscaled_E_true, unscaled_E_pred)
print('Elastic Modulus MAPE = ', MAPE_EM)

MAPE_Size = mean_absolute_percentage_error(unscaled_size_true, unscaled_size_pred)
print('Size MAPE = ', MAPE_Size)

MAPE_Z = mean_absolute_percentage_error(unscaled_Z_true, unscaled_Z_pred)
print('Impedance RMSE = ', MAPE_Z)

MAPE_c = mean_absolute_percentage_error(unscaled_c_true, unscaled_c_pred)
print('SoS RMSE = ', MAPE_c)
print()



# Calculate MAE for val data
print("******* MAE *******")
errors_EM = mean_absolute_error(unscaled_E_true, unscaled_E_pred)
print('Elastic Modulus MAE = ', errors_EM)

errors_Size = mean_absolute_error(unscaled_size_true, unscaled_size_pred)
print('Size MAE = ', errors_Size)

errors_Z = np.sqrt(mean_absolute_error(unscaled_Z_true, unscaled_Z_pred))
print('Impedance RMSE = ', errors_Z)

errors_c = np.sqrt(mean_absolute_error(unscaled_c_true, unscaled_c_pred))
print('SoS RMSE = ', errors_c)

test_size_RMSE.append(RMSE_Size)
test_size_MAPE.append(MAPE_Size)
test_size_MAE.append(errors_Size)

test_EM_RMSE.append(RMSE_EM)
test_EM_MAPE.append(MAPE_EM)
test_EM_MAE.append(errors_EM)

test_c_RMSE.append(RMSE_c)
test_c_MAPE.append(MAPE_c)
test_c_MAE.append(errors_c)

test_Z_RMSE.append(RMSE_Z)
test_Z_MAPE.append(MAPE_Z)
test_Z_MAE.append(errors_Z)



plt.clf()
plt.figure(figsize=(5,5))
plt.scatter(unscaled_E_true, unscaled_E_pred, c = 'darkviolet', s=100, alpha=0.5, marker = "o")
p1 = 10#max(max(unscaled_E_pred), max(unscaled_E_true))
p2 = 1000#min(min(unscaled_E_pred), min(unscaled_E_true))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual Values (kPa)', fontsize=15)
plt.ylabel('Predictions (kPa)', fontsize=15)
plt.suptitle('Elastic Modulus \n (Experimental Validation)', fontweight = 'bold', fontsize=20, x=0.55)
plt.title(f'MAPE = {MAPE_EM:.3f}, MAE = {errors_EM:.3f}, RMSE = {RMSE_EM:.3f}')
#plt.xticks(rotation=45)
plt.axis('equal')
plt.tight_layout()
plt.savefig('EM_Exp.png', dpi=1200, format = 'png')
plt.show()

plt.clf()
plt.figure(figsize=(5,5))
plt.scatter(unscaled_size_true, unscaled_size_pred, c = 'crimson', s=100, alpha=0.5, marker = "o")
p1 = max(max(unscaled_size_pred), max(unscaled_size_true))
p2 = min(min(unscaled_size_pred), min(unscaled_size_true))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual Values (mm)', fontsize=15)
plt.ylabel('Predictions (mm)', fontsize=15)
plt.suptitle('Size \n (Experimental Validation)', fontweight = 'bold', fontsize=20, x=0.55)
plt.axis('equal')
plt.title(f'MAPE = {MAPE_Size:.3f}, MAE = {errors_Size:.3f}, RMSE = {RMSE_Size:.3f}')
plt.tight_layout()
plt.savefig('Size_Exp.png', dpi=1200, format = 'png')
plt.show()

plt.clf()
plt.figure(figsize=(5,5))
plt.scatter(unscaled_c_true, unscaled_c_pred, c = 'green', s=100, alpha=0.5, marker = "o")
p1 = max(max(unscaled_c_pred), max(unscaled_c_true))
p2 = min(min(unscaled_c_pred), min(unscaled_c_true))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual Values (m/s)', fontsize=15)
plt.ylabel('Predictions (m/s)', fontsize=15)
plt.suptitle('Speed of Sound \n (Experimental Validation)', fontweight = 'bold', fontsize=20, x=0.55)
plt.axis('equal')
plt.title(f'MAPE = {MAPE_c:.3f}, MAE = {errors_c:.3f}, RMSE = {RMSE_c:.3f}')
plt.tight_layout()
plt.savefig('SoS_Exp.png', dpi=1200, format = 'png')
plt.show()

plt.clf()
plt.figure(figsize=(5,5))
plt.scatter(unscaled_Z_true/1000000, unscaled_Z_pred/1000000, c ='orange', s=100, alpha=0.5, marker = "o")
p1 = max(max(unscaled_Z_pred), max(unscaled_Z_true))/1000000
p2 = min(min(unscaled_Z_pred), min(unscaled_Z_true))/1000000
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('Actual Values (MRayl)', fontsize=15)
plt.ylabel('Predictions (MRayl)', fontsize=15)
plt.suptitle('Acoustic Impedance \n (Experimental Validation)', fontweight = 'bold', fontsize=20, x=0.55)
plt.axis('equal')
plt.title(f'MAPE = {MAPE_Z:.3f}, MAE = {errors_Z/1000000:.3f}, RMSE = {RMSE_Z/1000000:.3f}')
plt.tight_layout()
plt.savefig('Z_Exp.png', dpi=1200, format = 'png')
plt.show()