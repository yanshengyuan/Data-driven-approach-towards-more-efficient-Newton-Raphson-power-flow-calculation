import pandapower as pp
import numpy as np
 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
 
from sklearn.preprocessing import StandardScaler # normalize input features and target values
from sklearn.model_selection import ParameterGrid # for hyperparameter tuning
 
import pickle
import pandas as pd
import torch.optim as optim

supervised=0
semisupervised=0
unsupervised=1

data = np.load('./vector_data.npy')
print(data.shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DeepNN(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size):
        super(DeepNN, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))
        self.net = nn.Sequential(*layers)
 
    def forward(self, x):
        return self.net(x)
    
def physics_loss(P, Q, G, B, V, Phi):
    Phi = torch.deg2rad(Phi)
    p1 = torch.tensor(0.0, dtype=torch.float64).to(device)
    p2 = torch.tensor(0.0, dtype=torch.float64).to(device)
    q1 = torch.tensor(0.0, dtype=torch.float64).to(device)
    q2 = torch.tensor(0.0, dtype=torch.float64).to(device)
    for k in range(len(P)):
        p1 += V[0] * V[k] * ( G[0, k] * torch.cos(Phi[0] - Phi[k]) + B[0, k] * torch.sin(Phi[0] - Phi[k]) )
        p2 += V[1] * V[k] * ( G[1, k] * torch.cos(Phi[1] - Phi[k]) + B[1, k] * torch.sin(Phi[1] - Phi[k]) )
        q1 += V[0] * V[k] * ( G[0, k] * torch.sin(Phi[0] - Phi[k]) - B[0, k] * torch.cos(Phi[0] - Phi[k]) )
        q2 += V[1] * V[k] * ( G[1, k] * torch.cos(Phi[1] - Phi[k]) + B[1, k] * torch.sin(Phi[1] - Phi[k]) )
    res_p1 = torch.abs(p1 - P[0])
    res_p2 = torch.abs(p2 - P[1])
    res_q1 = torch.abs(q1 - Q[0])
    res_q2 = torch.abs(q2 - Q[1])
    
    return res_p1+res_p2+res_q1+res_q2

# Training setup
EPOCHS = 50
BATCH_SIZE = 1
 
# add dataset here
features = torch.tensor(data[:,:12], dtype=torch.float32)
labels = torch.tensor(data[:,16:20], dtype=torch.float32)
dataset = TensorDataset(features, labels)
 
# dataloaders
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

alpha = 0.00001
learning_rate = 0.001

# model, loss, optimizer
input_size = len(dataset[0][0])
output_size = len(dataset[0][1])
model = DeepNN(input_size=input_size, hidden_layers=[64,64], output_size=output_size).to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# training loop
best_val_loss = np.inf

for epoch in range(EPOCHS):
    model.train()
    for X,Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)
        outputs = model(X)
        
        if(supervised==1):
            loss = criterion(outputs, Y)
        if(semisupervised==1):
            loss = 10*criterion(outputs, Y)
            for i in range(len(X)):
                P = X[i, :2]
                Q = X[i, 2:4]
                G = X[i, 4:8].reshape(2, 2)
                B = X[i, 8:12].reshape(2, 2)
                V = outputs[i, :2]
                Phi = outputs[i, 2:4]
                loss += alpha*physics_loss(P, Q, G, B, V, Phi)
        if(unsupervised==1):
            P = X[0, :2]
            Q = X[0, 2:4]
            G = X[0, 4:8].reshape(2, 2)
            B = X[0, 8:12].reshape(2, 2)
            V = outputs[0, :2]
            Phi = outputs[0, 2:4]
            loss = physics_loss(P, Q, G, B, V, Phi)
            for i in range(1, len(X)):
                P = X[i, :2]
                Q = X[i, 2:4]
                G = X[i, 4:8].reshape(2, 2)
                B = X[i, 8:12].reshape(2, 2)
                V = outputs[i, :2]
                Phi = outputs[i, 2:4]
                loss += physics_loss(P, Q, G, B, V, Phi)
        
        optimizer.zero_grad()
        loss.backward()
        #print(loss.item())
        optimizer.step()
    scheduler.step()
 
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X,Y in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            outputs = model(X)
            loss_ = criterion(outputs, Y)
            val_loss += loss_.item()
    val_loss /= len(val_loader)
    #print(val_loss)
 
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), './best_model.pth')
        print(f"------------- BEST MODEL - Epoch {epoch} - Val Loss {val_loss:.6f} -------------")
 
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03}, Loss: {loss.item():.6f}, Val Loss: {val_loss:.6f}")

class SimpleTwoBus:
    def __init__(self, V_ext, P, Q, G, B, V_init, theta_init):
        '''This class creates a simple 2-bus network.'''
        self.V_ext = V_ext
        self.P = P
        self.Q = Q
        self.G = G
        self.B = B
        self.V_init = V_init
        self.theta_init = theta_init
        self.net = pp.create_empty_network()
        self.create_two_bus_grid()
 
    def create_two_bus_grid(self):
        # Create two buses with initialized voltage and angle
        bus1 = pp.create_bus(self.net, vn_kv=20.0, name="Bus 1")
        bus2 = pp.create_bus(self.net, vn_kv=0.4, name="Bus 2")
   
        # Initialize voltage and angle for buses
        self.net.bus.loc[bus1, 'vm_pu'] = self.V_init[0]
        self.net.bus.loc[bus1, 'va_degree'] = self.theta_init[0]
        self.net.bus.loc[bus2, 'vm_pu'] = self.V_init[1]
        self.net.bus.loc[bus2, 'va_degree'] = self.theta_init[1]
   
        # create a line between the two buses
        pp.create_line_from_parameters(
            self.net,
            from_bus=0,
            to_bus=1,
            length_km=1.0,
            r_ohm_per_km=1/self.G,
            x_ohm_per_km=1/self.B,
            c_nf_per_km=0.0,
            g_us_per_km=0.0,
            max_i_ka=100.0,
        )
 
        # Create a transformer between the two buses
        # pp.create_transformer(self.net, bus1, bus2, std_type="0.25 MVA 20/0.4 kV")
   
        # Create a load at bus 2 with specified P and Q
        pp.create_load(self.net, bus2, p_mw=self.P, q_mvar=self.Q, name="Load")
   
        # Create an external grid connection at bus 1 with specified G and B
        pp.create_ext_grid(self.net, bus1, vm_pu=self.V_ext, name="Grid Connection")
        
VM_PU_RANGE = [0.9, 1.1]  #
P_MW_RANGE = [0.0, 0.2]  #
Q_MVAR_RANGE = [0.0, 0.1]  #
G_RANGE = [80, 120]  #
B_RANGE = [0.01, 0.2]  #
INIT_VM_PU_MIN = 0.9  #
INIT_VM_PU_MAX = 1.1  #
INIT_THETA_MAX = -1
INIT_THETA_MIN = 1

maxrun = 100

iterations_list=[]
maev_list=[]
maePhi_list=[]
residual_list=[]

iterations_list_zero=[]
maev_list_zero=[]
maePhi_list_zero=[]
residual_list_zero=[]

for runcount in range(maxrun):
    V_ext = np.random.uniform(VM_PU_RANGE[0], VM_PU_RANGE[1])
    P = np.random.uniform(P_MW_RANGE[0], P_MW_RANGE[1])
    Q = np.random.uniform(Q_MVAR_RANGE[0], Q_MVAR_RANGE[1])
    G = np.random.uniform(G_RANGE[0], G_RANGE[1])  # Short-circuit power in MVA
    B = np.random.uniform(B_RANGE[0], B_RANGE[1])  # Short-circuit impedance
     
    V_init = [
        np.random.uniform(INIT_VM_PU_MIN, INIT_VM_PU_MAX),
        np.random.uniform(INIT_VM_PU_MIN, INIT_VM_PU_MAX),
    ]
    theta_init = [
        np.random.uniform(INIT_THETA_MIN, INIT_THETA_MAX),
        np.random.uniform(INIT_THETA_MIN, INIT_THETA_MAX),
    ]
     
    Net = SimpleTwoBus(V_ext,P,Q,G,B,V_init,theta_init)
    net = Net.net
    pp.runpp(net, max_iteration=1, tolerance_mva=np.inf)
     
    # Prepare input
    Ybus = net._ppc["internal"]["Ybus"].toarray()
    S = net._ppc["internal"]["Sbus"]
    input_tensor = torch.FloatTensor(np.concatenate([
        S.real,
        S.imag,
        Ybus.real.flatten(),
        Ybus.imag.flatten(),
    ]))
     
    # Load the best model
    best_model = DeepNN(input_size=input_size, hidden_layers=[64,64], output_size=output_size)
    best_model.load_state_dict(torch.load('./best_model.pth'))
    best_model.eval()
     
    # Get prediction
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
     
    # Split prediction into voltage magnitudes and angles
    n_buses = len(net.bus)
    V_mag_pred = output[:n_buses].cpu().numpy()
    V_ang_pred = output[n_buses:].cpu().numpy()

    def compute_residual(V_mag, V_ang, Ybus, S):
        #print(V_ang)
        V_ang = np.deg2rad(V_ang)
        complex_v = V_mag*(np.exp(V_ang*1j))
        current = Ybus@complex_v
        diag_V = np.diag(complex_v)
        residual = diag_V@np.conj(current) - S
        
        return residual[1:]

    # Run power flow to ensure internal data is available
    # pp.runpp(net, calculate_voltage_angles=True)
    net_refine = net.deepcopy()
    net_zero = net.deepcopy()
    
    pp.runpp(net,
             init="auto",
             init_vm_pu=V_mag_pred,
             init_va_degree=V_ang_pred,
             max_iteration=50,
             tolerance_mva=1e-5)
     
    # Get reference values from the network
    V_mag_ref = net.res_bus.vm_pu.values
    V_ang_ref = net.res_bus.va_degree.values
    V_ang_ref_rad = np.deg2rad(V_ang_ref)
     
    print("Predicted voltage magnitudes:", V_mag_pred)
    print("Predicted voltage angles:", V_ang_pred)
    print("Reference voltage magnitudes:", V_mag_ref)
    print("Reference voltage angles:", V_ang_ref_rad)
    #print("Iterations:", net._ppc["iterations"])
    maev = np.abs(V_mag_pred - V_mag_ref).mean()
    maePhi = np.abs(V_ang_pred - V_ang_ref).mean()
    print("MAE(||V||): ", np.abs(V_mag_pred - V_mag_ref).mean())
    print("MAE(Phi): ", np.abs(V_ang_pred - V_ang_ref).mean())

    pp.runpp(net_refine,
             init="auto",
             init_vm_pu=V_mag_pred,
             init_va_degree=V_ang_pred,
             max_iteration=50,
             tolerance_mva=1e-5)

    iterations = net_refine._ppc["iterations"]
    # print(f"Sample {_}: Converged in {iterations} iterations")

    # Extract ill-conditioned solution
    Ybus = net_refine._ppc["internal"]["Ybus"].toarray()
    S = net_refine._ppc["internal"]["Sbus"]
    it = net_refine._ppc["iterations"]
    V_mag = net_refine.res_bus.vm_pu.values
    V_ang = net_refine.res_bus.va_degree.values
    V_ang_rad = np.deg2rad(V_ang)
    resd = compute_residual(V_mag, V_ang, Ybus, S)

    print("Refined voltage magnitudes: ", V_mag)
    print("Refined voltage angles rad: ", V_ang_rad)
    print("Newton-Raphson converges at iteration ", it)
    print("Refined residual error: ", resd)
    
    iterations_list.append(it)
    maev_list.append(maev)
    maePhi_list.append(maePhi)
    residual_list.append(resd)
    
    try:
        pp.runpp(net_zero,
                 init="auto",
                 init_vm_pu=0.0,
                 init_va_degree=0.0,
                 max_iteration=50,
                 tolerance_mva=1e-5)
    except Exception as e:
        iterations_list_zero.append(float('inf'))
        maev_list_zero.append(float('inf'))
        maePhi_list_zero.append(float('inf'))
        residual_list_zero.append(float('inf'))
        print(f"Newton-Raphson failed to converge!")
        continue

    iterations = net_zero._ppc["iterations"]
    # print(f"Sample {_}: Converged in {iterations} iterations")

    # Extract ill-conditioned solution
    Ybus = net_zero._ppc["internal"]["Ybus"].toarray()
    S = net_zero._ppc["internal"]["Sbus"]
    it = net_zero._ppc["iterations"]
    V_mag = net_zero.res_bus.vm_pu.values
    V_ang = net_zero.res_bus.va_degree.values
    V_ang_rad = np.deg2rad(V_ang)
    resd = compute_residual(V_mag, V_ang, Ybus, S)

    print("zero voltage magnitudes: ", V_mag)
    print("zero voltage angles rad: ", V_ang_rad)
    print("Newton-Raphson converges at iteration ", it)
    print("zero residual error: ", resd)

    iterations_list_zero.append(it)
    maev_list_zero.append(maev)
    maePhi_list_zero.append(maePhi)
    residual_list_zero.append(resd)

iterations_record = np.array(iterations_list)
maev_record = np.array(maev_list)
maePhi_record = np.array(maePhi_list)
residual_record = np.array(residual_list)
np.save("Test_sample_N-R_iterations.npy", iterations_record)

print('\n')
print('\n')
print('Smart initialization!!')
print("average iterations:", iterations_record.mean())
print("average MAE(||V||):", maev_record.mean())
print("average MAE(Phi):", maePhi_record.mean())
print("average PF equation residual:", residual_record.mean())

iterations_record = np.array(iterations_list_zero)
maev_record = np.array(maev_list_zero)
maePhi_record = np.array(maePhi_list_zero)
residual_record = np.array(residual_list_zero)
np.save("Test_sample_N-R_iterations_zero.npy", iterations_record)

print('\n')
print('\n')
print('Zero initialization!!')
print("average iterations:", iterations_record.mean())
print("average MAE(||V||):", maev_record.mean())
print("average MAE(Phi):", maePhi_record.mean())
print("average PF equation residual:", residual_record.mean())