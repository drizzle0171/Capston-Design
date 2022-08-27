import math
import torch
import torchcde
import numpy as np
import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import pandas as pd

class CDEFunc(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels):
        ######################
        # input_channels is the number of input channels in the data X. (Determined by the data.)
        # hidden_channels is the number of channels for z_t. (Determined by you!)
        ######################
        super(CDEFunc, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels

        self.linear1 = torch.nn.Linear(hidden_channels, 128)
        self.linear2 = torch.nn.Linear(128, input_channels * hidden_channels)

    ######################
    # For most purposes the t argument can probably be ignored; unless you want your CDE to behave differently at
    # different times, which would be unusual. But it's there if you need it!
    ######################
    def forward(self, t, z):
        # z has shape (batch, hidden_channels)
        z = self.linear1(z)
        z = z.relu()
        z = self.linear2(z)
        ######################
        # Easy-to-forget gotcha: Best results tend to be obtained by adding a final tanh nonlinearity.
        ######################
        z = z.tanh()
        ######################
        # Ignoring the batch dimension, the shape of the output tensor must be a matrix,
        # because we need it to represent a linear map from R^input_channels to R^hidden_channels.
        ######################
        z = z.view(z.size(0), self.hidden_channels, self.input_channels)
        return z


######################
# Next, we need to package CDEFunc up into a model that computes the integral.
######################
class NeuralCDE(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, interpolation="cubic"):
        super(NeuralCDE, self).__init__()

        self.func = CDEFunc(input_channels, hidden_channels)
        self.initial = torch.nn.Linear(input_channels, hidden_channels)
        self.readout = torch.nn.Linear(hidden_channels, output_channels)
        self.interpolation = interpolation

    def forward(self, coeffs):
        if self.interpolation == 'cubic':
            X = torchcde.CubicSpline(coeffs)
        elif self.interpolation == 'linear':
            X = torchcde.LinearInterpolation(coeffs)
        else:
            raise ValueError("Only 'linear' and 'cubic' interpolation methods are implemented.")

        ######################
        # Easy to forget gotcha: Initial hidden state should be a function of the first observation.
        ######################
        X0 = X.evaluate(X.interval[0])
        z0 = self.initial(X0)

        ######################
        # Actually solve the CDE.
        ######################
        z_T = torchcde.cdeint(X=X,
                              z0=z0,
                              func=self.func,
                              t=X.interval)

        ######################
        # Both the initial value and the terminal value are returned from cdeint; extract just the terminal value,
        # and then apply a linear map.
        ######################
        z_T = z_T[:, 1]
        pred_y = self.readout(z_T)
        return pred_y



def main():
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else 'cpu')

    x = np.load('/nas/home/carlos/time_series/data/x_total.npy')
    y = np.load('/nas/home/carlos/time_series/data/y_total.npy')

    x = x[:5000]
    y = y[:5000]

    max_x = np.max(x)
    min_x = np.min(x)

    x = (x - min_x) / (max_x - min_x)
    y = (y - min_x) / (max_x - min_x)
    with open('/nas/datahub/mirae/datatime_index.pkl', 'rb') as f:
        time = pickle.load(f)

    _, time_idx, _, _ = train_test_split(list(range(len(x))), y, test_size=.1, random_state=42, shuffle=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.1, random_state=42, shuffle=True)
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.1, random_state=42, shuffle=True)

    train_X, train_y = torch.from_numpy(X_train).float().unsqueeze(-1).to(device), torch.from_numpy(y_train).float().to(device)
    val_X, val_y = torch.from_numpy(X_val).float().unsqueeze(-1).to(device), torch.from_numpy(y_val).float().to(device)

    model = NeuralCDE(input_channels=1, hidden_channels=32, output_channels=5).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)
    val_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(val_X)

    train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=256)

    val_dataset = torch.utils.data.TensorDataset(val_coeffs, val_y)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=256)
   
    num_epochs = 2
    model.train()
    print("Begin Training")

    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm.tqdm(train_dataloader):
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            #print(f"loss pred y:{pred_y.shape}, batch_y: {batch_y.shape}")
            loss = torch.nn.functional.mse_loss(pred_y, batch_y).to(device)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
        loss = epoch_loss/len(train_dataloader)
        print('Epoch: {}   Training loss: {}'.format(epoch, loss))

    
        model.eval()
        val_loss = 0
        total_loss = []
        output_list = []
        for batch in val_dataloader:
            batch_coeffs, batch_y = batch
            pred_y = model(batch_coeffs).squeeze(-1)
            #print(f"loss pred y:{pred_y.shape}, batch_y: {batch_y.shape}")
            loss = torch.nn.functional.mse_loss(pred_y, batch_y).to(device)
            total_loss.append(loss)
            val_loss += loss.mean().item()
            output_list.append(pred_y) 
        print('Epoch: {}   Validation loss: {}'.format(epoch, val_loss/len(val_dataloader)))

    torch.save(model, 'ncde.pt')
    total_loss = torch.cat(total_loss, dim=0).sum(dim=1).squeeze()
    top_5_ind = torch.topk(total_loss, 5)[1]
    bottom_5_ind = torch.topk(-total_loss, 5)[1]
    output_tensor = torch.cat(output_list, dim=0)

    inputs = x_test
    labels = y_test
    outputs = output_tensor * (train_dataset.max_x - train_dataset.min_x) + train_dataset.min_x
    plt.rcParams["figure.figsize"] = (16,6)

    for i in range(5):
        idx = bottom_5_ind[i]
        inputs_idx = inputs[idx].squeeze(1)
        labels_idx = labels[idx].squeeze(1)
        outputs_idx = outputs[idx].squeeze(1).cpu()
        input_time = time[time_idx[idx]:time_idx[idx]+60]
        label_time = time[time_idx[idx]+60:time_idx[idx]+65]
        plt.clf()
        plt.plot(input_time, inputs_idx, 'b', label='x')
        plt.plot(label_time, labels_idx, 'bo', ms=5, alpha=0.7, label='y')
        plt.plot(label_time, outputs_idx, 'ro', ms=5, alpha=0.7, label='y_hat')
        plt.title(f'Result of Top {i+1} (min): {total_loss[idx]:.5f}')
        plt.legend()
        plt.savefig(f'./result_ode/result_min_{i+1}_{idx}.png')

    # Max Top5
    for i in range(5):
        idx = top_5_ind[i]
        inputs_idx = inputs[idx].squeeze(1)
        labels_idx = labels[idx].squeeze(1)
        outputs_idx = outputs[idx].squeeze(1).cpu()
        input_time = time[time_idx[idx]:time_idx[idx]+60]
        label_time = time[time_idx[idx]+60:time_idx[idx]+65]
        plt.clf()
        plt.plot(input_time, inputs_idx, 'b', label='x')
        plt.plot(label_time, labels_idx, 'bo', ms=5, alpha=0.7, label='y')
        plt.plot(label_time, outputs_idx, 'ro', ms=5, alpha=0.7, label='y_hat')
        plt.title(f'Result of Top {i+1} (max): {total_loss[idx]:.5f}')
        plt.legend()
        plt.savefig(f'./result_ode/result_max_{i+1}_{idx}.png')

if __name__ == '__main__':
    main()
