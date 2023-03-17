import random
import torch
import torch.nn as nn
from torchdiffeq import odeint

class lstm_encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)

    def forward(self, x_input):
        output, (cell, hidden) = self.lstm(x_input)
        return cell, hidden

class lstm_decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers = 1):
        super(lstm_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size,num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)   

    def forward(self, x_input, cell, hidden):
        y_hat, (cell, hidden) = self.lstm(x_input.unsqueeze(1), (cell, hidden))
        output = self.linear(y_hat.squeeze(0))
        return output, cell, hidden

class lstm_encoder_decoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_encoder_decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.decoder = lstm_decoder(input_size = input_size, hidden_size = hidden_size)

    def forward(self, x_input, targets, target_len, teacher_forcing_ratio):
        batch_size = x_input.shape[0]
        input_size = x_input.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size).cuda()
        cell, hidden = self.encoder(x_input)
        decoder_input = x_input[:,-1, :]

        #원하는 길이가 될 때까지 decoder를 실행한다.
        for t in range(target_len): 
            out, cell, hidden = self.decoder(decoder_input, cell, hidden)
            out = out.squeeze(1)
            # teacher forcing을 구현한다.
            # teacher forcing에 해당하면 다음 인풋값으로는 예측한 값이 아니라 실제 값을 사용한다.
            if random.random() < teacher_forcing_ratio:
                decoder_input = targets[:, t, :]
            else:
                decoder_input = out
            outputs[:,t,:] = out
        return outputs
	
    # 편의성을 위해 예측해주는 함수도 생성한다.
    def predict(self, x_input, target_len):
        batch_size = x_input.shape[0]
        input_size = x_input.shape[2]
        outputs = torch.zeros(batch_size, target_len, input_size)
        cell, hidden = self.encoder(x_input)
        decoder_input = x_input[:,-1, :]

        for t in range(target_len):
            out, cell, hidden = self.decoder(decoder_input, cell, hidden)
            out =  out.squeeze(1)
            decoder_input = out
            outputs[:,t,:] = out
        return outputs

class latent_odefunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20):
        super(latent_odefunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out

class mlp_decoder(nn.Module):

    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20):
        super(mlp_decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class lstm_ode_model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(lstm_ode_model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = lstm_encoder(input_size = input_size, hidden_size = hidden_size)
        self.ode = latent_odefunc(latent_dim = hidden_size, nhidden = 2*hidden_size)
        self.decoder = mlp_decoder(latent_dim = hidden_size, obs_dim = input_size, nhidden = 2*hidden_size)

    def forward(self, inputs, targets, target_len):
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]

        _, hidden = self.encoder(inputs)
        ode_input = hidden[0][0]

        time_steps = [5/60. * i for i in range(5)]
        pred_z = odeint(self.ode, ode_input, torch.tensor(time_steps).cuda()).permute(1, 0, 2)
        
        outputs = self.decoder(pred_z)
        return outputs
	
    # 편의성을 위해 예측해주는 함수도 생성한다.
    def predict(self, inputs, target_len):
        self.eval()
        batch_size = inputs.shape[0]
        input_size = inputs.shape[2]
        
        outputs = torch.zeros(batch_size, target_len, input_size)
        cell, hidden = self.encoder(inputs)
        ode_input = hidden[0][0]

        time_steps = [5/60. * i for i in range(5)]
        pred_z = odeint(self.ode, ode_input, torch.tensor(time_steps).cuda()).permute(1, 0, 2)
        outputs = self.decoder(pred_z)
        return outputs.detach().numpy()
