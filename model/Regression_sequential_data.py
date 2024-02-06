from torch import nn

class RegressionRNN(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # number of features
        self.hidden_units = hidden_units
        self.num_layers = 2
        
        self.RNN = nn.RNN(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=0.2
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(x.device)
        
        output, hn = self.RNN(x, h0)
        out = self.linear(hn[0]).flatten()
        
        out = torch.sigmoid(out)
        
        out = torch.stack((out, 1 - out), dim=1)
        out = out.flatten()

        return out
    
    

class RegressionGRU(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # number of features
        self.hidden_units = hidden_units
        self.num_layers = 2
        
        self.GRU = nn.GRU(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=0.2
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(x.device)
        
        output, hn = self.GRU(x, h0)
        out = self.linear(hn[0]).flatten()
        out = torch.sigmoid(out)
        
        out = torch.stack((out, 1 - out), dim=1)
        out = out.flatten()

        return out
    
    
    
    
class RegressionLSTM(nn.Module):
    def __init__(self, num_sensors, hidden_units):
        super().__init__()
        self.num_sensors = num_sensors  # number of features
        self.hidden_units = hidden_units
        self.num_layers = 2

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=0.2
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_().to(x.device)
        
        output, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()
        out = torch.sigmoid(out)
        
        out = torch.stack((out, 1 - out), dim=1)
        out = out.flatten()

        return out