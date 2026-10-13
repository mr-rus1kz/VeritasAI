import torch
import torch.nn as nn
import torch.optim as optim

class EmptyNet(nn.Module):
    def __init__(self):
        super(EmptyNet, self).__init__()
        self.hidden = nn.Linear(100, 32685)
        self.rnn = nn.GRUCell(32685, 32685)

    def forward(self, x, h=None):
        x = torch.relu(self.hidden(x))
        h = self.rnn(x, h)
        return h, h

model = EmptyNet()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

inputs = torch.randn(10, 16, 100)
targets = torch.randn(10, 16, 32685)

model.train()
h = None

for step in range(10):
    optimizer.zero_grad()
    output, h = model(inputs[step], h)
    h = h.detach() 
    loss = criterion(output, targets[step])
    loss.backward()
    optimizer.step()

torch.save(model.state_dict(), "trained_model_32685.pth")