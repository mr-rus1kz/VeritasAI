import json
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import EmptyNet

if __name__ == "__main__":
    with open("config/model_config.json", "r") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    model = EmptyNet(
        input_dim=config["input_dim"], 
        hidden_dim=config["hidden_dim"],
        verify_steps=config["verify_steps"]
    )
    
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        model = torch.compile(model)
        
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    inputs = torch.randn(config["seq_len"], config["batch_size"], config["input_dim"], device=device)
    targets = torch.randn(config["seq_len"], config["batch_size"], config["hidden_dim"], device=device)

    model.train()
    h = None
    optimizer.zero_grad()
    total_loss = 0

    for step in range(config["seq_len"]):
        output, h = model(inputs[step], h)
        loss = criterion(output, targets[step])
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    torch.save(model.state_dict(), "trained_model_32685.pth")