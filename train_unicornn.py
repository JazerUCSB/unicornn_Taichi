import torch
import numpy as np
from unicornn_taichi import forward_torch, backward_torch, initialize_params

class UnICORNN_Torch(torch.nn.Module):
    def __init__(self):
        super(UnICORNN_Torch, self).__init__()

    def forward(self, x):
        x_numpy = x.detach().cpu().numpy()  # Convert to NumPy
        y_numpy = forward_torch(x_numpy)  # Get output from Taichi

        # ✅ Ensure PyTorch tracks gradients
        y_tensor = torch.tensor(y_numpy, dtype=torch.float32, requires_grad=True).to(x.device)
        return y_tensor

    def backward_custom(self, grad):
        """Custom backward pass calling Taichi's BPTT."""
        grad_numpy = grad.detach().cpu().numpy()  # Convert grad to NumPy
        backward_torch(grad_numpy)  # Call Taichi backward function




def generate_data(batch_size=32, seq_len=100, hidden_size=128, window_size=1000, stride=250):
    """Generates streaming data using a rolling window approach."""
    t = torch.linspace(0, 10, window_size + stride * (seq_len - 1))  # Larger time range

    x_raw = torch.sin(t)  # Simulated sine wave signal
    y_raw = torch.cos(t)  # Target: shifted sine wave

    # Apply rolling window to create streaming sequences
    x = torch.stack([x_raw[i: i + window_size] for i in range(0, len(x_raw) - window_size, stride)])
    y = torch.stack([y_raw[i: i + window_size] for i in range(0, len(y_raw) - window_size, stride)])

    # Adjust batch dimensions
    x = x[:batch_size].unsqueeze(-1).repeat(1, 1, hidden_size)  # (batch, window, features)
    y = y[:batch_size].unsqueeze(-1).repeat(1, 1, hidden_size)  # (batch, window, features)

    print("🛠️ Generated Data Shape:", x.shape, y.shape)  # Debugging
    return x, y



# ✅ Training Pipeline
def train():
    # Initialize Taichi model parameters
    initialize_params()

    # Create PyTorch model
    model = UnICORNN_Torch()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to prevent explosion

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam([torch.tensor(np.random.randn(128), requires_grad=True)], lr=0.001)  # Reduce from 0.01 → 0.001

    EPOCHS = 10
    BATCH_SIZE = 32

    for epoch in range(EPOCHS):
        x_train, y_train = generate_data(BATCH_SIZE)

        optimizer.zero_grad()

        y_pred = model(x_train)  # Forward pass

        # Fix shape mismatch by transposing y_train
        y_train = y_train.permute(1, 0, 2)  # (32, 100, 128) → (100, 32, 128)

        # Debugging: Print final shapes before loss computation
        print("🔍 y_pred shape:", y_pred.shape)
        print("🔍 y_train shape (after permute):", y_train.shape)

        loss = criterion(y_pred, y_train)  # Compute loss
        loss.backward()  # PyTorch gradient tracking

        model.backward_custom(y_pred.grad)  # Call Taichi backward pass

        optimizer.step()

        print(f"✅ Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()
