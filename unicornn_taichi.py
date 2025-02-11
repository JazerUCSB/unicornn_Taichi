import taichi as ti
import numpy as np

# Initialize Taichi
# Change to `ti.cpu` if no GPU available
ti.init(arch=ti.gpu)

# Define dimensions
BATCH = 32
SEQ_LEN = 100
HIDDEN_SIZE = 128

# Taichi Fields (Tensors)
x = ti.field(dtype=ti.f32, shape=(SEQ_LEN, BATCH, HIDDEN_SIZE))  # Input
hy = ti.field(dtype=ti.f32, shape=(BATCH, HIDDEN_SIZE))  # Hidden state
hz = ti.field(dtype=ti.f32, shape=(BATCH, HIDDEN_SIZE))  # Auxiliary state
weight_hh = ti.field(dtype=ti.f32, shape=HIDDEN_SIZE)  # Recurrent weights
weight_linear = ti.field(dtype=ti.f32, shape=(HIDDEN_SIZE, HIDDEN_SIZE))  # Linear transformation weights
c = ti.field(dtype=ti.f32, shape=HIDDEN_SIZE)  # Scaling coefficient
alpha = ti.field(dtype=ti.f32, shape=())  # Alpha parameter
dt = ti.field(dtype=ti.f32, shape=())  # Time step

# Sigmoid function
@ti.func
def sigmoid(x):
    return 1.0 / (1.0 + ti.exp(-x))

@ti.func
def sigmoid_grad(x):
    return ti.exp(-x) / ((1.0 + ti.exp(-x)) * (1.0 + ti.exp(-x)))

# Linear transformation layer
@ti.kernel
def apply_linear_transform(x_in: ti.types.ndarray(), x_out: ti.types.ndarray()):
    for i, j in ti.ndrange(HIDDEN_SIZE, HIDDEN_SIZE):
        x_out[i] += x_in[j] * weight_linear[j, i]

# 🔄 Forward Pass
@ti.kernel
def forward():
    for col in range(BATCH * HIDDEN_SIZE):
        b = col // HIDDEN_SIZE
        h = col % HIDDEN_SIZE

        weight = weight_hh[h]
        c_val = c[h]
        hy_t = hy[b, h]
        hz_t = hz[b, h]

        for t in range(SEQ_LEN):
            sigmoid_c = sigmoid(c_val)
            tanh_term = ti.tanh(hy_t * weight + x[t, b, h])

            hz_t -= dt[None] * sigmoid_c * (tanh_term + alpha[None] * hy_t)
            hy_t += dt[None] * sigmoid_c * hz_t
            x[t, b, h] = hy_t  # Store hidden state

        hy[b, h] = hy_t
        hz[b, h] = hz_t

# 🔄 Backpropagation Through Time (BPTT)
@ti.kernel
def backward(grad_h: ti.types.ndarray()):
    for col in range(BATCH * HIDDEN_SIZE):
        b = col // HIDDEN_SIZE
        h = col % HIDDEN_SIZE

        weight = weight_hh[h]
        c_val = c[h]

        gweight_hh = 0.0
        gc = 0.0
        delta_y = grad_h[SEQ_LEN - 1, b, h]
        delta_z = 0.0

        hy_t = hy[b, h]
        hz_t = hz[b, h]

        for i in range(SEQ_LEN):
            t = SEQ_LEN - 1 - i  # Reverse index

            delta_dt = delta_y * dt[None] * sigmoid_grad(c_val) * hz_t
            hy_t -= dt[None] * sigmoid(c_val) * hz_t
            hz_t += dt[None] * sigmoid(c_val) * (ti.tanh(hy_t * weight + x[t, b, h]) + alpha[None] * hy_t)

            delta_z += delta_y * dt[None] * sigmoid(c_val)

            cosh_x = (ti.exp(hy_t * weight + x[t, b, h]) + ti.exp(-(hy_t * weight + x[t, b, h]))) / 2
            gweight_hh -= delta_z * dt[None] * sigmoid(c_val) * (1 / cosh_x ** 2) * hy_t

            gc += delta_dt - delta_z * (dt[None] * sigmoid_grad(c_val) * (ti.tanh(hy_t * weight + x[t, b, h]) + alpha[None] * hy_t))

        weight_hh[h] += gweight_hh
        c[h] += gc

# Converts NumPy to Taichi and processes the sequence in chunks
def forward_torch(x_numpy):
    num_windows = x_numpy.shape[1] // SEQ_LEN  # Split into 100-length chunks
    outputs = []

    for i in range(num_windows):
        x_window = x_numpy[:, i * SEQ_LEN:(i + 1) * SEQ_LEN, :]
        x_window = x_window.reshape((SEQ_LEN, BATCH, HIDDEN_SIZE))

        x_transformed = np.zeros_like(x_window)  # Placeholder for linear output
        apply_linear_transform(x_window, x_transformed)  # Apply Taichi linear transformation

        x.from_numpy(x_transformed)  # Copy into Taichi
        forward()
        outputs.append(x.to_numpy())

    return np.concatenate(outputs, axis=0)  # Reassemble output windows

# Run backward pass using NumPy gradient from PyTorch
def backward_torch(grad_numpy):
    backward(grad_numpy)

# Initialize parameters
def initialize_params():
    weight_hh.from_numpy(np.random.randn(HIDDEN_SIZE).astype(np.float32) * 0.1)  # Scale down init weights
    weight_linear.from_numpy(np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE).astype(np.float32) * 0.1)
    c.from_numpy(np.random.randn(HIDDEN_SIZE).astype(np.float32) * 0.1)
    alpha[None] = 0.1
    dt[None] = 0.01

if __name__ == "__main__":
    initialize_params()
    print("✅ Taichi UnICORNN Module Loaded Successfully!")
