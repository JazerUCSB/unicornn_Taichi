import taichi as ti

ti.init(arch=ti.gpu)  

len_seq = 100     
batch = 32         
hidden_size = 128  

ncols = batch * hidden_size  

weight_hh = ti.field(dtype=ti.f32, shape=(hidden_size,))  
c = ti.field(dtype=ti.f32, shape=(hidden_size,))          

hy_initial = ti.field(dtype=ti.f32, shape=(batch, hidden_size))  
hz_initial = ti.field(dtype=ti.f32, shape=(batch, hidden_size))  

hy_final = ti.field(dtype=ti.f32, shape=(batch, hidden_size))  
hz_final = ti.field(dtype=ti.f32, shape=(batch, hidden_size))  

fwd_x = ti.field(dtype=ti.f32, shape=(len_seq, batch, hidden_size))   
fwd_hy_all = ti.field(dtype=ti.f32, shape=(len_seq, batch, hidden_size))  

grad_h = ti.field(dtype=ti.f32, shape=(len_seq, batch, hidden_size)) 
grad_x = ti.field(dtype=ti.f32, shape=(len_seq, batch, hidden_size))  
grad_weight_hh = ti.field(dtype=ti.f32, shape=(hidden_size,))  
grad_c = ti.field(dtype=ti.f32, shape=(hidden_size,))

@ti.func
def sigmoid(x):
    return 1.0 / (1.0 + ti.exp(-x))

@ti.func
def sigmoid_grad(x):
    ti.exp(-x)/((1.+ti.exp(-x))*(1.+ti.exp(-x)))

@ti.func
def activation(x):
    ti.tanh(x)

@ti.func
def ti_cosh(x):
    return (ti.exp(x) + ti.exp(-x)) / 2.0

@ti.func
def calc_grad_activation(x):
    1/(ti_cosh(x)*ti_cosh(x))

@ti.kernel
def unicornn_fwd(dt: ti.f32, alpha: ti.f32):
    for col in range(ncols):  
        i = col // hidden_size  
        j = col % hidden_size   

        weight_hh_cur = weight_hh[j] 
        c_cur = c[j]  

        hy = hy_initial[i, j]  
        hz = hz_initial[i, j]  

        for row in range(len_seq):  
            hz -= dt * sigmoid(c_cur) * (activation(hy * weight_hh_cur + fwd_x[row, i, j]) + alpha * hy)
            hy += dt * sigmoid(c_cur) * hz

            fwd_hy_all[row, i, j] = hy 

        hy_final[i, j] = hy  
        hz_final[i, j] = hz  
    

@ti.kernel
def unicornn_bwd(dt: ti.f32, alpha: ti.f32):
    for col in range(ncols):  
        i = col // hidden_size  
        j = col % hidden_size   

        weight_hh_cur = weight_hh[j]  
        c_cur = c[j]  

        gweight_hh = 0.0
        gc = 0.0
        delta_z = 0.0
        delta_dt = 0.0

        hy = hy_final[i, j]  
        hz = hz_final[i, j]  
        delta_y = grad_h[len_seq - 1, i, j]  

        
        for row in range(len_seq - 1, -1, -1):
            delta_dt = delta_y * dt * sigmoid_grad(c_cur) * hz  

            hy -= dt * sigmoid(c_cur) * hz
            hz += dt * sigmoid(c_cur) * (activation(hy * weight_hh_cur + fwd_x[row, i, j]) + alpha * hy)

            delta_z += delta_y * dt * sigmoid(c_cur)

            gweight_hh -= delta_z * dt * sigmoid(c_cur) * calc_grad_activation(hy * weight_hh_cur + fwd_x[row, i, j]) * hy
            gc += delta_dt - delta_z * (dt * sigmoid_grad(c_cur) * (activation(hy * weight_hh_cur + fwd_x[row, i, j]) + alpha * hy))

            grad_x[row, i, j] = -delta_z * dt * sigmoid(c_cur) * calc_grad_activation(hy * weight_hh_cur + fwd_x[row, i, j])

           
            if row > 0:
                delta_y += -delta_z * dt * sigmoid(c_cur) * (
                    calc_grad_activation(hy * weight_hh_cur + fwd_x[row, i, j]) * weight_hh_cur + alpha
                ) + grad_h[row - 1, i, j]

        
        for k in range(hidden_size):  
            if j == k:
                grad_weight_hh[j] += gweight_hh
                grad_c[j] += gc
    