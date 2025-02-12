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
    


    