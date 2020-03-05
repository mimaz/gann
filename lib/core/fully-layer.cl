__kernel void reduce_inputs (__global const float *input,
                                      __global const float *weight,
                                      __global float *value) {
__local float local_sum[INPUTS];
int gid, lid, wid, off;

lid = get_local_id (0);
gid = get_global_id (0);
wid = get_group_id (0);

local_sum[lid] = input[lid] * weight[gid];

for(off = get_local_size (0) / 2; off > 0; off /= 2) {
barrier(CLK_LOCAL_MEM_FENCE);

if (lid < off) {
local_sum[lid] += local_sum[lid + off];
}
}

if (lid == 0) {
value[wid] = local_sum[0];
}
}

__kernel void bias_activate (__global const float *bias,
                                      __global float *value) {
int gid;
float sum;

gid = get_global_id (0);
sum = value[gid] + bias[gid];

sum = 1.0f / (1.0f + exp (-sum));
if (gid < OUTPUTS) {
value[gid] = sum;
}
}

__kernel void clear_input_gradient (__global float *gradient) {
    gradient[get_global_id (0)] = 0;
}

__kernel void bias_backprop (__global const float *gradient,
                                      __global float *delta,
                                      __global float *bias,
                                      const float rate,
                                      const float momentum,
                                      const float decay)
{
int gid;
float d;

    gid = get_global_id (0);
d = delta[gid] * momentum + gradient[gid] * rate * (1 - momentum);
    delta[gid] = d;
    bias[gid] = bias[gid] * decay + d;
}

__kernel void weight_backprop (__global const float *gradient,
                                        __global float *delta,
                                        __global float *weight,
                                        __global float *value,
                                        __global float *input_gradient,
                                        __global const float *input_value,
                                        const float rate,
                                        const float momentum,
                                        const float decay)
{
    __local float partial[OUTPUTS];
    int gid = get_global_id (0);
    int lid = get_local_id (0);

    int index = lid * INPUTS;

    float g = gradient[index];
    float d = delta[index] * momentum
    + g * rate * (1 - momentum)
    * input_value[lid];
    float w = weight[index] * decay + d;

    delta[index] = d;
    weight[index] = w;
    partial[lid] = g * w;

    for (int off = get_local_size (0) / 2; off > 0; off /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < off) {
            partial[lid] += partial[lid + off];
        }
    }

    if (lid == 0) {
        input_gradient[get_group_id (0)] = partial[0];
    }
}

__kernel void derive_gradient (__global const float *value,
                                        __global float *gradient)
{
    int gid = get_global_id (0);

    float v = value[gid];

    gradient[gid] *= v * (1 - v);
}
