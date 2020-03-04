__kernel void reduce_inputs (__global const float *input,
                                      __global const float *weight,
                                      __global float *value) {
__local float partial[INPUTS];

int local_id = get_local_id (0);
int global_id = get_global_id (0);
int group_id = get_group_id (0);

partial[local_id] = input[local_id] * weight[global_id];

for(int off = get_local_size (0) / 2; off > 0; off /= 2) {
barrier(CLK_LOCAL_MEM_FENCE);

if (local_id < off) {
partial[local_id] += partial[local_id + off];
}
}

if (local_id == 0) {
value[group_id] = partial[0];
}
}

__kernel void bias_activate (__global const float *bias,
                                      __global float *value) {
int global_id = get_global_id (0);
float sum = value[global_id] + bias[global_id];

sum = 1.0f / (1.0f + exp (-sum));
value[global_id] = sum;
}
