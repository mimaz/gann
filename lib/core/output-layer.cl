__kernel void backprop (__global const float *truth_v,
                        __global const float *value_v,
                        __global float *prev_gradient_v,
                        __global float *loss_p)
{
    __local float local_loss[SIZE];
    __private float sub, loss;
    __private int index, off;

    index = get_local_id (0);
    sub = truth_v[index] - value_v[index];

    local_loss[index] = sub * sub;

    for (off = SIZE_P2U >> 1; off > 0; off >>= 1) {
        if (index <= off && index + off < SIZE) {
            local_loss[index] += local_loss[index + off];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (index == 0) {
        loss = sqrt (local_loss[0]);
        loss_p[0] = loss;
#ifdef CALC_GRADIENT
        local_loss[0] = loss;
#endif
    }

#ifdef CALC_GRADIENT
    barrier(CLK_LOCAL_MEM_FENCE);
    prev_gradient_v[index] = sub * local_loss[0];
#endif
}
