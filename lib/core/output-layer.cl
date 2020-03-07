__kernel void backprop (__global const float *truth_v,
                        __global const float *value_v,
                        __global float *prev_gradient_v,
                        __global float *loss_p)
{
    __local float local_loss[OUTPUTS];
    int gid, off;
    float sub, loss;

    gid = get_local_id (0);

    if (gid < OUTPUTS) {
        sub = truth_v[gid] - value_v[gid];

        local_loss[gid] = sub * sub;

        barrier(CLK_LOCAL_MEM_FENCE);

        if (gid == 0) {
            for (off = 1; off < OUTPUTS; off++) {
                local_loss[0] += local_loss[off];
            }

            local_loss[0] = sqrt (local_loss[0]);
            loss_p[0] = local_loss[0];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        prev_gradient_v[gid] = sub * local_loss[0];
    } else {
        barrier(CLK_LOCAL_MEM_FENCE);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
