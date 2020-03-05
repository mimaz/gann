__kernel void backprop (__global const float *truth_v,
                        __global const float *value_v,
                        __global float *prev_gradient_v,
                        __global float *loss_p)
{
    __local float local_loss[OUTPUTS];
    int gid, lid, off;
    float sub, loss;

    gid = get_global_id (0);
    lid = get_local_id (0);
    sub = truth_v[gid] - value_v[gid];

    local_loss[lid] = sub * sub;

    for (off = lid / 2; off > 0; off /= 2) {
        if (lid < off) {
            local_loss[lid] += local_loss[lid + off];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        local_loss[0] = sqrt (local_loss[0]);
        loss_p[0] = local_loss[0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    prev_gradient_v[lid] = sub * local_loss[0];
}
