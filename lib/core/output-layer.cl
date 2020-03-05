__kernel void calc_error (__global const float *truth,
                            __global const float *value,
                            __global float *gradient,
                            __global float *prev_gradient,
                            __global float *out_loss)
{
    __local float local_loss[OUTPUTS];
    int gid, lid, off;
    float sub;

    gid = get_global_id (0);
    lid = get_local_id (0);
    sub = truth[gid] - value[gid];
    gradient[gid] = sub;

    local_loss[lid] = sub * sub;

    for (off = lid / 2; off > 0; off /= 2) {
        if (lid < off) {
            local_loss[lid] += local_loss[lid + off];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    prev_gradient[lid] = sub * local_loss[0];

    if (lid == 0) {
        out_loss[0] = sqrt (local_loss[0]);
    }
}
