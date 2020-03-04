__kernel void calc_error (__global const float *truth,
                            __global const float *value,
                            __global float *gradient,
                            __global float *loss)
{
    __local float losses[OUTPUTS];
    int gid = get_global_id (0);
    int lid = get_local_id (0);
    float sub = truth[gid] - value[gid];
    gradient[gid] = sub;

    losses[lid] = sub * sub;

    for (int off = get_local_size (0) / 2; off > 0; off /= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid < off) {
            losses[lid] += losses[lid + off];
        }
    }

    if (lid == 0) {
        loss[0] = losses[0];
    }
}
