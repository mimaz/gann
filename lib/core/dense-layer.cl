__kernel void forward (__global const float *input_value_v,
                       __global const float *weight_v,
                       __global const float *bias_v,
                       __global float *value_v)
{
    __private float sum;
    __private int outid, inid;

    outid = get_global_id (0);

    if (outid < OUTPUTS) {
        sum = bias_v[outid];

        for (inid = 0; inid < INPUTS; inid++) {
            sum += input_value_v[inid] * weight_v[outid * INPUTS + inid];
        }

        sum = 1.0f / (1.0f + exp (-sum));
        value_v[outid] = sum;
    }
}

__kernel void derive_gradient (__global const float *value_v,
#ifdef DERIVATIVE_WITH_INPUT
                               __global const float *input_v,
#endif
                               __global float *gradient_v)
{
    __private int outid;

    outid = get_global_id (0);

    if (outid < OUTPUTS) {
#ifdef DERIVATIVE_WITH_INPUT
        gradient_v[outid] *= activation_derivative (value_v[outid],
                                                    input_v[outid]);
#else
        gradient_v[outid] *= activation_derivative (value_v[outid]);
#endif
    }
}

__kernel void backward (__global const float *input_value_v,
                        __global const float *gradient_v,
                        __global float *input_gradient_v,
                        __global float *weight_v,
                        __global float *delta_v,
                        const float rate,
                        const float momentum,
                        const float decay)
{
    __private int inid, w_index, outid;
    __private float in, g, d, w;
#ifdef CALC_GRADIENT
    __private float sum;
#endif

    inid = get_global_id (0);

    if (inid < INPUTS) {
        in = input_value_v[inid];
#ifdef CALC_GRADIENT
        sum = 0;
#endif

        for (outid = 0; outid < OUTPUTS; outid++) {
            w_index = outid * INPUTS + inid;
            g = gradient_v[outid];
            d = delta_v[w_index];
            w = weight_v[w_index];

            d = d * momentum + g * rate * in;
            w = w * decay + d;
#ifdef CALC_GRADIENT
            sum += g * w;
#endif

            weight_v[w_index] = w;
            delta_v[w_index] = d;
        }

#ifdef CALC_GRADIENT
        input_gradient_v[inid] = sum;
#endif
    }
}

__kernel void backward_bias (__global const float *gradient_v,
                             __global float *bias_v,
                             __global float *delta_v,
                             const float rate,
                             const float momentum,
                             const float decay)
{
    __private float d, g, b;
    __private int id;

    id = get_global_id (0);

    if (id < OUTPUTS) {
        d = delta_v[id];
        g = gradient_v[id];
        b = bias_v[id];

        d = d * momentum + g * rate;
        b = b * decay + d;

        bias_v[id] = b;
        delta_v[id] = d;
    }
}
