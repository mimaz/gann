/*
 * dense-layer.cl
 *
 * Copyright 2020 Mieszko Mazurek <mimaz@gmx.com>
 *
 * This file is part of Gann.
 *
 * Gann is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gann is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gann.  If not, see <http://www.gnu.org/licenses/>.
 */

__kernel void forward (__global const float *input_value_v,
                       __global const float *weight_v,
                       __global const float *bias_v,
                       __global float *value_v
#ifdef WITH_DERIVATIVE
                       , __global float *derivative_v
#endif
                       )
{
    __private float sum, derivative;
    __private int outid, inid;

    outid = get_global_id (0);

    if (outid < OUTPUTS) {
        sum = bias_v[outid];

        for (inid = 0; inid < INPUTS; inid++) {
            sum += input_value_v[inid] * weight_v[outid * INPUTS + inid];
        }

#ifdef WITH_DERIVATIVE
        value_v[outid] = activate (sum, &derivative);
        derivative_v[outid] = derivative;
#else
        value_v[outid] = activate (sum);
#endif
    }
}

#ifdef WITH_DERIVATIVE
__kernel void derive_gradient (__global const float *derivative_v,
                               __global float *gradient_v)
{
    __private int outid;

    outid = get_global_id (0);

    if (outid < OUTPUTS) {
        gradient_v[outid] *= derivative_v[outid];
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
        input_gradient_v[inid] += sum;
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
#endif
