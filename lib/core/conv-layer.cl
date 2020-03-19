/*
 * conv-layer.cl
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

__constant float *input_channel (__constant float *input_v,
                                 __constant float *zero_v,
                                 const int y,
                                 const int x)
{
    __private int off;

    if (x < 0 || x >= WIDTH || y < 0 || y >= HEIGHT) {
        return zero_v;
    }

    off = y * HEIGHT * DEPTH + x * DEPTH;

    return input_v + off;
}

__constant float *filter_channel (__constant float *kernel_v,
                                  const int y,
                                  const int x,
                                  const int z)
{
    __private int off;

    off = z * SIZE * SIZE * DEPTH + y * SIZE * DEPTH + x * DEPTH;

    return kernel_v + off;
}

__kernel void forward (__constant float *input_v,
                       __constant float *kernel_v,
                       __constant float *zero_v,
                       __global float *output_v)
{
    __constant float *__private xvector;
    __constant float *__private kvector;
    __private int x, y, z, yk, xk, d, id;
    __private float sum;

    y = get_global_id (0);
    x = get_global_id (1);
    z = get_global_id (2);

    sum = 0;

    for (yk = 0; yk < SIZE; yk++) {
        for (xk = 0; xk < SIZE; xk++) {
            xvector = input_channel (input_v, zero_v,
                                     y + yk - YKSHIFT,
                                     x + xk - XKSHIFT);
            kvector = filter_channel (kernel_v,
                                      y, x, z);

            for (d = 0; d < DEPTH; d++) {
                /* sum += xvector[d] * kvector[d]; */
                sum += xvector[d] * 1.0f / (SIZE * SIZE * DEPTH);
            }
        }
    }

    id = y * HEIGHT * FILTERS + x * FILTERS + z;

    /* output_v[id] = input_v[id]; */
    output_v[id] = sum;
}
