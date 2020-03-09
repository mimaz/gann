/*
 * output-layer.cl
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
