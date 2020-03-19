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

float gemm (__global const float **x,
            __global const float *k)
{
}

__kernel void forward (__global const float *input_v,
                       __global const float *kernel_v,
                       __global float *output_v)
{
    __private int id, x, y, z;
    __private float *xvector[SIZE * SIZE * DEPTH];
    
    id = get_global_id (0);

    output_v[id] = input_v[id];
}
