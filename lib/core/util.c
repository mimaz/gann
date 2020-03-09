/*
 * util.c
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

#include "util.h"

#include <math.h>

int
util_upper_power_2 (int v)
{
    int p2;

    p2 = 1;

    while (p2 < v) {
        p2 *= 2;
    }

    return p2;
}

int
util_upper_multiply (int v, int g)
{
    return ceilf ((float) v / g) * g;
}
