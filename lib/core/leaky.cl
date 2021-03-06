
/*
 * leaky.cl
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

#ifdef WITH_DERIVATIVE
float activate (float x, float *d)
#else
float activate (float x)
#endif
{
    const float alpha = 0.01f;
    if (x > 0) {
#ifdef WITH_DERIVATIVE
        *d = 1;
#endif
        return x;
    } else {
#ifdef WITH_DERIVATIVE
        *d = alpha;
#endif
        return x * alpha;
    }
}
