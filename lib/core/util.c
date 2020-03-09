#include "utils.h"

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
