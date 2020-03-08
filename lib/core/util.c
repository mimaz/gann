#include "utils.h"

int util_upper_power_2 (int v)
{
    int p2;

    p2 = 1;

    while (p2 < v) {
        p2 *= 2;
    }

    return p2;
}
