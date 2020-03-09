__kernel void clear (__global const unsigned char *pattern,
                     __global unsigned char *data,
                     const int size,
                     const int count)
{
    __private int index, i;

    index = get_global_id (0);

    for (i = 0; i < size; i++) {
        data[index * size + i] = pattern[i];
    }
}
