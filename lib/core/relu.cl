#ifdef WITH_DERIVATIVE
float activate (float x, float *d)
#else
float activate (float x)
#endif
{
    if (x > 0) {
#ifdef WITH_DERIVATIVE
        *d = 1;
#endif
        return x;
    } else {
#ifdef WITH_DERIVATIVE
        *d = 0;
#endif
        return 0;
    }
}
