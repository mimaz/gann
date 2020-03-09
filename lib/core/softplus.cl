#ifdef WITH_DERIVATIVE
float activate (float x, float *d)
#else
float activate (float x)
#endif
{
    float e = exp (x);
#ifdef WITH_DERIVATIVE
    *d = e / (1.0f + e);
#endif
    return log (1.0f + e);
}
