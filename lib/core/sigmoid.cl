#ifdef WITH_DERIVATIVE
float activate (float x, float *d)
#else
float activate (float x)
#endif
{
    float s = 1.0f / (1.0f + exp (-x));
#ifdef WITH_DERIVATIVE
    *d = s * (1.0f - s);
#endif
    return s;
}
