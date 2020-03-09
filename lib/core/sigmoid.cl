#ifdef WITH_DERIVATIVE
float activate (float input,
                float *derivative)
#else
float activate (float input)
#endif
{
    float out;

    out = 1.0f / (1.0f + exp (-input));
#ifdef WITH_DERIVATIVE
    *derivative = out * (1.0f - out);
#endif

    return out;
}
