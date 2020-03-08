float activation_value (float input)
{
    return 1.0f / (1.0f + exp (-input));
}

float activation_derivative (float value)
{
    return value * (1.0f - value);
}
