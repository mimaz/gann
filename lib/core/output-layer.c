#include "layer.h"
#include "network.h"

enum
{
    K_CALC_ERROR,
    N_KERNELS,
};

struct output_layer
{
    struct layer base;
    float *truth_v;
    float loss;
};

static void forward (struct layer *lay);
static void backward (struct layer *lay);
static void release (struct layer *lay);

struct layer *
layer_make_output (struct network *net)
{
    struct output_layer *out;
    struct layer *base, *prev;
    int size;
    g_autofree char *options;
    cl_program program;
    cl_int err;

    out = g_new0 (struct output_layer, 1);
    base = (struct layer *) out;
    prev = network_layer_last (net);
    size = prev->width * prev->height * prev->depth;

    options = g_strdup_printf ("-DINPUTS=%d -DOUTPUTS=%d",
                               prev->size, size);
    program = context_build_program (net->ctx,
                                     options,
                                     "output-layer.cl",
                                     NULL);

    base->net = net;
    base->prev = prev;
    base->type = LAYER_OUTPUT;
    base->activation = ACTIVATION_LINEAR;
    base->value_v = g_new (float, size);
    base->value_mem = clCreateBuffer (net->ctx->context,
                                      CL_MEM_READ_WRITE,
                                      size * sizeof (cl_float),
                                      NULL, &err);
    g_assert (err == CL_SUCCESS);
    base->gradient_v = g_new (float, size);
    base->gradient_mem = clCreateBuffer (net->ctx->context,
                                         CL_MEM_READ_WRITE,
                                         size * sizeof (cl_float),
                                         NULL, &err);
    g_assert (err == CL_SUCCESS);
    base->weight_v = NULL;
    base->weight_mem = 0;
    base->delta_v = NULL;
    base->delta_mem = 0;
    base->bias_delta_v = NULL;
    base->bias_delta_mem = 0;
    base->program = program;
    base->width = prev->width;
    base->height = prev->height;
    base->depth = prev->depth;
    base->size = size;
    base->weights = 0;
    base->forward = forward;
    base->backward = backward;
    base->release = release;

    layer_create_kernel (base, K_CALC_ERROR, "calc_error");

    network_push_layer (net, base);

    return base;
}

void
layer_output_set_truth (struct layer *lay,
                        const float *data,
                        int size)
{
    struct output_layer *out;

    g_assert (lay->type == LAYER_OUTPUT);
    g_assert (lay->size == size);

    out = (struct output_layer *) lay;

    g_clear_pointer (&out->truth_v, g_free);

    out->truth_v = g_memdup (data, size * sizeof (float));
}

static void
forward (struct layer *lay)
{
    g_assert (lay->size == lay->prev->size);

    memcpy (lay->value_v, lay->prev->value_v,
            sizeof (float) * lay->size);
}

static void
backward (struct layer *lay)
{
    struct output_layer *out;
    float sum, sub;
    int i;

    g_assert (lay->size == lay->prev->size);

    out = (struct output_layer *) lay;
    sum = 0;

    for (i = 0; i < lay->size; i++) {
        sub = out->truth_v[i] - lay->value_v[i];
        sum += sub * sub;
        /* g_message ("sub %f %f %f", sub, out->truth_v[i], lay->value_v[i]); */

        lay->gradient_v[i] = sub;
    }

    out->loss = sqrtf (sum);
    g_assert (out->loss == out->loss);

    lay->net->loss += out->loss;

    for (i = 0; i < lay->size; i++) {
        lay->prev->gradient_v[i] = lay->gradient_v[i] * out->loss;
        /* g_message ("prev gradient %f", lay->prev->gradient_v[i]); */
    }
}

static void
release (struct layer *lay)
{
    struct output_layer *out;

    out = (struct output_layer *) lay;

    clReleaseMemObject (lay->value_mem);
    clReleaseMemObject (lay->gradient_mem);

    for (int i = 0; i < N_KERNELS; i++) {
        clReleaseKernel (lay->kernels[i]);
    }

    g_clear_pointer (&lay->value_v, g_free);
    g_clear_pointer (&lay->gradient_v, g_free);
    g_clear_pointer (&out->truth_v, g_free);
}
