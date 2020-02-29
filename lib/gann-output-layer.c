#include "gann-output-layer.h"

struct _GannOutputLayer
{
    GannLayer parent_instance;
};

G_DEFINE_TYPE (GannOutputLayer, gann_output_layer, GANN_TYPE_LAYER);

static void
gann_output_layer_init (GannOutputLayer *self)
{
}

static void
gann_output_layer_class_init (GannOutputLayerClass *cls)
{
}

void
gann_output_layer_set_truth (GannOutputLayer *self,
                             const gfloat *data,
                             gsize datasize)
{
    gann_network_
}

void
gann_output_layer_set_truth_floats (GannOutputLayer *self,
                                    gfloat first, ...)
{
    GArray *arr;
    va_list args;

    arr = g_array_new (FALSE, FALSE, sizeof (gfloat));
    va_start (args, first);

    do {
        g_array_append_val (arr, first);

        /* floats are promoted as doubles */
        first = va_arg (args, gdouble);
    } while (first >= 0);

    gann_output_layer_set_truth (self, (gfloat *) arr->data, arr->len);

    g_array_unref (arr);
    va_end (args);
}
