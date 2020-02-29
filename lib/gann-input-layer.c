#include "gann-input-layer.h"
#include "gann-layer-private.h"

#include "gann-network-private.h"

#include "core/layer.h"

struct _GannInputLayer
{
    GObject parent_instance;
    GArray *owned_array;
};

G_DEFINE_TYPE (GannInputLayer, gann_input_layer,
               GANN_TYPE_LAYER);

static void dispose (GObject *gobj);
static void constructed (GObject *gobj);

static void
gann_input_layer_init (GannInputLayer *self)
{
}

static void
gann_input_layer_class_init (GannInputLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
}

static void
dispose (GObject *gobj)
{
    GannInputLayer *self = GANN_INPUT_LAYER (gobj);

    if (self->owned_array != NULL) {
        gann_layer_get_core (GANN_LAYER (self))->value_v = NULL;
    }

    g_clear_pointer (&self->owned_array, g_array_unref);

    G_OBJECT_CLASS (gann_input_layer_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannLayer *layer;
    GannNetwork *network;
    gint width, height, depth;
    struct layer *core;

    layer = GANN_LAYER (gobj);
    network = gann_layer_get_network (layer);

    width = gann_layer_get_width (layer);
    height = gann_layer_get_height (layer);
    depth = gann_layer_get_depth (layer);

    core = layer_make_input (gann_network_get_core (network),
                             width, height, depth);
    gann_layer_set_core (layer, core);

    G_OBJECT_CLASS (gann_input_layer_parent_class)->constructed (gobj);
}

void
gann_input_layer_set_input (GannInputLayer *self,
                            const gfloat *data,
                            gsize datasize)
{
    struct layer *lay;

    lay = gann_layer_get_core (GANN_LAYER (self));
    g_assert (datasize == lay->size);

    lay->value_v = (gfloat *) data;
}

void
gann_input_layer_set_input_ints (GannInputLayer *self,
                                 gint first, ...)
{
    GArray *arr;
    va_list args;
    gint value;
    gfloat fvalue;

    arr = g_array_new (FALSE, FALSE, sizeof (gfloat));
    value = first;

    va_start (args, first);

    do {
        fvalue = value;
        g_array_append_val (arr, fvalue);
        value = va_arg (args, gint);
    } while (value >= 0);

    gann_input_layer_set_input (self, (const gfloat *) arr->data, arr->len);

    g_clear_pointer (&self->owned_array, g_array_unref);
    va_end (args);

    self->owned_array = arr;
}
