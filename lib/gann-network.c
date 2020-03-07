#include "gann-network.h"

#include "gann-layer-private.h"
#include "gann-input-layer.h"
#include "gann-output-layer.h"
#include "gann-fully-layer.h"
#include "gann-context.h"

#include "core/core.h"

typedef struct {
    struct network *net;
    GannContext *context;
    GPtrArray *layer_arr;
    gfloat avg_loss;
} GannNetworkPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (GannNetwork, gann_network, G_TYPE_OBJECT);

enum
{
    PROP_0,
    PROP_CONTEXT,
    PROP_RATE,
    PROP_MOMENTUM,
    PROP_DECAY,
    PROP_LAYER_COUNT,
    PROP_LOSS,
    PROP_AVERAGE_LOSS,
    N_PROPS,
};

static GParamSpec *props[N_PROPS];

static void dispose (GObject *gobj);
static void constructed (GObject *gobj);
static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);

static void
gann_network_init (GannNetwork *self)
{
}

static void
gann_network_class_init (GannNetworkClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
    gcls->set_property = set_property;
    gcls->get_property = get_property;

    props[PROP_CONTEXT] =
        g_param_spec_object ("context",
                             "Context",
                             "Network's context",
                             GANN_TYPE_CONTEXT,
                             G_PARAM_READWRITE |
                             G_PARAM_CONSTRUCT_ONLY |
                             G_PARAM_STATIC_STRINGS);

    props[PROP_RATE] =
        g_param_spec_float ("rate",
                            "Rate",
                            "Learning rate",
                            0.0f, 1.0f, 0.001f,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_MOMENTUM] =
        g_param_spec_float ("momentum",
                            "Momentum",
                            "Learning momentum",
                            0.0f, 1.0f, 0.9f,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_DECAY] =
        g_param_spec_float ("decay",
                            "Decay",
                            "Learning decay",
                            0.0f, 1.0f, 1.0f,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_LAYER_COUNT] =
        g_param_spec_int ("layer-count",
                          "Layer count",
                          "Number of layers",
                          0, G_MAXINT32, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_LOSS] =
        g_param_spec_float ("loss",
                            "Loss",
                            "Latest loss",
                            0, G_MAXFLOAT, 0,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    props[PROP_AVERAGE_LOSS] =
        g_param_spec_float ("average-loss",
                            "Average loss",
                            "Average loss",
                            0, G_MAXFLOAT, 0,
                            G_PARAM_READWRITE |
                            G_PARAM_STATIC_STRINGS);

    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
dispose (GObject *gobj)
{
    GannNetwork *self = GANN_NETWORK (gobj);
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    g_clear_pointer (&p->layer_arr, g_ptr_array_unref);
    g_clear_pointer (&p->net, network_free);
    g_clear_object (&p->context);

    G_OBJECT_CLASS (gann_network_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannNetwork *self = GANN_NETWORK (gobj);
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    p->net = network_make_empty (gann_context_get_core (p->context));
    p->layer_arr = g_ptr_array_new_with_free_func (g_object_unref);
    p->avg_loss = -1;

    G_OBJECT_CLASS (gann_network_parent_class)->constructed (gobj);
}

static void
set_property (GObject *gobj,
              guint propid,
              const GValue *value,
              GParamSpec *spec)
{
    GannNetwork *self = GANN_NETWORK (gobj);
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    switch (propid)
    {
    case PROP_CONTEXT:
        g_set_object (&p->context, g_value_get_object (value));
        break;

    case PROP_RATE:
        gann_network_set_rate (self, g_value_get_float (value));
        break;

    case PROP_MOMENTUM:
        gann_network_set_momentum (self, g_value_get_float (value));
        break;

    case PROP_DECAY:
        gann_network_set_decay (self, g_value_get_float (value));
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
get_property (GObject *gobj,
              guint propid,
              GValue *value,
              GParamSpec *spec)
{
    GannNetwork *self = GANN_NETWORK (gobj);
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    switch (propid)
    {
    case PROP_CONTEXT:
        g_value_set_object (value, p->context);
        break;

    case PROP_RATE:
        g_value_set_float (value, p->net->rate);
        break;

    case PROP_MOMENTUM:
        g_value_set_float (value, p->net->momentum);
        break;

    case PROP_DECAY:
        g_value_set_float (value, p->net->decay);
        break;

    case PROP_LAYER_COUNT:
        g_value_set_int (value, p->layer_arr->len);
        break;

    case PROP_LOSS:
        g_value_set_float (value, p->net->loss);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

GannNetwork *
gann_network_new (GannContext *context)
{
    return gann_network_new_full (context, 0.001f, 0.99f, 1.0f);
}

GannNetwork *
gann_network_new_full (GannContext *context,
                       gfloat rate, gfloat momentum, gfloat decay)
{
    return g_object_new (GANN_TYPE_NETWORK,
                         "context", context,
                         "rate", rate,
                         "momentum", momentum,
                         "decay", decay,
                         NULL);
}

static void
insert_layer (GannNetwork *self,
              GannLayer *layer)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    g_assert (GANN_IS_INPUT_LAYER (layer) || p->layer_arr->len > 0);
    g_ptr_array_insert (p->layer_arr, -1, layer);
    g_object_notify_by_pspec (G_OBJECT (self),
                              props[PROP_LAYER_COUNT]);
}

GannInputLayer *
gann_network_create_input (GannNetwork *self,
                           gint width,
                           gint height,
                           gint depth)
{
    GannLayer *layer;

    layer = gann_layer_new_input (self, width, height, depth);
    insert_layer (self, layer);

    return GANN_INPUT_LAYER (layer);
}

GannOutputLayer *
gann_network_create_output (GannNetwork *self)
{
    GannLayer *layer;

    layer = gann_layer_new_output (self);
    insert_layer (self, layer);

    return GANN_OUTPUT_LAYER (layer);
}

GannFullyLayer *
gann_network_create_fully (GannNetwork *self,
                           gint width,
                           gint height,
                           gint depth,
                           GannActivation activation)
{
    GannLayer *layer;

    layer = gann_layer_new_fully (self, width, height, depth, activation);
    insert_layer (self, layer);

    return GANN_FULLY_LAYER (layer);
}

void
gann_network_forward (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    network_forward (p->net);
}

void
gann_network_backward (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    network_backward (p->net);

    if (p->avg_loss < 0) {
        p->avg_loss = p->net->loss;
    } else {
        p->avg_loss *= 0.95f;
        p->avg_loss += p->net->loss * 0.05f;
    }

    g_object_notify_by_pspec (G_OBJECT (self),
                              props[PROP_LOSS]);
    g_object_notify_by_pspec (G_OBJECT (self),
                              props[PROP_AVERAGE_LOSS]);
}

GannLayer *
gann_network_get_layer (GannNetwork *self,
                        gint index)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    g_return_val_if_fail (index < p->layer_arr->len, NULL);
    return g_ptr_array_index (p->layer_arr, index);
}

struct network *
gann_network_get_core (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net;
}

GannContext *
gann_network_get_context (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->context;
}

void
gann_network_set_rate (GannNetwork *self,
                       gfloat rate)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (rate != p->net->rate) {
        p->net->rate = rate;
        g_object_notify_by_pspec (G_OBJECT (self),
                                  props[PROP_RATE]);
    }
}

gfloat
gann_network_get_rate (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net->rate;
}

void
gann_network_set_momentum (GannNetwork *self,
                           gfloat momentum)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (momentum != p->net->momentum) {
        p->net->momentum = momentum;
        g_object_notify_by_pspec (G_OBJECT (self),
                                  props[PROP_MOMENTUM]);
    }
}

gfloat
gann_network_get_momentum (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net->momentum;
}

void
gann_network_set_decay (GannNetwork *self,
                        gfloat decay)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    if (decay != p->net->decay) {
        p->net->decay = decay;
        g_object_notify_by_pspec (G_OBJECT (self),
                                  props[PROP_DECAY]);
    }
}

gfloat
gann_network_get_decay (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net->decay;
}

gint
gann_network_get_layer_count (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->layer_arr->len;
}

gfloat
gann_network_get_loss (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->net->loss;
}

gfloat
gann_network_get_average_loss (GannNetwork *self)
{
    GannNetworkPrivate *p = gann_network_get_instance_private (self);

    return p->avg_loss;
}
