#include "gann-layer.h"

#include "gann-layer-private.h"
#include "gann-input-layer.h"
#include "gann-output-layer.h"
#include "gann-fully-layer.h"
#include "gann-network.h"
#include "gann-network-private.h"

#include "core/layer.h"

typedef struct {
    GannNetwork *network;

    gint width;
    gint height;
    gint depth;
    GannActivation activation;

    struct layer *l;
    enum activation_type core_activation;
} GannLayerPrivate;

G_DEFINE_TYPE_WITH_PRIVATE (GannLayer, gann_layer, G_TYPE_OBJECT);

enum
{
    PROP_0,
    PROP_NETWORK,
    PROP_WIDTH,
    PROP_HEIGHT,
    PROP_DEPTH,
    PROP_ACTIVATION,
    N_PROPS,
};

static GParamSpec *props[N_PROPS];

static void dispose (GObject *gobj);
static void constructed (GObject *gobj);
static void set_property (GObject *gobj, guint propid,
                          const GValue *value, GParamSpec *spec);
static void get_property (GObject *gobj, guint propid,
                          GValue *value, GParamSpec *spec);
static void update_core_activation (GannLayer *self);

static void
gann_layer_init (GannLayer *self)
{
}

static void
gann_layer_class_init (GannLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
    gcls->set_property = set_property;
    gcls->get_property = get_property;

    props[PROP_NETWORK] =
        g_param_spec_object ("network",
                             "Network",
                             "Network object",
                             GANN_TYPE_NETWORK,
                             G_PARAM_READWRITE |
                             G_PARAM_CONSTRUCT_ONLY |
                             G_PARAM_STATIC_STRINGS);

    props[PROP_WIDTH] =
        g_param_spec_int ("width",
                          "Width",
                          "Layer width",
                          0, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);
    props[PROP_HEIGHT] =
        g_param_spec_int ("height",
                          "Height",
                          "Layer height",
                          0, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_DEPTH] =
        g_param_spec_int ("depth",
                          "Depth",
                          "Layer depth",
                          0, G_MAXINT16, 0,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_ACTIVATION] =
        g_param_spec_enum ("activation",
                           "Activation",
                           "Layer activation",
                           GANN_TYPE_ACTIVATION,
                           GANN_ACTIVATION_LINEAR,
                           G_PARAM_READWRITE |
                           G_PARAM_CONSTRUCT_ONLY |
                           G_PARAM_STATIC_STRINGS);

    g_object_class_install_properties (gcls, N_PROPS, props);
}

static void
dispose (GObject *gobj)
{
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    g_clear_weak_pointer (&p->network);

    G_OBJECT_CLASS (gann_layer_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    G_OBJECT_CLASS (gann_layer_parent_class)->constructed (gobj);

    g_assert_nonnull (p->l);
}

static void
set_property (GObject *gobj,
              guint propid,
              const GValue *value,
              GParamSpec *spec)
{
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    switch (propid)
    {
    case PROP_NETWORK:
        g_set_weak_pointer (&p->network, g_value_get_object (value));
        break;

    case PROP_WIDTH:
        p->width = g_value_get_int (value);
        break;

    case PROP_HEIGHT:
        p->height = g_value_get_int (value);
        break;

    case PROP_DEPTH:
        p->depth = g_value_get_int (value);
        break;

    case PROP_ACTIVATION:
        p->activation = g_value_get_enum (value);
        update_core_activation (self);
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
    GannLayer *self = GANN_LAYER (gobj);
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    switch (propid)
    {
    case PROP_NETWORK:
        g_value_set_object (value, p->network);
        break;

    case PROP_WIDTH:
        g_value_set_int (value, p->width);
        break;

    case PROP_HEIGHT:
        g_value_set_int (value, p->height);
        break;

    case PROP_DEPTH:
        g_value_set_int (value, p->depth);
        break;

    case PROP_ACTIVATION:
        g_value_set_enum (value, p->activation);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

static void
update_core_activation (GannLayer *self)
{
    static enum activation_type core_v[] = {
        [GANN_ACTIVATION_LINEAR] = ACTIVATION_LINEAR,
        [GANN_ACTIVATION_RELU] = ACTIVATION_RELU,
        [GANN_ACTIVATION_SIGMOID] = ACTIVATION_SIGMOID,
        [GANN_ACTIVATION_LEAKY] = ACTIVATION_LEAKY,
        [GANN_ACTIVATION_ELU] = ACTIVATION_ELU,
        [GANN_ACTIVATION_STEP] = ACTIVATION_STEP,
    };

    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    GannActivation act;

    act = gann_layer_get_activation (self);
    g_assert (act < G_N_ELEMENTS (core_v));
    p->core_activation = core_v[act];
}

const gfloat *
gann_layer_get_data (GannLayer *self,
                     gsize *size)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);

    if (size != NULL) {
        *size = p->l->size;
    }

    return p->l->value_v;
}

GannNetwork *
gann_layer_get_network (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->network;
}

gint
gann_layer_get_width (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->width;
}

gint
gann_layer_get_height (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->height;
}

gint
gann_layer_get_depth (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->depth;
}

GannActivation
gann_layer_get_activation (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->activation;
}

/* PRIVATE */

void
gann_layer_set_core (GannLayer *self,
                     struct layer *core)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    g_assert (p->l == NULL);
    p->l = core;
}

struct layer *
gann_layer_get_core (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->l;
}

enum activation_type
gann_layer_get_core_activation (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->core_activation;
}

GannLayer *
gann_layer_new_input (GannNetwork *network,
                      gint width,
                      gint height,
                      gint depth)
{
    return g_object_new (GANN_TYPE_INPUT_LAYER,
                         "network", network,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         NULL);
}

GannLayer *
gann_layer_new_output (GannNetwork *network)
{
    return g_object_new (GANN_TYPE_OUTPUT_LAYER,
                         "network", network,
                         NULL);
}

GannLayer *
gann_layer_new_fully (GannNetwork *network,
                      gint width,
                      gint height,
                      gint depth,
                      GannActivation activation)
{
    return g_object_new (GANN_TYPE_FULLY_LAYER,
                         "network", network,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         "activation", activation,
                         NULL);
}
