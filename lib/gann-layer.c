#include "gann-layer.h"

#include "gann-layer-private.h"
#include "gann-input-layer.h"
#include "gann-fully-layer.h"
#include "gann-network.h"
#include "gann-network-private.h"

#include "core/layer.h"

G_DEFINE_TYPE_WITH_PRIVATE (GannLayer, gann_layer, G_TYPE_OBJECT);

enum
{
    PROP_0,
    PROP_NETWORK,
    PROP_WIDTH,
    PROP_HEIGHT,
    PROP_DEPTH,
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
                          1, G_MAXINT16, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);
    props[PROP_HEIGHT] =
        g_param_spec_int ("height",
                          "Height",
                          "Layer height",
                          1, G_MAXINT16, 1,
                          G_PARAM_READWRITE |
                          G_PARAM_CONSTRUCT_ONLY |
                          G_PARAM_STATIC_STRINGS);

    props[PROP_DEPTH] =
        g_param_spec_int ("depth",
                          "Depth",
                          "Layer depth",
                          1, G_MAXINT16, 1,
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

    g_clear_weak_pointer (&p->gnet);

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
        g_set_weak_pointer (&p->gnet, g_value_get_object (value));
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
        g_value_set_object (value, p->gnet);
        break;

    case PROP_WIDTH:
        g_value_set_int (value, p->l->width);
        break;

    case PROP_HEIGHT:
        g_value_set_int (value, p->l->height);
        break;

    case PROP_DEPTH:
        g_value_set_int (value, p->l->depth);
        break;

    default:
        G_OBJECT_WARN_INVALID_PROPERTY_ID (gobj, propid, spec);
    }
}

gint
gann_layer_get_width (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->l->width;
}

gint
gann_layer_get_height (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->l->height;
}

gint
gann_layer_get_depth (GannLayer *self)
{
    GannLayerPrivate *p = gann_layer_get_instance_private (self);
    return p->l->depth;
}

/* PRIVATE */

GannLayerPrivate *
gann_layer_get_private (gpointer self)
{
    return gann_layer_get_instance_private (GANN_LAYER (self));
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
gann_layer_new_fully (GannNetwork *network,
                      gint width,
                      gint height,
                      gint depth)
{
    return g_object_new (GANN_TYPE_FULLY_LAYER,
                         "network", network,
                         "width", width,
                         "height", height,
                         "depth", depth,
                         NULL);
}
