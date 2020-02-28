#include "gann-input-layer.h"
#include "gann-layer-private.h"

#include "gann-network-private.h"

#include "core/layer.h"

struct _GannInputLayer
{
    GObject parent_instance;
};

G_DEFINE_TYPE (GannInputLayer, gann_input_layer,
               GANN_TYPE_LAYER);

static void constructed (GObject *gobj);

static void
gann_input_layer_init (GannInputLayer *self)
{
}

static void
gann_input_layer_class_init (GannInputLayerClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->constructed = constructed;
}

static void
constructed (GObject *gobj)
{
    GannLayerPrivate *p = gann_layer_get_private (gobj);
    struct network *net;

    net = gann_network_get_core (p->gnet);

    g_message ("make input %p %d %d %d", net, p->width,
               p->height, p->depth);

    p->l = layer_make_input (net,
                             p->width,
                             p->height,
                             p->depth);

    G_OBJECT_CLASS (gann_input_layer_parent_class)->constructed (gobj);
}
