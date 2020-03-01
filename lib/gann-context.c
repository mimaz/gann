#include "gann-context.h"

#include "core/core.h"

struct _GannContext
{
    GObject parent_instance;
    struct context *core;
    GSList *networks;
};

G_DEFINE_TYPE (GannContext, gann_context, G_TYPE_OBJECT);

static void dispose (GObject *gobj);
static void constructed (GObject *gobj);

static void
gann_context_init (GannContext *self)
{
    self->networks = NULL;
}

static void
gann_context_class_init (GannContextClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
    gcls->constructed = constructed;
}

static void
dispose (GObject *gobj)
{
    GannContext *self = GANN_CONTEXT (gobj);

    g_clear_pointer (&self->core, context_free);

    G_OBJECT_CLASS (gann_context_parent_class)->dispose (gobj);
}

static void
constructed (GObject *gobj)
{
    GannContext *self = GANN_CONTEXT (gobj);

    self->core = context_create ();

    G_OBJECT_CLASS (gann_context_parent_class)->constructed (gobj);
}

GannContext *
gann_context_new ()
{
    return g_object_new (GANN_TYPE_CONTEXT, NULL);
}

void
gann_context_add_network (GannContext *self,
                          GannNetwork *network)
{
    self->networks = g_slist_append (self->networks, network);
}

void
gann_context_remove_network (GannContext *self,
                             GannNetwork *network)
{
    self->networks = g_slist_remove (self->networks, network);
}

struct context *
gann_context_get_core (GannContext *self)
{
    return self->core;
}
