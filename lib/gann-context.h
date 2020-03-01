#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

struct context;
typedef struct _GannNetwork GannNetwork;

#define GANN_TYPE_CONTEXT (gann_context_get_type ())

G_DECLARE_FINAL_TYPE (GannContext, gann_context,
                      GANN, CONTEXT, GObject);

GannContext *gann_context_new ();
void gann_context_add_network (GannContext *self,
                               GannNetwork *network);
void gann_context_remove_network (GannContext *self,
                                  GannNetwork *network);
struct context *gann_context_get_core (GannContext *self);

G_END_DECLS
