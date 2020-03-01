#include "context.h"
#include "network.h"

struct context *
context_create ()
{
    struct context *ctx;

    ctx = g_new (struct context, 1);
    ctx->netlist = NULL;

    return ctx;
}

void
context_free (struct context *ctx)
{
    g_slist_free_full (ctx->netlist, (GDestroyNotify) network_free);
    g_free (ctx);
}
