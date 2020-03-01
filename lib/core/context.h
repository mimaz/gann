#pragma once

#include <glib.h>

struct context
{
    GSList *netlist;
};

struct context *context_create ();
void context_free (struct context *ctx);
