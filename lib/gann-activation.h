#pragma once

#include <glib-object.h>

G_BEGIN_DECLS

typedef enum
{
    GANN_ACTIVATION_LINEAR,
    GANN_ACTIVATION_RELU,
    GANN_ACTIVATION_SIGMOID,
} GannActivation;

#define GANN_TYPE_ACTIVATION (gann_activation_get_type ())

GType gann_activation_get_type ();

G_END_DECLS
