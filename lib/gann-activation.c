#include "gann-activation.h"

GType
gann_activation_get_type ()
{
    static volatile GType id_volatile;
    GType id;

    if (g_once_init_enter (&id_volatile)) {
        static GEnumValue values[] = {
            { GANN_ACTIVATION_LINEAR, "GANN_ACTIVATION_LINEAR", "linear" },
            { GANN_ACTIVATION_RELU, "GANN_ACTIVATION_RELU", "relu" },
            { GANN_ACTIVATION_SIGMOID, "GANN_ACTIVATION_SIGMOID", "sigmoid" },
            { GANN_ACTIVATION_LEAKY, "GANN_ACTIVATION_LEAKY", "leaky" },
            { GANN_ACTIVATION_ELU, "GANN_ACTIVATION_ELU", "elu" },
            { GANN_ACTIVATION_STEP, "GANN_ACTIVATION_STEP", "step" },
            { 0, NULL, NULL },
        };

        id = g_enum_register_static (g_intern_static_string ("GannActivation"),
                                     values);

        g_once_init_leave (&id_volatile, id);
    }

    return id_volatile;
}
