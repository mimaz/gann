#include "application.h"

gint
main (gint argc, gchar **argv)
{
    g_autoptr (TestApplication) app;

    app = test_application_new ();

    return g_application_run (G_APPLICATION (app), argc, argv);
}
