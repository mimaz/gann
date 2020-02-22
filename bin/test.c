#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <glib.h>

gint
main (gint argc, gchar **argv)
{
    char *driver_version;
    clGetDeviceInfo (0, CL_DRIVER_VERSION, sizeof (char *),
                     &driver_version, NULL);
    g_print ("driver_version: %s\n", driver_version);
    return 0;
}
