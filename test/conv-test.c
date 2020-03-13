/*
 * conv-test.c
 *
 * Copyright 2020 Mieszko Mazurek <mimaz@gmx.com>
 *
 * This file is part of Gann.
 *
 * Gann is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Gann is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Gann.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <gann.h>
#include <gtk/gtk.h>

static void
on_activate (GtkApplication *app)
{
    GtkWidget *window;
    GtkWidget *image;
    GdkPixbuf *pix;
    GError *err = NULL;

    window = gtk_application_window_new (app);
    pix = gdk_pixbuf_new_from_file_at_scale ("dog.jpg", 32, 32,
                                             FALSE, &err);
    g_assert (err == NULL);
    image = gtk_image_new_from_pixbuf (pix);

    gtk_container_add (GTK_CONTAINER (window), image);

    gtk_widget_show (window);
}

gint
main (gint argc, gchar **argv)
{
    g_autoptr (GannContext) ctx;
    g_autoptr (GannNetwork) net;
    g_autoptr (GtkApplication) app;
    GtkWidget *image;
    GannConvLayer *conv;

    ctx = gann_context_new ();
    net = gann_network_new (ctx, 0.5f, 0.99f, 1.0f);
    gann_network_create_input (net, 32, 32, 1);

    conv = gann_network_create_conv (net, 3, 1, 1, "linear");

    app = gtk_application_new ("pl.mimaz.gann", G_APPLICATION_FLAGS_NONE);

    g_signal_connect (app, "activate",
                      G_CALLBACK (on_activate), &image);

    return g_application_run (G_APPLICATION (app), argc, argv);
}
