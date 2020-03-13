/*
 * application.c
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

#include "application.h"

#include <gann.h>

struct _TestApplication
{
    GtkApplication parent_instance;
    GdkPixbuf *pixbuf;
    GannContext *context;
    GannNetwork *network;
};

G_DEFINE_TYPE (TestApplication, test_application, GTK_TYPE_APPLICATION);

static void activate (TestApplication *self);
static void dispose (GObject *self);

static void
test_application_init (TestApplication *self)
{
    g_signal_connect (self, "activate",
                      G_CALLBACK (activate), NULL);
}

static void
test_application_class_init (TestApplicationClass *cls)
{
    GObjectClass *gcls = G_OBJECT_CLASS (cls);

    gcls->dispose = dispose;
}

TestApplication *
test_application_new ()
{
    return g_object_new (TEST_TYPE_APPLICATION, NULL);
}

static void
activate (TestApplication *self)
{
    GtkWidget *win;

    win = gtk_application_window_new (GTK_APPLICATION (self));

    gtk_widget_show (win);
}

static void
dispose (GObject *gobj)
{
    TestApplication *self = TEST_APPLICATION (gobj);

    g_clear_object (&self->pixbuf);
    g_clear_object (&self->context);
    g_clear_object (&self->network);
}
