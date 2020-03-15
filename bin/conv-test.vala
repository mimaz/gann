/*
 * conv-test.vala
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

public void conv_test () throws Error
{
    var pixbuf = new Gdk.Pixbuf.from_file ("dog.jpg");
    var stage = new Clutter.Stage ();
    var image = new Clutter.Image ();

    pixbuf = pixbuf.scale_simple (32, 32, Gdk.InterpType.NEAREST);

    var context = new Gann.Context ();
    var network = new Gann.Network (context, 0.5f, 0.99f, 1.0f);
    var input = network.create_input (32, 32, 3);
    network.create_dense (32, 32, 3, "leaky");
    // network.create_conv (1, 1, 3, "softplus");
    var output = network.create_output ();

    input.set_data_bytes (pixbuf.get_pixels_with_length ());

    network.forward ();

    var result = output.get_data_bytes ();
    message ("result: %d", result.length);

    stage.set_size (500, 500);
    stage.show ();
    stage.hide.connect (Clutter.main_quit);
    stage.content = image;
    image.set_data (result,
                    Cogl.PixelFormat.RGB_888,
                    pixbuf.width,
                    pixbuf.height,
                    pixbuf.rowstride);

    Clutter.main ();
}
