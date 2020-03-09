Gann

This is an example implementation of neural network which makes use
of OpenCL and GLib Object.

I started this project for my own educational purpose to learn OpenCL
and deep learning basics. However it's possible this library might be
pretty useful in the future due to very good performance and advanced
optimization.

The library is split into two main parts:

First one is the core library (ganncore). It's the real neural network
implementation using OpenCL 1.0 API. It also depends on core GLib (but
no GObject). It implements all computational logic. Here I can focus
on doing optimizations with minimum computation overhead.

Second part is a kind of GObject wrapper over ganncore. It's purpose is
to deliver more friendly API, some additional high level functions and
bindings to other programming languages. I would like to make the code
accessible from as many languages as possible.

Currently I'm still working on dense layers and computation core.
Already I got working mulitple dense layers with different activation
functions. The basic idea is to make each layer has it's own OpenCL
program with static definitions about functionality the layer needs.
Therefore we can eliminate many repetitive dynamic checkings for example
for the layer size (staticaly defined sizes), activation functions
(they are fully inlined), or the need of backpropagation (if we
propatgate just forward, there's no need to have some buffers or
derivative calculations) and so.

If I get this stuff stable and functional, I will start working on
convolutional layers to make use of real deep learning and computer
vision.

Mieszko Mazurek <mimaz@gmx.com>
