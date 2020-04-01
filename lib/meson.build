subdir('core/')

source = [
    'gann-network.c',
    'gann-layer.c',
    'gann-context.c',
    'gann-input-layer.c',
    'gann-output-layer.c',
    'gann-dense-layer.c',
    'gann-conv-layer.c',
    'gann-program-builder.c',
    'gann-buffer.c',
    'gann-barrier.c',
]

private_header = [
  'gann-layer-private.h',
  'gann-context-private.h',
  'gann-program-builder.h',
  'gann-buffer-private.h',
]

public_header = [
    'gann-network.h',
    'gann-layer.h',
    'gann-context.h',
    'gann-input-layer.h',
    'gann-output-layer.h',
    'gann-dense-layer.h',
    'gann-conv-layer.h',
    'gann-buffer.h',
    'gann-barrier.h',
]

dependencies = [
    ganncore_dep,
    gobject_dep,
    opencl_dep,
]

gir_includes = [
  'GObject-2.0',
  'GLib-2.0',
]

libgann = shared_library(meson.project_name(),
                         source,
                         install: true,
                         dependencies: dependencies)

introspection = gnome.generate_gir(libgann,
                                   sources: [ source, public_header ],
                                   includes: gir_includes,
                                   namespace: 'Gann',
                                   nsversion: '1.0',
                                   header: 'gann.h',
                                   install: true,
                                   dependencies: dependencies)

vapi = gnome.generate_vapi('gann',
                           sources: introspection.get(0),
                           install: true)

gann_dep = declare_dependency(link_with: libgann,
                              dependencies: [ vapi ],
                              include_directories: '.')
