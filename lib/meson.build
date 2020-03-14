subdir('core/')

source = [
    'gann-network.c',
    'gann-layer.c',
    'gann-context.c',
    'gann-input-layer.c',
    'gann-output-layer.c',
    'gann-dense-layer.c',
    'gann-conv-layer.c',
]

public_header = [
    'gann-network.h',
    'gann-layer.h',
    'gann-context.h',
    'gann-input-layer.h',
    'gann-output-layer.h',
    'gann-dense-layer.h',
    'gann-conv-layer.h',
]

dependencies = [
    ganncore_dep,
    gobject_dep,
]

libgann = shared_library(meson.project_name(),
                         source,
                         install: true,
                         dependencies: dependencies)

introspection = gnome.generate_gir(libgann,
                                   sources: [ source, public_header ],
                                   includes: [ 'GObject-2.0' ],
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
