subdir('core/')

source = [
    'gann-network.c',
    'gann-layer.c',
    'gann-input-layer.c',
    'gann-output-layer.c',
    'gann-dense-layer.c',
    'gann-context.c',
]

public_header = [
    'gann-network.h',
    'gann-layer.h',
    'gann-input-layer.h',
    'gann-output-layer.h',
    'gann-dense-layer.h',
]

dependencies = [
    ganncore_dep,
    gobject_dep,
]

libgann = shared_library(meson.project_name(),
                         source,
                         install: true,
                         dependencies: dependencies)

gann_dep = declare_dependency(link_with: libgann,
                              include_directories: '.')
