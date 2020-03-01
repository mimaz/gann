source = [
    'network.c',
    'layer.c',
    'fully-layer.c',
    'input-layer.c',
    'output-layer.c',
    'context.c',
]

dependencies = [
    glib_dep,
    math_dep,
]

libganncore = shared_library(meson.project_name() + 'core',
                             source,
                             install: true,
                             dependencies: dependencies)

ganncore_dep = declare_dependency(link_with: libganncore,
                                  include_directories: '.')
