source = [
    'network.c',
    'layer.c',
    'fully-layer.c',
    'input-layer.c',
    'output-layer.c',
    'context.c',
    'util.c',
]

resource = gnome.compile_resources(
  'cl_code', 'cl-code-gresource.xml',
  c_name: 'cl_code',
)

dependencies = [
    glib_dep,
    gio_dep,
    math_dep,
    opencl_dep,
]

libganncore = shared_library(meson.project_name() + 'core',
                             [ source, resource ],
                             install: true,
                             dependencies: dependencies)

ganncore_dep = declare_dependency(link_with: libganncore,
                                  include_directories: '.')
