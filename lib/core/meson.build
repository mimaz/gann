source = [
    'network.c',
    'layer.c',
    'fully-layer.c',
    'input-layer.c',
    'output-layer.c',
    'context.c',
]

resource = gnome.compile_resources(
  'cl', 'cl-gresource.xml',
  source_dir: 'cl',
  c_name: 'cl',
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
