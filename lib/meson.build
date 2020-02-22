source = [
  'network.c',
  'layer.c',
  'convolution-layer.c',
  'activation-layer.c',
  'input-layer.c',
]

dependencies = [
  glib_dep,
  opencl_dep,
]

libgann = shared_library(
  meson.project_name(),
  source,
  install: true,
  dependencies: dependencies,
)
