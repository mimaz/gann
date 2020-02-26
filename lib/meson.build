source = [
  'network.c',
  'layer.c',
  'convolution-layer.c',
  'fully-layer.c',
]

dependencies = [
  glib_dep,
  math_dep,
]

libgann = shared_library(
  meson.project_name(),
  source,
  install: true,
  dependencies: dependencies,
)
