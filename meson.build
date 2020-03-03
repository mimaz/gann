project('gann', 'c')

cc = meson.get_compiler('c')
gnome = import('gnome')

glib_dep = dependency('glib-2.0')
gio_dep = dependency('gio-2.0')
gobject_dep = dependency('gobject-2.0')
opencl_dep = dependency('OpenCL')
math_dep = cc.find_library('m')

subdir('lib/')
subdir('bin/')
