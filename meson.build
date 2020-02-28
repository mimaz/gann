project('gann', 'c')

cc = meson.get_compiler('c')

glib_dep = dependency('glib-2.0')
gobject_dep = dependency('gobject-2.0')
math_dep = cc.find_library('m')

subdir('lib/')
subdir('bin/')
