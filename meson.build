project('gann', 'vala', 'c')

cc = meson.get_compiler('c')
gnome = import('gnome')

glib_dep = dependency('glib-2.0')
gio_dep = dependency('gio-2.0')
gobject_dep = dependency('gobject-2.0')
opencl_dep = dependency('OpenCL')
clutter_dep = dependency('clutter-1.0')
gtk_dep = dependency('gtk4')
pixbuf_dep = dependency('gdk-pixbuf-2.0')
math_dep = cc.find_library('m')

subdir('lib/')
subdir('bin/')
