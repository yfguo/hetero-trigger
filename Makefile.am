#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

ACLOCAL_AMFLAGS = -I confdb
AM_CPPFLAGS =

lib_LTLIBRARIES = libht.la
include_HEADERS = include/ht_flag.h       \
                  include/ht_op.h

external_subdirs =
external_ldflags = -lcuda -lcudart
convenience_libs =

noinst_HEADERS =
bin_SCRIPTS =
CLEANFILES =
libht_la_SOURCES =
libht_la_LDFLAGS = $(external_ldflags)
libht_la_LIBADD = $(convenience_libs)
EXTRA_libht_la_DEPENDENCIES = $(convenience_libs)
EXTRA_DIST =
EXTRA_PROGRAMS =

include src/Makefile.mk
include test/Makefile.mk

EXTRA_DIST += CHANGES COPYRIGHT maint

DIST_SUBDIRS = test ${external_subdirs}
SUBDIRS = ${external_subdirs}
