##
## Copyright (C) by Argonne National Laboratory
##     See COPYRIGHT in top-level directory
##

LDADD =  libht.la
test_cppflags = -I$(build_dir)/include -I$(srcdir)

EXTRA_PROGRAMS += \
	test/set_perf

test_set_perf_CPPFLAGS = $(test_cppflags)

# test-set-perf:
# 	@$(top_srcdir)/test/runtests.py --summary=$(top_builddir)/test/simple/summary.junit.xml \
#                 test/simple/testlist.gen

CLEANFILES += $(EXTRA_PROGRAMS)
