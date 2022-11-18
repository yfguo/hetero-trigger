#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS += -I$(top_builddir)/include -I$(top_srcdir)/include

EXTRA_DIST += $(top_srcdir)/src/cudalt.sh

libht_la_SOURCES += \
	src/ht_flag.c \
	src/ht_queue.c

if HAVE_CUDA
libht_la_SOURCES += \
	src/ht_op.cu
.cu.lo:
	@if $(AM_V_P) ; then \
		$(top_srcdir)/src/cudalt.sh --verbose $@ \
			$(NVCC) $(NVCC_FLAGS) $(AM_CPPFLAGS) $(CUDA_GENCODE) -c $< ; \
	else \
		echo "  NVCC     $@" ; \
		$(top_srcdir)/src/cudalt.sh $@ $(NVCC) $(NVCC_FLAGS) $(AM_CPPFLAGS) $(CUDA_GENCODE) -c $< ; \
	fi
else
if HAVE_HIP
libht_la_SOURCES += \
	src/ht_op.hip
.hip.lo:
	@if $(AM_V_P) ; then \
		$(top_srcdir)/src/cudalt.sh --verbose $@ \
			$(NVCC) $(NVCC_FLAGS) $(AM_CPPFLAGS) $(CUDA_GENCODE) -c $< ; \
	else \
		echo "  NVCC     $@" ; \
		$(top_srcdir)/src/cudalt.sh $@ $(NVCC) $(NVCC_FLAGS) $(AM_CPPFLAGS) $(CUDA_GENCODE) -c $< ; \
	fi
else
if HAVE_ZE
endif
endif
endif
