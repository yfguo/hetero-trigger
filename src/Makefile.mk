#
# Copyright (C) 2018. See COPYRIGHT in top-level directory.
#

AM_CPPFLAGS += -I$(top_builddir)/include -I$(top_srcdir)/include

EXTRA_DIST += $(top_srcdir)/src/cudalt.sh
EXTRA_DIST += $(top_srcdir)/src/hiplt.sh

libht_la_SOURCES += \
	src/ht_flag.c \
	src/ht_queue.c \
	src/ht_op.cu \
	src/ht_op.hip

.cu.lo:
	@if $(AM_V_P) ; then \
		$(top_srcdir)/src/cudalt.sh --verbose $@ \
			$(NVCC) $(NVCC_FLAGS) $(AM_CPPFLAGS) $(CUDA_GENCODE) -c $< ; \
	else \
		echo "  NVCC     $@" ; \
		$(top_srcdir)/src/cudalt.sh $@ $(NVCC) $(NVCC_FLAGS) $(AM_CPPFLAGS) $(CUDA_GENCODE) -c $< ; \
	fi

.hip.lo:
	@if $(AM_V_P) ; then \
		$(top_srcdir)/src/backend/hip/hiplt.sh --verbose $@ \
			$(HIPCC) $(AM_CPPFLAGS) -g $(HIP_GENCODE) -c $< ; \
	else \
		echo "  HIPCC     $@" ; \
		$(top_srcdir)/src/backend/hip/hiplt.sh $@ $(HIPCC) $(AM_CPPFLAGS) -g $(HIP_GENCODE) -c $< ; \
	fi

