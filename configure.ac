#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])

########################################
# Generate releaes version
########################################

m4_include([maint/version.m4])
dnl 2nd arg is intentionally underquoted
AC_INIT([HT],
        HT_VERSION_m4,
        [yguo@anl.gov],
        [ht],
        [https://github.com/yfguo/hetero-trigger])

# build info
CONFIGURE_ARGS_CLEAN=`echo $* | tr '"' ' '`

# these values come from the m4_include above
HT_VERSION=HT_VERSION_m4
AC_SUBST(HT_VERSION)
HT_RELEASE_DATE="HT_RELEASE_DATE_m4"
AC_SUBST(HT_RELEASE_DATE)

if test -z "$HT_VERSION" ; then
    AC_MSG_ERROR([HT_VERSION is empty, check maint/version.m4 for errors])
fi

# Produce a numeric version assuming the following format:
# Version: [MAJ].[MIN].[REV][EXT][EXT_NUMBER]
# Example: 1.0.7rc1 has
#          MAJ = 1
#          MIN = 0
#          REV = 7
#          EXT = rc
#          EXT_NUMBER = 1
#
# Converting to numeric version will convert EXT to a format number:
#          ALPHA (a) = 0
#          BETA (b)  = 1
#          RC (rc)   = 2
#          PATCH (p) = 3
# Regular releases are treated as patch 0
#
# Numeric version will have 1 digit for MAJ, 2 digits for MIN,
# 2 digits for REV, 1 digit for EXT and 2 digits for EXT_NUMBER.
changequote(<<,>>)
V1=`expr $HT_VERSION : '\([0-9]*\)\.[0-9]*\.*[0-9]*[a-zA-Z]*[0-9]*'`
V2=`expr $HT_VERSION : '[0-9]*\.\([0-9]*\)\.*[0-9]*[a-zA-Z]*[0-9]*'`
V3=`expr $HT_VERSION : '[0-9]*\.[0-9]*\.*\([0-9]*\)[a-zA-Z]*[0-9]*'`
V4=`expr $HT_VERSION : '[0-9]*\.[0-9]*\.*[0-9]*\([a-zA-Z]*\)[0-9]*'`
V5=`expr $HT_VERSION : '[0-9]*\.[0-9]*\.*[0-9]*[a-zA-Z]*\([0-9]*\)'`
changequote([,])

if test "$V2" -le 9 ; then V2=0$V2 ; fi
if test "$V3" = "" ; then V3=0; fi
if test "$V3" -le 9 ; then V3=0$V3 ; fi
if test "$V4" = "a" ; then
    V4=0
elif test "$V4" = "b" ; then
    V4=1
elif test "$V4" = "rc" ; then
    V4=2
elif test "$V4" = "" ; then
    V4=3
    V5=0
elif test "$V4" = "p" ; then
    V4=3
fi
if test "$V5" -le 9 ; then V5=0$V5 ; fi

HT_NUMVERSION=`expr $V1$V2$V3$V4$V5 + 0`
AC_SUBST(HT_NUMVERSION)


########################################
# Check and configure setup
########################################

AC_CONFIG_AUX_DIR(confdb)
AC_CONFIG_MACRO_DIR(confdb)

# needed by hwloc in embedded mode.  Must come very early to avoid
# bizarre expansion ordering warnings
AC_CANONICAL_TARGET
AC_ARG_PROGRAM

# also needed by hwloc in embedded mode, must also come early for expansion
# ordering reasons
AC_USE_SYSTEM_EXTENSIONS

AM_INIT_AUTOMAKE([-Wall -Werror -Wno-portability-recursive silent-rules foreign 1.12.3 subdir-objects])

# Bug in libtool adds -O2 and -g by default
PAC_PUSH_FLAG([CFLAGS])
AC_PROG_CC(mpicc)
PAC_POP_FLAG([CFLAGS])

# suppress default "-g -O2" from AC_PROG_CXX
: ${CXXFLAGS=""}
# Bug in automake always check dependency even AC_PROG_CXX is conditionally checked.
AC_PROG_CXX(mpicxx)
AM_PROG_AR

LT_PREREQ([2.2.6])

# Set the appropriate macro for different platform
AS_CASE([$host_os],
    [darwin*],
        [AC_DEFINE(USE_OSX,1,[Define on Apple OSX.])],
    [linux*],
        [AC_DEFINE(USE_LINUX,1,[Define on Linux.])],
    [freebsd*],
        [AC_DEFINE(USE_FREEBSD,1,[Define on FreeBSD.])],
    [AC_MSG_ERROR([Unsupported operating system.])]
)

AC_CONFIG_HEADER([include/htconf.h])

# Bug in libtool adds -O2 and -g by default
PAC_PUSH_FLAG([CFLAGS])
LT_INIT()
PAC_POP_FLAG([CFLAGS])
 
CONFIGURE_ARGS_CLEAN="$CONFIGURE_ARGS_CLEAN CC=$CC CFLAGS=$CFLAGS "
CONFIGURE_ARGS_CLEAN="$CONFIGURE_ARGS_CLEAN LDFLAGS=$LDFLAGS LIBS=$LIBS CPPFLAGS=$CPPFLAGS  "
CONFIGURE_ARGS_CLEAN="$CONFIGURE_ARGS_CLEAN $BUILD_INFO LT_SYS_LIBRARY_PATH=$LT_SYS_LIBRARY_PATH CPP=$CPP"
AC_SUBST(CONFIGURE_ARGS_CLEAN)

## Check for C99
AC_PROG_CC_C99
if test "$ac_cv_prog_cc_c99" = "no" ; then
  AC_ERROR([C99 not supported by the compiler])
fi

# Checks for header files.
AC_CHECK_HEADERS([stdlib.h math.h mpi.h])

# Checks for library functions.
AC_CHECK_FUNCS([memset memcpy atol atexit strtof ceil])

# Non-verbose make
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_CONST
AC_C_RESTRICT
AC_C_INLINE

# Get top srcdir
# This test is complicated by the fact that top_srcdir is not set until
# the very end of configure.  Instead, we get it ourselves
if test -z "$top_srcdir" ; then
    ht_top_srcdir=$srcdir   
else
    ht_top_srcdir=$top_srcdir
fi
if test -z "$ht_top_srcdir" ; then 
    # This needs to be an absolute pathname
    case "$ht_top_srcdir" in
    /*) ;;
    *)
        ht_top_srcdir=`(cd $ht_top_srcdir && pwd)`
    ;;  
    esac
fi

# Get the directory that we're running in...
if test -z "$ht_top_builddir" ; then
   ht_top_builddir="`pwd`"
fi

AC_ARG_ENABLE(fast,
[  --enable-fast=[option]
                         Options to enable fast execution of HT implementation.
                         Supported options include (select multiple options separated by commas):
                            O<n>   - Appends optimization flag -O<n> to CFLAGS. (default -O2)
                            ndebug - Appends "-DHT_DISABLE_DEBUG -DHT_DISABLE_ERROR_CHECKING" to CFLAGS.
                                     It disables runtime control of debug message and disables all internal
                                     error checking.
                            ipo    - Appends "-DHT_ENABLE_IPO -ipo" to CFLAGS.
                            most|yes   - "O2" and "ndebug" are enabeld
                            all    - "most" and "ipo" are enabled
                            none   - None of above options, i.e. --disable-fast],
                          [ enable_fast=$enableval ],
                          [ enable_fast=O2 ])
AC_MSG_CHECKING(fast build support)
AC_MSG_RESULT($enable_fast)

enable_fast_opts=O2
enable_fast_ndebug=no
enable_fast_ipo=no
# strip off multiple options, separated by commas
save_IFS="$IFS"
IFS=","
for option in $enable_fast ; do
    case "$option" in
        O*)
        enable_fast_opts=$option
        ;;
        ndebug)
        enable_fast_ndebug=yes
        ;;
        ipo)
        enable_fast_ipo=yes
        ;;
        most|yes)
        enable_fast_ndebug=yes
        enable_fast_opts=O2
        ;;
        all)
        enable_fast_ndebug=yes
        enable_fast_opts=O2
        enable_fast_ipo=yes
        ;;
        none|no)
        enable_fast_ndebug=no
        enable_fast_opts=
        ;;
        *)
  IFS="$save_IFS"
        AC_MSG_WARN([Unknown value $option for --enable-fast])
  IFS=","
        ;;
    esac
done
IFS="$save_IFS"

if test -n "$enable_fast_opts" ; then
   # Allows O<n> where <n> can be [0-9] or ' '.
   opt_flags=`echo $enable_fast_opts | sed -e 's%\(O[0-9] \)%\1%g'`
   if test -n "$opt_flags" ; then
      PAC_APPEND_FLAG([-$enable_fast_opts -DHT_FAST_OPTS=$enable_fast_opts],[CFLAGS])
   else
      AC_MSG_WARN([Unknown value $enable_fast_opts for --enable-fast])
   fi
fi

if test "$enable_fast_ndebug" = "yes" ; then
    PAC_APPEND_FLAG([-DHT_DISABLE_DEBUG -DHT_DISABLE_ERROR_CHECKING],[CFLAGS])
fi

if test "$enable_fast_ipo" = "yes" ; then
    PAC_APPEND_FLAG([-DHT_ENABLE_IPO -ipo],[CFLAGS])
fi

## Debugging support
AC_ARG_ENABLE(g, AC_HELP_STRING([--enable-g],[Turn on debugging [default=no]]),
                                [ enable_debug=$enableval ],
                                [ enable_debug=no ])
AC_MSG_CHECKING(debugging support)
AC_MSG_RESULT($enable_debug)
# always disable debug if enable_fast is set
if test "$enable_fast" = "no"; then
  if test "$enable_debug" = "yes"; then
    PAC_APPEND_FLAG([-g -O0 -DHT_ENABLE_DBG],[CFLAGS])
  fi
fi

# Check for enabling strict
PAC_ARG_STRICT

# Thread package used in critical section and asynchronous thread
AC_ARG_WITH([thread-package],
[  --with-thread-package=posix|pthread
                          Thread package to implement internal critical section 
                          when multiple threads are present. Supported thread 
                          packages include:
                          posix or pthreads - POSIX threads (default)
],,with_thread_package=default)
AC_MSG_CHECKING(Thread package to implement critical section)
AC_MSG_RESULT($with_thread_package)

if test "$enable_threads" == "multiple"; then
    THREAD_PACKAGE_NAME=NONE
    case $with_thread_package in
        default|posix|pthreads)
            # check if pthread is supported (e.g., invalid on windows or solaris)
            have_pthreads=no
            AC_CHECK_HEADERS(pthread.h)
            AC_CHECK_LIB([pthread],[pthread_create],have_lpthread=yes)
            if test "$have_lpthread" = "yes" ; then
                PAC_PREPEND_FLAG([-lpthread],[LIBS])
                # this check should come after the AC_CHECK_LIB for -lpthread
                AC_CHECK_FUNCS([pthread_mutex_lock],have_pthreads=yes,AC_MSG_ERROR([unable to find pthreads library.]))
            fi
            THREAD_PACKAGE_NAME=HT_THREAD_CS_LOCK__PTHREAD_MUTEX
        ;;
        *)
            AC_MSG_ERROR([The specified thread package, $with_thread_package, is not supported.])
        ;;
    esac
    AC_DEFINE_UNQUOTED(HT_THREAD_CS_LOCK,[$THREAD_PACKAGE_NAME],
                       [Thread package to implement critical section])
fi

AC_CHECK_HEADERS(sys/mman.h sys/stat.h sys/time.h fcntl.h)
AC_CHECK_FUNC(mmap, [HAVE_MMAP=1], [HAVE_MMAP=0])

gl_FUNC_RANDOM_R
if test "$HAVE_RANDOM_R" = "1" -a "$HAVE_STRUCT_RANDOM_DATA" = "1" -a "$HAVE_MMAP" = "1"; then
    AC_DEFINE(HT_ENABLE_SYMM_ALLOC,1,[Define if we can enable symmatric allocatio])
    AC_MSG_NOTICE([Enable symmetric allocation])
else
    AC_MSG_NOTICE([Disable symmetric allocation])
fi

# --with-cuda
PAC_SET_HEADER_LIB_PATH([cuda])
if test "$with_cuda" != "no" ; then
    PAC_CHECK_HEADER_LIB([cuda_runtime_api.h],[cudart],[cudaStreamSynchronize],[have_cuda=yes],[have_cuda=no])
    if test "${have_cuda}" = "yes" ; then
        AC_MSG_CHECKING([whether nvcc works])
        if test -n "$ac_save_CC" ; then
            NVCC_FLAGS="$NVCC_FLAGS -ccbin $ac_save_CC"
            # - pgcc/nvc doesn't work, use pgc++/nvc++ instead
            # - Extra optins such as `gcc -std=gnu99` doesn't work, strip the option
            NVCC_FLAGS=$(echo $NVCC_FLAGS | sed -e 's/nvc/nvc++/g' -e 's/pgcc/pgc++/g' -e's/ -std=.*//g')
        fi
        # try nvcc from PATH if 'with-cuda' does not contain a valid path
        if test -d ${with_cuda} ; then
            nvcc_bin=${with_cuda}/bin/nvcc
        else
            nvcc_bin=nvcc
        fi

        # save language settings, customize ac_ext and ac_compile to support CUDA
        AC_LANG_PUSH([C])
        ac_ext=cu
        ac_compile='$nvcc_bin $NVCC_FLAGS -c conftest.$ac_ext >&5'
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([__global__ void foo(int x) {}],[])],
        [
            AC_DEFINE([HAVE_CUDA],[1],[Define is CUDA is available])
            AS_IF([test -d "${with_cuda}"],[NVCC=${with_cuda}/bin/nvcc],[NVCC=nvcc])
            AC_SUBST(NVCC)
            AC_SUBST(NVCC_FLAGS)
            AC_MSG_RESULT([yes])
        ],[
            have_cuda=no
            AC_MSG_RESULT([no])
            AC_MSG_ERROR([CUDA was not functional with provided host compiler (CC)])
        ])
        # done with CUDA, back to C
        AC_LANG_POP([C])

        # nvcc compiled applications need libstdc++ to be able to link
        # with a C compiler
        AC_MSG_CHECKING([if $CC can link libstdc++])
        PAC_PUSH_FLAG([LIBS])
        PAC_APPEND_FLAG([-lstdc++],[LIBS])
        AC_LINK_IFELSE(
            [AC_LANG_PROGRAM([int x = 5;],[x++;])],
            [libstdcpp_works=yes],
            [libstdcpp_works=no])
        PAC_POP_FLAG([LIBS])
        if test "${libstdcpp_works}" = "yes" ; then
            PAC_APPEND_FLAG([-lstdc++],[LIBS])
            AC_MSG_RESULT([yes])
        else
            have_cuda=no
            AC_MSG_RESULT([no])
        fi
    fi
    if test "${have_cuda}" = "no" -a "$with_cuda" != ""; then
        AC_MSG_ERROR([CUDA was requested but it is not functional])
    fi
fi

# --with-hip
PAC_SET_HEADER_LIB_PATH([hip])
if test "$with_hip" != "no" ; then
    PAC_PUSH_FLAG(CPPFLAGS)
    PAC_APPEND_FLAG([-D__HIP_PLATFORM_AMD__], [CPPFLAGS])
    PAC_CHECK_HEADER_LIB([hip/hip_runtime_api.h],[amdhip64],[hipStreamSynchronize],[have_hip=yes],[have_hip=no])
    PAC_POP_FLAG(CPPFLAGS)
    if test "${have_hip}" = "yes" ; then
        AC_MSG_CHECKING([whether hipcc works])
        cat>conftest.c<<EOF
        #include <hip/hip_runtime_api.h>
        void foo(int deviceId) {
        hipError_t ret;
        ret = hipGetDevice(&deviceId);
        }
EOF
        PAC_PUSH_FLAG(CPPFLAGS)
        PAC_APPEND_FLAG([-D__HIP_PLATFORM_AMD__], [CPPFLAGS])
        ${with_hip}/bin/hipcc -c conftest.c 2> /dev/null
        PAC_POP_FLAG(CPPFLAGS)
        if test "$?" = "0" ; then
            AC_DEFINE([HAVE_HIP],[1],[Define is HIP is available])
            PAC_APPEND_FLAG([-D__HIP_PLATFORM_AMD__], [CPPFLAGS])
            AS_IF([test -n "${with_hip}"],[HIPCC=${with_hip}/bin/hipcc],[HIPCC=hipcc])
            AC_SUBST(HIPCC)
            # hipcc compiled applications need libstdc++ to be able to link
            # with a C compiler
            PAC_PUSH_FLAG([LIBS])
            PAC_APPEND_FLAG([-lstdc++],[LIBS])
            AC_LINK_IFELSE(
                [AC_LANG_PROGRAM([int x = 5;],[x++;])],
                [libstdcpp_works=yes],
                [libstdcpp_works=no])
            PAC_POP_FLAG([LIBS])
            if test "${libstdcpp_works}" = "yes" ; then
                PAC_APPEND_FLAG([-lstdc++],[LIBS])
                AC_MSG_RESULT([yes])
            else
                have_hip=no
                AC_MSG_RESULT([no])
                AC_MSG_ERROR([hipcc compiled applications need libstdc++ to be able to link with a C compiler])
            fi
        else
            have_hip=no
            AC_MSG_RESULT([no])
        fi
        rm -f conftest.*
    fi
fi

# check for attribute support
PAC_C_GNU_ATTRIBUTE

AC_SUBST(WRAPPER_LDFLAGS)
AC_SUBST(WRAPPER_LIBS)

AC_CONFIG_FILES([Makefile])
AC_OUTPUT
