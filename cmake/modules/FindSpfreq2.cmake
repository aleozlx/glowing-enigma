# Find SPFREQ2 shared object (.so)
#  SPFREQ2_LIBRARIES    - List of libraries when using SPFREQ2.
#  SPFREQ2_FOUND        - True if SPFREQ2 found.

FIND_LIBRARY(SPFREQ2_LIBRARY spfreq2_op.so /usr/lib /mnt/builds/spfreq2 /tank/builds/spfreq2)
MARK_AS_ADVANCED(SPFREQ2_LIBRARY)

SET(SPFREQ2_LIBRARIES  "${SPFREQ2_LIBRARY}")
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(Spfreq2 DEFAULT_MSG SPFREQ2_LIBRARIES)
