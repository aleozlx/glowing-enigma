#ifndef __FEATURES_HPP__
#define __FEATURES_HPP__
#define VERHELPER(x) #x
#define VERSTR(x) VERHELPER(x)
#endif

// Required:
FEATURE_VER(VER_OPENCV);

// Optional:
#ifdef HAS_CUDA
    FEATURE_VER(HAS_CUDA);
#endif
#ifdef HAS_TF
    FEATURE_VER(HAS_TF);
#endif
#ifdef HAS_LIBGSLIC
    FEATURE(HAS_LIBGSLIC);
#endif
#ifdef HAS_SPFREQ2
    FEATURE_VER(HAS_SPFREQ2);
#endif

