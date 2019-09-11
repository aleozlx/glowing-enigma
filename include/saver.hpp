#ifndef __SAVER_HPP__
#define __SAVER_HPP__
#include <string>

extern "C" {
#include "fpconv.h"
}

#define dtoa_(fp, dest) fpconv_dtoa(fp, dest)
#define MAX_LEN_DTOA (24)
#define RESERVE_VEC2STR(dim) ((MAX_LEN_DTOA+1)*(dim)+4) // (DTOA+SEP)*DIM+PADDING

namespace spt::pgsaver {
    template<typename F>
    size_t vec2cstr(size_t dim, F *vec, char *dst) {
        if (dim <= 0) return 0;
        char *dst0 = dst;
        *dst++ = '{';
        do {
            dst += dtoa_(static_cast<double>(*vec), dst);
            *dst++ = ',';
            vec++;
        } while (--dim);
        dst[-1] = '}';
        dst[0] = '\0';
        return dst - dst0;
    }

    template<typename V>
    void vec2str(V vec, std::string &dst) {
        size_t dim = vec.size();
        dst.resize(RESERVE_VEC2STR(dim));
        size_t len = vec2cstr(dim, vec.data(), const_cast<char *>(dst.data()));
        dst.resize(len);
    }

    template<typename V>
    void vec2str(V vec, size_t offset, size_t dim, std::string &dst) {
        dst.resize(RESERVE_VEC2STR(dim));
        size_t len = vec2cstr(dim, vec.data()+offset, const_cast<char *>(dst.data()));
        dst.resize(len);
    }
}
#endif
