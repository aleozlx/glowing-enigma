#include "misc_os.hpp"
#include <iostream>
#include <cstring>
namespace os_misc {

    Glob::Glob(const char *pattern) {
        std::memset(&glob_result, 0, sizeof(glob_result));
        if (glob(pattern, GLOB_TILDE, NULL, &glob_result) != 0) {
            std::cerr << "glob() error" << std::endl;
            _error = true;
        }
    }

    Glob::~Glob() {
        globfree(&glob_result);
    }

    const char *Glob::operator[](int i) {
        return glob_result.gl_pathv[i];
    }

    size_t Glob::size() {
        return _error ? 0 : glob_result.gl_pathc;
    }

}
