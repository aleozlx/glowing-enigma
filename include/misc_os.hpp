#ifndef GLOWING_ENIGMA_MISC_OS_HPP
#define GLOWING_ENIGMA_MISC_OS_HPP

extern "C" {
#include <glob.h>
}

namespace os_misc {
    class Glob {
    protected:
        glob_t glob_result;
        bool _error = false;
    public:
        Glob(const char *pattern);

        ~Glob();

        size_t size();

        const char *operator[](int i);
    };

}

#endif //GLOWING_ENIGMA_MISC_OS_HPP
