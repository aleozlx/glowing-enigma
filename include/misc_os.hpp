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

extern "C" {
#include <unistd.h>
#include <sys/wait.h>
}

namespace os_misc {
    class ProcessPool {
    protected:
        size_t nproc;
    public:
        ProcessPool(size_t nproc);

        int fork();
    };

    struct ScopedProcess {
        size_t nproc;
        pid_t pid;
        int tid;

        inline bool isChild() {
            return tid >= 0;
        }

        ScopedProcess(int tid, size_t nproc);

        ~ScopedProcess();
    };
}

#endif //GLOWING_ENIGMA_MISC_OS_HPP
