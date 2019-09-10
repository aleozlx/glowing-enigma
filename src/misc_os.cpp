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


    ProcessPool::ProcessPool(size_t nproc) : nproc(nproc) {

    }

    int ProcessPool::fork() {
        if (nproc<=1) // sequential execution
            return 0;
        for (int i = 0; i < nproc; ++i) {
            pid_t child = ::fork();
            if (child < 0) {
//                    std::cerr<<"fork() error"<<std::endl;
                return -2;
            } else if (child == 0) return i;
        }

        while (waitpid(-1, nullptr, 0) > 0);
        return -1;
    }

    ScopedProcess::ScopedProcess(int tid, size_t nproc) : tid(tid), nproc(nproc) {
        this->pid = ::getpid();
    }

    ScopedProcess::~ScopedProcess() {
        if (this->isChild() && nproc>1) // destroy the process itself
            ::_exit(0);
    }
}