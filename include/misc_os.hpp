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

        size_t size() const;

        const char *operator[](int i) const;
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
        int tid;
        size_t nproc;
        pid_t pid;

        inline bool isChild() {
            return tid >= 0;
        }

        ScopedProcess(int tid, size_t nproc);

        ~ScopedProcess();
    };
}

extern "C" {
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
}

namespace os_misc {
    template<typename D>
    struct SharedMem {
        bool _init = false;
        const char *shm_name;
        const size_t shm_size;
        D *data;

        /// The ctor only creates a header; call init() for shm allocation
        SharedMem(const char *shm_name, size_t units): shm_name(shm_name), shm_size(units * sizeof(D)) {

        }

        ~SharedMem() {
            if (_init) {
                ::munmap(this->data, this->shm_size);
                ::shm_unlink(shm_name);
            }
        }

        /// Allocate memory; use placement new to construct object as needed
        void init() {
            int shm_fd = ::shm_open(shm_name, O_CREAT | O_RDWR, 0666);
            ::ftruncate(shm_fd, shm_size);
            this->data = static_cast<D *>(::mmap(0, shm_size, PROT_WRITE, MAP_SHARED, shm_fd, 0));
            _init = true;
        }
    };
}


extern "C" {
#include <semaphore.h>
#include <pthread.h>
}

namespace os_misc {
    struct IPCMutex {
        bool _init = false;
        pthread_mutex_t _mutex;
        int _err;

        IPCMutex();

        ~IPCMutex();

        inline int lock() {
            return pthread_mutex_lock(&_mutex);
        }

        inline int unlock() {
            return pthread_mutex_unlock(&_mutex);
        }
    };

    struct IPCSem {
        sem_t _sem;

        IPCSem();
        IPCSem(int pshared, unsigned int value);
        ~IPCSem();

        inline int wait() {
            return ::sem_wait(&_sem);
        }

        inline int post() {
            return ::sem_post(&_sem);
        }
    };

    // Dummy data structure used to calculate the max size of IPC objects
    union IPCMax {
        IPCMutex _mutex;
        IPCSem _sem;
    };
}
#endif //GLOWING_ENIGMA_MISC_OS_HPP
