#pragma once

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <semaphore.h>

namespace SharedMemoryAPI {	

extern const char*	name;
extern const size_t	MEM_SIZE;

struct SharedMemMutex
{
	sem_t			mSem;
	bool			post();
	bool			wait();
	bool			trywait();
	bool			timedwait(const struct timespec* abs_timeout);
    bool            getValue(int* val);
};


int					createSharedMem(const char* name, int oflag = 0, mode_t mod = O_RDWR);
int					openSharedMem(const char* name, int oflag = 0, mode_t mod = O_RDWR);
int					deleteSharedMem(const char* name);

void*				mapSharedMem(int fd, size_t size, int oflag = 0);

SharedMemMutex*		createSharedMemLock(void** addr, int pshared = 1, int value = 0);
SharedMemMutex*		openSharedMemLock(void**addr);
bool				destroySharedMemLock(SharedMemMutex* pLock);

}
