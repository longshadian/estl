#include "SharedMemoryAPI.h"
#include <stdlib.h>

namespace SharedMemoryAPI
{
const char* name = "fscs1.smem";

const size_t MEM_SIZE = 1000;


bool SharedMemMutex::post()
{
	return sem_post(&mSem) == 0;
}

bool SharedMemMutex::wait()
{
	return sem_wait(&mSem) == 0;
}

bool SharedMemMutex::trywait()
{
	return sem_trywait(&mSem) == 0;
}

bool SharedMemMutex::timedwait(const struct timespec* abs_timeout)
{
	return sem_timedwait(&mSem, abs_timeout) == 0;
}

bool SharedMemMutex::getValue(int* val)
{
    return sem_getvalue(&mSem, val) == 0;
}

/// 
int createSharedMem(const char* name, int oflag, mode_t mod)
{
	mod = 0777;
	int fd = -1;
	fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, mod);
	return fd;
}

int openSharedMem(const char* name, int oflag, mode_t mod)
{
	mod = 0777;
	int fd = -1;
	fd = shm_open(name, O_EXCL|O_RDWR, mod);
	return fd;
}


int deleteSharedMem(const char* name)
{
	return shm_unlink(name);
}

void* mapSharedMem(int fd, size_t size, int oflag)
{
	if (oflag == 0) {
		oflag = PROT_READ | PROT_WRITE;
	}
	void* addr = mmap(NULL, size, oflag, MAP_SHARED, fd, 0);
	return addr;
}

SharedMemMutex* createSharedMemLock(void** addr, int pshared, int value)
{
	SharedMemMutex* pMtx = (SharedMemMutex*)(*addr);
	if (sem_init(&pMtx->mSem, pshared, value) == -1) {
		return NULL;
	}
	*addr = (char*)(*addr) + sizeof(SharedMemMutex);
	return pMtx;
}

SharedMemMutex* openSharedMemLock(void**addr)
{
	SharedMemMutex* pMtx = (SharedMemMutex*)(*addr);
	*addr = (char*)(*addr) + sizeof(SharedMemMutex);
	return pMtx;
}

bool destroySharedMemLock(SharedMemMutex* pLock)
{
	return sem_destroy(&pLock->mSem) == 0;
}
}