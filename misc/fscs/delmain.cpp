#include <cstdio>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#include "SharedMemoryAPI.h"


int main(int argc, char** argv)
{
    int fd = SharedMemoryAPI::openSharedMem(SharedMemoryAPI::name);
    if (fd < 0) {
        printf("ERROR openSharedMem %d\n", fd);
        return 0;
    }


    void* addr = SharedMemoryAPI::mapSharedMem(fd, SharedMemoryAPI::MEM_SIZE);

    SharedMemoryAPI::SharedMemMutex* pMtx = SharedMemoryAPI::openSharedMemLock(&addr);
    if (!SharedMemoryAPI::destroySharedMemLock(pMtx)) {
		printf("ERROR destroySharedMemLock\n");
		return 0;
    }

	if (SharedMemoryAPI::deleteSharedMem(SharedMemoryAPI::name) == -1) {
		printf("ERROR deleteSharedMem\n");
		return 0;
	}


	printf("OK\n");
	return 0;
}