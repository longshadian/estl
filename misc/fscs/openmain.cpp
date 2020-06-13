#include <cstdio>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <chrono>
#include <thread>

#include <cstring>
#include <string>

#include "SharedMemoryAPI.h"


int fun()
{
	int fd = SharedMemoryAPI::openSharedMem(SharedMemoryAPI::name);
	if (fd < 0) {
		printf("ERROR openSharedMem %d\n", fd);
		return 0;
	}


	//void* addr = SharedMemoryAPI::mapSharedMem(fd, SharedMemoryAPI::MEM_SIZE, PROT_READ);
	void* addr = SharedMemoryAPI::mapSharedMem(fd, SharedMemoryAPI::MEM_SIZE);
	void* pos = addr;

	SharedMemoryAPI::SharedMemMutex* pMtx = SharedMemoryAPI::openSharedMemLock(&addr);

	printf("OK %p %p %ld\n", pos, addr, sizeof(SharedMemoryAPI::SharedMemMutex));

	std::string str = "abcdefg\n";
	std::memcpy(addr, str.c_str(), str.length());
	pMtx->post();

	pMtx->wait();
    std::this_thread::sleep_for(std::chrono::seconds(3));

	//write(STDOUT_FILENO, addr, SharedMemoryAPI::MEM_SIZE - sizeof(SharedMemoryAPI::SharedMemMutex));
	
	printf("OK %p\n", addr);
    return 0;
}

int fun1()
{
    const char* name = "fscs1.smem";
    int fd = shm_open(name, O_EXCL | O_RDWR, 0777);
	if (fd < 0) {
		printf("ERROR shm_open %d\n", fd);
        return 0;
	}

    void* addr = mmap(NULL, 1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    sem_t* pMtx = (sem_t*)addr;
    addr = (char*)addr + sizeof(sem_t);

	std::string str = "abcdefg\n";
	std::memcpy(addr, str.c_str(), str.length());

	printf("OK %p %p\n", pMtx, addr);

	sem_post(pMtx);

    sem_wait(pMtx);
	printf("OK %p\n", addr);
    return 0;
}


int main(int argc, char** argv)
{
    fun1();
	return 0;
}