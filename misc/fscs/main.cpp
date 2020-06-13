#include <cstdio>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>
#include <sched.h>

#include <thread>
#include <chrono>

#include <string>
#include <cstring>

#include "SharedMemoryAPI.h"


int fun()
{
	int fd = SharedMemoryAPI::createSharedMem(SharedMemoryAPI::name);
	if (fd < 0) {
		printf("ERROR createSharedMem %d\n", fd);
		return 0;
	}
	
	if (ftruncate(fd, SharedMemoryAPI::MEM_SIZE) == -1) {
		printf("ERROR ftruncate %d\n", errno);
		return 0;
	}

	void* addr = SharedMemoryAPI::mapSharedMem(fd, SharedMemoryAPI::MEM_SIZE);
	if (addr == MAP_FAILED) {
		printf("ERROR mapSharedMem %d\n", errno);
		return 0;
	}

	void* pos = addr;
	SharedMemoryAPI::SharedMemMutex* pMtx = SharedMemoryAPI::createSharedMemLock(&addr);
	if (!pMtx) {
		printf("ERROR createSharedMemLock %d\n", errno);
		return 0;
	}
	printf("OK %p %p\n", pos, addr);

	std::string str = "1234567890123456789012345678901234567890\n";
	std::memcpy(addr, str.c_str(), str.length());

	int n = 10;
	while (n--) {
		pMtx->wait();
		write(STDOUT_FILENO, addr, SharedMemoryAPI::MEM_SIZE - sizeof(SharedMemoryAPI::SharedMemMutex));
        int val = 0;
        bool b = pMtx->getValue(&val);
        printf("post 1 %d %d %d\n", n,b, val);
		pMtx->post();

        val = 0;
        b = pMtx->getValue(&val);
        printf("post 2 %d %d %d\n", n,b, val);
        //std::this_thread::sleep_for(std::chrono::microseconds(100));
        //sched_yield();
	}

	printf("OK %p\n", addr);
	return 0;
}

int fun1()
{
    const char* name = "fscs1.smem";
    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0777);
	if (fd < 0) {
		printf("ERROR shm_open %d\n", fd);
		return 0;
	}
	
	if (ftruncate(fd, 1000) == -1) {
		printf("ERROR ftruncate %d\n", errno);
		return 0;
	}

    void* addr = mmap(NULL, 1000, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (addr == MAP_FAILED) {
		printf("ERROR mmap %d\n", errno);
		return 0;
	}

    if (sem_init((sem_t*)addr, 1, 0) == -1) {
		printf("ERROR sem_init %d\n", errno);
        return 0;
    }
    sem_t* pMtx = (sem_t*)addr;
    addr = (char*)addr + sizeof(sem_t);

	int n = 3;
	while (n--) {
        sem_wait(pMtx);
		write(STDOUT_FILENO, addr, 1000-sizeof(sem_t));
        sem_post(pMtx);
	}
	return 0;
}


int main(int argc, char** argv)
{
    fun1();
	return 0;
}