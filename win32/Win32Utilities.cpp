#include "stdafx.h"

#include <windows.h>
#include <TlHelp32.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <chrono>
#include <array>
#include <vector>

#include "Win32Utilities.h"

//#include "Log.h"

#define WindowsPrintLog(fmt, ...) if (0) printf(fmt "\n", ##__VA_ARGS__)

void Win32Utilities::ThisThreadSleepMilliseconds(std::uint32_t millisec)
{
    std::this_thread::sleep_for(std::chrono::milliseconds(millisec));
}

bool Win32Utilities::StartService1(const std::string& service_name)
{
	SC_HANDLE hsc = ::OpenSCManager(NULL, NULL, GENERIC_EXECUTE);
	if (!hsc) {
        WindowsPrintLog("OpenSCManager failed");
		return false;
	}

	SC_HANDLE hsvc = ::OpenService(hsc, service_name.c_str(), SERVICE_START | SERVICE_QUERY_STATUS | SERVICE_STOP);
	if (!hsvc) {
        WindowsPrintLog("OpenService failed");
		::CloseServiceHandle(hsc);
		return false;
	}

	// 获得服务的状态
	SERVICE_STATUS status;
	if (::QueryServiceStatus(hsvc, &status) == FALSE) {
        WindowsPrintLog("QueryServiceStatus failed");
		::CloseServiceHandle(hsvc);
		::CloseServiceHandle(hsc);
		return false;
	}

	if (status.dwCurrentState == SERVICE_STOPPED) {
		// 启动服务
		if (!::StartService(hsvc, NULL, NULL)) {
            WindowsPrintLog("StartService failed");
			::CloseServiceHandle(hsvc);
			::CloseServiceHandle(hsc);
			return false;
		}

		// 等待服务启动
		while (::QueryServiceStatus(hsvc, &status) == TRUE) {
            ThisThreadSleepMilliseconds(100);
			if (status.dwCurrentState == SERVICE_RUNNING) {
				::CloseServiceHandle(hsvc);
				::CloseServiceHandle(hsc);
                break;
			}
		}
	}
    return true;
}

bool Win32Utilities::StopService(const std::string& service_name)
{
	SC_HANDLE hsc = ::OpenSCManager(NULL, NULL, GENERIC_EXECUTE);
	if (!hsc) {
        WindowsPrintLog("OpenSCManager failed");
		return false;
	}

	SC_HANDLE hsvc = ::OpenService(hsc, service_name.c_str(), SERVICE_START | SERVICE_QUERY_STATUS | SERVICE_STOP);
	if (!hsvc) {
        WindowsPrintLog("OpenService failed");
		::CloseServiceHandle(hsc);
		return false;
	}

	// 获得服务的状态
	SERVICE_STATUS status;
	if (!::QueryServiceStatus(hsvc, &status)) {
        WindowsPrintLog("QueryServiceStatus failed");
		::CloseServiceHandle(hsvc);
		::CloseServiceHandle(hsc);
		return false;
	}

	//如果处于停止状态则启动服务，否则停止服务。
	if (status.dwCurrentState == SERVICE_RUNNING) {
		if (!::ControlService(hsvc, SERVICE_CONTROL_STOP, &status)) {
            WindowsPrintLog("ControlService failed");
			::CloseServiceHandle(hsvc);
			::CloseServiceHandle(hsc);
			return false;
		}

		while (::QueryServiceStatus(hsvc, &status) == TRUE) {
            ThisThreadSleepMilliseconds(100);
			if (status.dwCurrentState == SERVICE_STOPPED) {
				::CloseServiceHandle(hsvc);
				::CloseServiceHandle(hsc);
                break;
			}
		}
	}
    return true;
}

bool Win32Utilities::StartProcess(const std::string& exe_name)
{
	STARTUPINFO si;
    std::memset(&si, 0, sizeof(si));
	si.cb = sizeof(STARTUPINFO);
	si.dwFlags = STARTF_USESHOWWINDOW;
	si.wShowWindow = SW_SHOW;

    PROCESS_INFORMATION pi;
    std::memset(&pi, 0, sizeof(pi));

    WindowsPrintLog("StartProcess %s", exe_name.c_str());
    std::array<char, 1024> cmd_line{};
	BOOL succ = ::CreateProcessA(exe_name.c_str(), cmd_line.data(), NULL, NULL, FALSE, 0, NULL, NULL, (LPSTARTUPINFOA)&si, &pi);
    return succ;
}

void Win32Utilities::KillProcess(const std::string& process_name)
{
    PROCESSENTRY32 ps;
    std::memset(&ps, 0, sizeof(ps));
	ps.dwSize = sizeof(PROCESSENTRY32);

	//遍历进程
	HANDLE hSnapshot = ::CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (hSnapshot == INVALID_HANDLE_VALUE) {
        WindowsPrintLog("CreateToolhelp32Snapshot failed");
		return;
    }

    if (!::Process32First(hSnapshot, &ps)) {
        WindowsPrintLog("Process32First failed");
		return;
    }

	do {
		//比较进程名
        std::string s = ps.szExeFile;
        if (s == process_name) {
			//找到了
			DWORD pid = ps.th32ProcessID;
			HANDLE hdProc = ::OpenProcess(PROCESS_TERMINATE, FALSE, pid);
			TerminateProcess(hdProc, 0);
			break;
		}
	} while (::Process32Next(hSnapshot, &ps));

	//没有找到
    ::CloseHandle(hSnapshot);
}

struct tm* Win32Utilities::LocaltimeEx(const time_t* t, struct tm* output)
{
#ifdef WIN32
    localtime_s(output, t);
#else
    localtime_r(t, output);
#endif
    return output;
}

std::string Win32Utilities::LocaltimeYYYMMDD_HHMMSS(std::time_t t)
{
    struct tm cur_tm{};
    LocaltimeEx(&t, &cur_tm);
    char buffer[128] = { 0 };

    snprintf(buffer, sizeof(buffer), "%04d%02d%02d-%02d%02d%02d"
        , cur_tm.tm_year + 1900, cur_tm.tm_mon+1, cur_tm.tm_mday
        , cur_tm.tm_hour, cur_tm.tm_min, cur_tm.tm_sec
        );
    std::string s = buffer;
    return s;
}

std::string Win32Utilities::GetExePath()
{
	char szFullPath[MAX_PATH];
	char szdrive[_MAX_DRIVE];
	char szdir[_MAX_DIR];
	::GetModuleFileNameA(NULL, szFullPath, MAX_PATH);
	_splitpath_s(szFullPath, szdrive, _MAX_DRIVE, szdir, _MAX_DIR, NULL, NULL, NULL, NULL);

	std::string szPath;
	szPath = StringFormat("%s%s", szdrive, szdir);
	//szPath = szPath.Left(szPath.GetLength() - 1);

	return szPath;
}

//格式化字符串string
std::string Win32Utilities::StringFormat(const char *fmt, ...)
{
	std::string strResult = "";
	if (NULL != fmt)
	{
		va_list marker = NULL;
		va_start(marker, fmt);                            //初始化变量参数 
		size_t nLength = _vscprintf(fmt, marker) + 1;    //获取格式化字符串长度
		std::vector<char> vBuffer(nLength, '\0');        //创建用于存储格式化字符串的字符数组
		int nWritten = _vsnprintf_s(&vBuffer[0], vBuffer.size(), nLength, fmt, marker);
		if (nWritten > 0)
		{
			strResult = &vBuffer[0];
		}
		va_end(marker);                                    //重置变量参数
	}
	return strResult;
}

