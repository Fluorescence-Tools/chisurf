#include <windows.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

/*
   This wrapper ensures that the specified Python module runs without
   showing a terminal window on Windows.
*/

/* Run the given Python command, passing it the parameters we ourselves were given. */
static DWORD run_python_command(const char *command)
{
    STARTUPINFO si;
    PROCESS_INFORMATION pi;

    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    ZeroMemory(&pi, sizeof(pi));

    // Start the child process.
    if (!CreateProcess(NULL,   // No module name (use command line)
        (LPSTR)command,        // Command line
        NULL,           // Process handle not inheritable
        NULL,           // Thread handle not inheritable
        FALSE,          // Set handle inheritance to FALSE
        CREATE_NO_WINDOW, // No window
        NULL,           // Use parent's environment block
        NULL,           // Use parent's starting directory
        &si,            // Pointer to STARTUPINFO structure
        &pi)           // Pointer to PROCESS_INFORMATION structure
        )
    {
        fprintf(stderr, "CreateProcess failed (%d).\n", GetLastError());
        return 1;
    }

    // Wait until child process exits.
    WaitForSingleObject(pi.hProcess, INFINITE);

    // Get the exit code.
    DWORD exit_code;
    GetExitCodeProcess(pi.hProcess, &exit_code);

    // Close process and thread handles.
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);

    return exit_code;
}

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {
    const char *command = "python -m chisurf";
    return run_python_command(command);
}
