# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = "C:\Program Files (x86)\JetBrains\CLion 2016.3.5\bin\cmake\bin\cmake.exe"

# The command to remove a file.
RM = "C:\Program Files (x86)\JetBrains\CLion 2016.3.5\bin\cmake\bin\cmake.exe" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/windows.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/windows.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/windows.dir/flags.make

CMakeFiles/windows.dir/main.cpp.obj: CMakeFiles/windows.dir/flags.make
CMakeFiles/windows.dir/main.cpp.obj: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/windows.dir/main.cpp.obj"
	C:\mingw-w64\mingw64\bin\g++.exe   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\windows.dir\main.cpp.obj -c C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\main.cpp

CMakeFiles/windows.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/windows.dir/main.cpp.i"
	C:\mingw-w64\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\main.cpp > CMakeFiles\windows.dir\main.cpp.i

CMakeFiles/windows.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/windows.dir/main.cpp.s"
	C:\mingw-w64\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\main.cpp -o CMakeFiles\windows.dir\main.cpp.s

CMakeFiles/windows.dir/main.cpp.obj.requires:

.PHONY : CMakeFiles/windows.dir/main.cpp.obj.requires

CMakeFiles/windows.dir/main.cpp.obj.provides: CMakeFiles/windows.dir/main.cpp.obj.requires
	$(MAKE) -f CMakeFiles\windows.dir\build.make CMakeFiles/windows.dir/main.cpp.obj.provides.build
.PHONY : CMakeFiles/windows.dir/main.cpp.obj.provides

CMakeFiles/windows.dir/main.cpp.obj.provides.build: CMakeFiles/windows.dir/main.cpp.obj


# Object files for target windows
windows_OBJECTS = \
"CMakeFiles/windows.dir/main.cpp.obj"

# External object files for target windows
windows_EXTERNAL_OBJECTS =

windows.exe: CMakeFiles/windows.dir/main.cpp.obj
windows.exe: CMakeFiles/windows.dir/build.make
windows.exe: CMakeFiles/windows.dir/linklibs.rsp
windows.exe: CMakeFiles/windows.dir/objects1.rsp
windows.exe: CMakeFiles/windows.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable windows.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\windows.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/windows.dir/build: windows.exe

.PHONY : CMakeFiles/windows.dir/build

CMakeFiles/windows.dir/requires: CMakeFiles/windows.dir/main.cpp.obj.requires

.PHONY : CMakeFiles/windows.dir/requires

CMakeFiles/windows.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\windows.dir\cmake_clean.cmake
.PHONY : CMakeFiles/windows.dir/clean

CMakeFiles/windows.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\cmake-build-debug C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\cmake-build-debug C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\windows\cmake-build-debug\CMakeFiles\windows.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/windows.dir/depend
