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
CMAKE_SOURCE_DIR = C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/fpsdll.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/fpsdll.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/fpsdll.dir/flags.make

CMakeFiles/fpsdll.dir/av_routines.cpp.obj: CMakeFiles/fpsdll.dir/flags.make
CMakeFiles/fpsdll.dir/av_routines.cpp.obj: ../av_routines.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/fpsdll.dir/av_routines.cpp.obj"
	C:\mingw-w64\mingw64\bin\g++.exe   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\fpsdll.dir\av_routines.cpp.obj -c C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\av_routines.cpp

CMakeFiles/fpsdll.dir/av_routines.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fpsdll.dir/av_routines.cpp.i"
	C:\mingw-w64\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\av_routines.cpp > CMakeFiles\fpsdll.dir\av_routines.cpp.i

CMakeFiles/fpsdll.dir/av_routines.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fpsdll.dir/av_routines.cpp.s"
	C:\mingw-w64\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\av_routines.cpp -o CMakeFiles\fpsdll.dir\av_routines.cpp.s

CMakeFiles/fpsdll.dir/av_routines.cpp.obj.requires:

.PHONY : CMakeFiles/fpsdll.dir/av_routines.cpp.obj.requires

CMakeFiles/fpsdll.dir/av_routines.cpp.obj.provides: CMakeFiles/fpsdll.dir/av_routines.cpp.obj.requires
	$(MAKE) -f CMakeFiles\fpsdll.dir\build.make CMakeFiles/fpsdll.dir/av_routines.cpp.obj.provides.build
.PHONY : CMakeFiles/fpsdll.dir/av_routines.cpp.obj.provides

CMakeFiles/fpsdll.dir/av_routines.cpp.obj.provides.build: CMakeFiles/fpsdll.dir/av_routines.cpp.obj


CMakeFiles/fpsdll.dir/av_testbench.cpp.obj: CMakeFiles/fpsdll.dir/flags.make
CMakeFiles/fpsdll.dir/av_testbench.cpp.obj: ../av_testbench.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/fpsdll.dir/av_testbench.cpp.obj"
	C:\mingw-w64\mingw64\bin\g++.exe   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\fpsdll.dir\av_testbench.cpp.obj -c C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\av_testbench.cpp

CMakeFiles/fpsdll.dir/av_testbench.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/fpsdll.dir/av_testbench.cpp.i"
	C:\mingw-w64\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\av_testbench.cpp > CMakeFiles\fpsdll.dir\av_testbench.cpp.i

CMakeFiles/fpsdll.dir/av_testbench.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/fpsdll.dir/av_testbench.cpp.s"
	C:\mingw-w64\mingw64\bin\g++.exe  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\av_testbench.cpp -o CMakeFiles\fpsdll.dir\av_testbench.cpp.s

CMakeFiles/fpsdll.dir/av_testbench.cpp.obj.requires:

.PHONY : CMakeFiles/fpsdll.dir/av_testbench.cpp.obj.requires

CMakeFiles/fpsdll.dir/av_testbench.cpp.obj.provides: CMakeFiles/fpsdll.dir/av_testbench.cpp.obj.requires
	$(MAKE) -f CMakeFiles\fpsdll.dir\build.make CMakeFiles/fpsdll.dir/av_testbench.cpp.obj.provides.build
.PHONY : CMakeFiles/fpsdll.dir/av_testbench.cpp.obj.provides

CMakeFiles/fpsdll.dir/av_testbench.cpp.obj.provides.build: CMakeFiles/fpsdll.dir/av_testbench.cpp.obj


# Object files for target fpsdll
fpsdll_OBJECTS = \
"CMakeFiles/fpsdll.dir/av_routines.cpp.obj" \
"CMakeFiles/fpsdll.dir/av_testbench.cpp.obj"

# External object files for target fpsdll
fpsdll_EXTERNAL_OBJECTS =

libfpsdll.a: CMakeFiles/fpsdll.dir/av_routines.cpp.obj
libfpsdll.a: CMakeFiles/fpsdll.dir/av_testbench.cpp.obj
libfpsdll.a: CMakeFiles/fpsdll.dir/build.make
libfpsdll.a: CMakeFiles/fpsdll.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX static library libfpsdll.a"
	$(CMAKE_COMMAND) -P CMakeFiles\fpsdll.dir\cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\fpsdll.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/fpsdll.dir/build: libfpsdll.a

.PHONY : CMakeFiles/fpsdll.dir/build

CMakeFiles/fpsdll.dir/requires: CMakeFiles/fpsdll.dir/av_routines.cpp.obj.requires
CMakeFiles/fpsdll.dir/requires: CMakeFiles/fpsdll.dir/av_testbench.cpp.obj.requires

.PHONY : CMakeFiles/fpsdll.dir/requires

CMakeFiles/fpsdll.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\fpsdll.dir\cmake_clean.cmake
.PHONY : CMakeFiles/fpsdll.dir/clean

CMakeFiles/fpsdll.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2 C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2 C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\cmake-build-debug C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\cmake-build-debug C:\Users\peulen\OneDrive\Programming\ChiSurf\mfm\fluorescence\fps\dll\v1.2\cmake-build-debug\CMakeFiles\fpsdll.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/fpsdll.dir/depend
