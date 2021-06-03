# Running Linux CS on Windows

ChiSurf (CS) is developed and tested on Linux. Windows and
macOS versions are only released seldomly. The Linux CS version
can be used on Windows using Windows Subystem for Linux (WSL). 
Below is briefly outlined how to use CS on WSL.

1. Install Windows Subystem for Linux (WSL)

    - Enter turn windows features on/off in Start menu
    - Select Windows Subystem for Linux (you may have to reboot)

2. Install Ubuntu

    - Open the "Windows Store"
    - Search for Ubuntu (best 18.04)
    - Install Ubuntu

3. Install an Windows X-Server / or use MobaXterm

    - <https://sourceforge.net/projects/vcxsrv/>
    - see: <https://medium.com/javarevisited/using-wsl-2-with-x-server-linux-on-windows-a372263533c3>

4. Start Ubuntu shell with X11 Server running in background

5. Download and install Miniconda

    - In the installation make sure that conda is initialized by default
    - wget <https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh>
    - chmod +x ./Miniconda3-latest-Linux-x86_64.sh
    - ./Miniconda3-latest-Linux-x86_64.sh

6. Open new Ubuntu shell

7. Install mamba to speed up installation (faster resolution of packages)

    - conda install mamba -c conda-forge

8. Make sure that .condarc file contains all needed channels
    - conda config --add channels conda-forge
    - conda config --add channels salilab
    - conda config --add channels tpeulen

9. Create, activate new conda python 3.7 environment for chisurf

    - conda create -n chisurf python=3.7.5
    - conda activate chisurf

10. Install chisurf

11. Mount windows shares in WSL

    - sudo mkdir /mnt/net
    - sudo mount -t drvfs '\\192.xxx.xxx.xxx\Share' /mnt/net ;
