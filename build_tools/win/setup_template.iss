[Setup]
AppId={{ AppId }}
AppName={{ AppName }}
AppVersion={{ AppVersion }}
AppPublisher={{ AppPublisher }}
AppPublisherURL={{ AppURL }}
AppSupportURL={{ AppURL }}
AppUpdatesURL={{ AppURL }}
DefaultDirName={pf}\{{ AppName }}
UsePreviousAppDir=no
DefaultGroupName={{ AppName }}
LicenseFile={{ LicenseFile }}
OutputDir={{ Output_dir }}
OutputBaseFilename=setup_{{ AppVersion }}
SetupIconFile=icons\icon.ico
Compression=lzma2/ultra64
;Compression=none
SolidCompression=yes
CompressionThreads=4
UninstallLogMode=overwrite
DirExistsWarning=yes
UninstallDisplayIcon={app}\{{ AppName }}
DisableProgramGroupPage=no
;DiskSliceSize=1073741824
;DiskSpanning=true

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Icon files must be explicitely included
Source: ".\icons\*.ico"; DestDir: "{app}\icons"
Source: "chisurf.cmd"; DestDir: "{app}"
Source: "{{ SourceDir }}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{{ vc_runtime_path }}\*"; DestDir: {tmp}; Flags: deleteafterinstall
Source: "fix_shebangs.py"; DestDir: "{app}";

[Icons]
Name: "{group}\{cm:UninstallProgram,{{ AppName }}}"; Filename: "{uninstallexe}"
{% for entry_point in gui_entry_points %}Name: "{group}\{{ entry_point.lower() }}";Filename: "{app}\chisurf.cmd"; Parameters: {{ entry_point }}.exe;IconFilename: "{app}\icons\icon.ico";
{% endfor %}

[UninstallDelete]
Type: files; Name: {app}\install.log

[Run]
{% for vc_runtime in vc_runtimes %}Filename: {tmp}\{{ vc_runtime }}; Parameters: /q
{% endfor %}
; We do not want the user to have the option of avoiding this script, so no 'postinstall; flag
{% for entry_point in gui_entry_points %} Filename:{app}\python.exe; WorkingDir:{app}; Parameters: "fix_shebangs.py {{ entry_point }}"; Flags: runascurrentuser runmaximized
{% endfor %}
