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
OutputBaseFilename=setup-{{ AppVersion }}
SetupIconFile={{ SetupIconFile }}
Compression=lzma2/ultra64
;Compression=lzma2/fast
;Compression=none
SolidCompression=yes
CompressionThreads=auto
UninstallLogMode=overwrite
DirExistsWarning=yes
UninstallDisplayIcon="{app}\{{ AppName }}"
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
;Source: "chisurf.exe"; DestDir: "{app}"
Source: "{{ SourceDir }}\dist\win\**"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs;
;uncomment below to add VC Runtimes
;Source: "{{ vc_runtime_path }}\*"; DestDir: {tmp}; Flags: deleteafterinstall
Source: "fix_shebangs.py"; DestDir: "{app}";

[Icons]
{% for entry_point in gui_entry_points %}Name: "{group}\{{ entry_point.lower() }}";Filename: {app}\Scripts\{{entry_point}}.exe; IconFilename: "{app}\icons\icon.ico";
{% endfor %}

[UninstallDelete]
Type: files; Name: "{app}\install.log"

[Run]
;uncomment below to add VC Runtimes
;{% for vc_runtime in vc_runtimes %}Filename: {tmp}\{{ vc_runtime }}; Parameters: /q
;{% endfor %}
