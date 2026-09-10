; Inno Setup script for the HALS Post-Processing GUI.
;
; Compile from the repository root after building the PyInstaller one-directory
; app into dist\HALS_Post, e.g.:
;
;   ISCC /DMyAppVersion=2.3.0 packaging\hals_installer.iss
;
; The resulting installer is written to installer\HALS_Post_Setup_<version>.exe

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

#define MyAppName "HALS Post-Processing"
#define MyAppPublisher "HALS Project"
#define MyAppURL "https://github.com/dfapinov/lah-scanner"
#define MyAppExeName "HALS_Post.exe"

[Setup]
AppId={{7B1E4C2A-3D5F-4E8B-9C1A-0F2A6C4D8E10}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\HALS Post-Processing
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\installer
OutputBaseFilename=HALS_Post_Setup_{#MyAppVersion}
SetupIconFile=..\src\HALS_icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64
ArchitecturesAllowed=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The full PyInstaller one-directory output (executable + dependencies).
Source: "..\dist\HALS_Post\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
