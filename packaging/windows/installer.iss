; Inno Setup script for Exam Paper Analyzer.
; Compile with: ISCC.exe packaging\windows\installer.iss
; (run scripts\build_windows.ps1 to do the full PyInstaller + Inno Setup
; build in one step). Expects PyInstaller's onedir output to already exist
; at dist\ExamPaperAnalyzer\ (see packaging\exampapersorter.spec).

#define MyAppName "Exam Paper Analyzer"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Exam Paper Analyzer"
#define MyAppExeName "ExamPaperAnalyzer.exe"
#define RepoRoot ".."  ; this .iss lives in packaging\windows

[Setup]
AppId={{B9D6F0B0-6B1E-4C6E-9B7A-3E9E7F9C7E10}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
; Per-machine by default (Program Files); user data always lives in the
; per-user AppData directory (see exampapersorter/paths.py), never here --
; so uninstalling never touches analysis output, the database, or the
; embedding-model cache.
OutputDir=..\..\dist\installer
OutputBaseFilename=ExamPaperAnalyzer-Windows-Setup
SetupIconFile=..\icon\icon.ico
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\..\dist\ExamPaperAnalyzer\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

; Uninstall deliberately does NOT remove %LOCALAPPDATA%\ExamPaperAnalyzer
; (database, reports, cached embedding model) -- only the installed
; program files. A user who reinstalls keeps their prior analysis history;
; someone who genuinely wants a clean slate can delete that folder by hand.
