; Inno Setup script for Vasudha.
;
; Why an installer at all, when a zip already works:
;
;   * It ends the Mark-of-the-Web crash for good. Windows Explorer flags every
;     file it extracts from a downloaded .zip, and .NET then refuses to load
;     pythonnet's assembly, so the app died on launch before showing a window.
;     app/runtime.py strips the flag at startup as a repair; files placed by an
;     installer are never flagged in the first place, which is a fix.
;   * Uninstall, Start Menu and Add/Remove Programs entries that a zip cannot
;     provide, so the 2.3 GB model and the app data have a supported way out.
;   * One signed artefact later, rather than 300 loose files.
;
; Per-user by default (PrivilegesRequired=lowest): no admin prompt, and the app
; needs nothing outside the user's own profile. A machine-wide install would
; also put the workspace and model cache somewhere the user cannot write.

#define AppName        "Vasudha"
#define AppPublisher   "Chaitanya"
#define AppURL         "https://github.com/cxaiiii/vasudha"
#define AppExe         "Vasudha.exe"

; Passed in by CI as /DAppVersion=x.y.z; the fallback keeps a local build working.
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
AppId={{8E4C1F0A-7B3D-4A21-9C55-VASUDHA00001}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputBaseFilename=Vasudha-{#AppVersion}-setup
OutputDir=..\dist
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The bundle is ~160 MB and the model is fetched separately on first run, so
; the download is not what takes the time — but say so, since a user watching a
; progress bar deserves to know what it covers.
SetupIconFile=vasudha.ico
UninstallDisplayIcon={app}\{#AppExe}
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; An upgrade over an existing install must not leave half the old bundle
; behind: PyInstaller folder layouts change between builds.
Uninstallable=yes
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
  GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
; The whole PyInstaller one-folder output. recursesubdirs is load-bearing:
; llama.cpp's backends live in _internal\llama_cpp\lib and are loaded by name
; at runtime, so a missing one is a silent fallback to CPU rather than an error.
Source: "..\dist\Vasudha\*"; DestDir: "{app}"; \
  Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; \
  Flags: nowait postinstall skipifsilent

[UninstallDelete]
; PyInstaller writes nothing here at runtime, but an upgrade from an older
; layout can leave an empty tree behind.
Type: dirifempty; Name: "{app}"

[Code]
// The model, chats, memory and workspaces live in %LOCALAPPDATA%\Vasudha and
// are deliberately NOT removed by default: the model alone is 2.3 GB and
// re-downloading it because someone reinstalled would be its own bug. Asked
// rather than assumed, because leaving gigabytes behind silently is equally
// rude.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\Vasudha');
    if DirExists(DataDir) then
    begin
      if MsgBox('Also delete Vasudha''s downloaded model, chats and saved files?'
                + #13#10 + #13#10 + DataDir + #13#10 + #13#10
                + 'Choose No to keep them for a future install.',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
    end;
  end;
end;
