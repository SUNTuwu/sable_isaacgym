$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskBlender = 'D:\program_D\Blender\blender.exe'
$taskBlend = Join-Path $taskRoot 'blender\sable_rig.blend'
$taskExportScript = Join-Path $PSScriptRoot 'export_description.py'
if (-not (Test-Path -LiteralPath $taskBlend)) { throw 'Missing blender/sable_rig.blend' }
& $taskBlender --background --factory-startup --disable-autoexec --threads 2 $taskBlend --python-exit-code 1 --python $taskExportScript
if ($LASTEXITCODE -ne 0) { throw "Blender export failed with exit code $LASTEXITCODE" }
