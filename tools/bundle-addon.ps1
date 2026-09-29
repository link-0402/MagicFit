<#
.SYNOPSIS
    Builds the installable extension zip of Magic Fit with Blender.

.DESCRIPTION
    Runs Blender's own packager on magic_fit/ and writes
    <OutputDirectory>\magic_fit-<version>.zip. Blender leaves out
    __pycache__ folders, dot-files and zips, and always includes the manifest
    and the wheels it lists.

    Before building, it checks that blender_manifest.toml and bl_info in
    __init__.py have the same version, and that the wheels the manifest lists
    are there and stripped (see tools/vendor_wheels.py). After building, it
    checks the zip for the manifest, the wheels and stray bytecode.

    Building only reads your Blender profile: nothing is installed or enabled.
    -Check installs the zip into a throwaway profile (tools/check_install.py).

.PARAMETER BlenderPath
    Blender executable. Defaults to Blender 5.2 in Program Files, else blender
    on PATH.

.PARAMETER OutputDirectory
    Folder for the zip. Defaults to dist\ in the repository root.

.PARAMETER Check
    Runs tools/check_install.py on the zip afterwards: installs it into a
    throwaway profile and checks Weight Transfer, Customize+, Hair and Face.
    Face needs the game installed where Magic Fit finds it.

.EXAMPLE
    .\tools\bundle-addon.ps1

.EXAMPLE
    .\tools\bundle-addon.ps1 -Check
#>
[CmdletBinding()]
param(
    [string] $BlenderPath = '',
    [string] $OutputDirectory = '',
    [switch] $Check
)

$ErrorActionPreference = 'Stop'

$DefaultBlender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Definition }
$RepositoryRoot = (Resolve-Path -LiteralPath (Join-Path $ScriptRoot '..')).Path
$SourceDirectory = Join-Path $RepositoryRoot 'magic_fit'
$ManifestPath = Join-Path $SourceDirectory 'blender_manifest.toml'
$InitPath = Join-Path $SourceDirectory '__init__.py'

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Get-TomlString([string] $Source, [string] $Key) {
    $Match = [regex]::Match($Source, '(?m)^\s*' + [regex]::Escape($Key) + '\s*=\s*"(?<value>[^"]*)"')
    if (-not $Match.Success) {
        throw "Could not read '$Key' from $ManifestPath"
    }
    return $Match.Groups['value'].Value
}

function Get-TomlStringList([string] $Source, [string] $Key) {
    $Match = [regex]::Match($Source, '(?ms)^\s*' + [regex]::Escape($Key) + '\s*=\s*\[(?<items>.*?)\]')
    if (-not $Match.Success) {
        return @()
    }
    return @([regex]::Matches($Match.Groups['items'].Value, '"(?<value>[^"]+)"') | ForEach-Object { $_.Groups['value'].Value })
}

function Format-Size([double] $Bytes) {
    return '{0:N1} MB' -f ($Bytes / 1MB)
}

function Invoke-Native([string] $Executable, [string[]] $Arguments) {
    # Blender and Python write to stderr. Collect it as text instead of letting
    # Windows PowerShell turn every line into a terminating error.
    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $Lines = @(& $Executable @Arguments 2>&1 | ForEach-Object { "$_" })
        return [pscustomobject]@{ ExitCode = $LASTEXITCODE; Output = $Lines }
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }
}

# Versions: the manifest names the zip, bl_info is what older Blenders and the Preferences show.
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Could not find the extension manifest: $ManifestPath"
}
$Manifest = Get-Content -LiteralPath $ManifestPath -Raw
$ExtensionId = Get-TomlString $Manifest 'id'
$Version = Get-TomlString $Manifest 'version'
$BlInfoVersion = [regex]::Match((Get-Content -LiteralPath $InitPath -Raw), '"version"\s*:\s*\((?<version>[^)]*)\)')
if (-not $BlInfoVersion.Success) {
    throw "Could not read bl_info['version'] from $InitPath"
}
$BlInfoVersion = ($BlInfoVersion.Groups['version'].Value -split ',' | ForEach-Object { $_.Trim() } |
    Where-Object { $_ }) -join '.'
if ($BlInfoVersion -ne $Version) {
    throw "blender_manifest.toml has version $Version but bl_info in __init__.py has $BlInfoVersion. Update both before building."
}

# Wheels: present, and stripped by tools/vendor_wheels.py (whose test suites would otherwise ship).
$WheelNames = @()
foreach ($WheelPath in Get-TomlStringList $Manifest 'wheels') {
    $Relative = $WheelPath -replace '^\./', ''
    $Source = Join-Path $SourceDirectory ($Relative.Replace('/', [System.IO.Path]::DirectorySeparatorChar))
    if (-not (Test-Path -LiteralPath $Source -PathType Leaf)) {
        throw "blender_manifest.toml lists a wheel that is missing: $Source`nRun: python tools/vendor_wheels.py"
    }
    $Archive = [System.IO.Compression.ZipFile]::OpenRead($Source)
    try {
        $Unstripped = @($Archive.Entries | Where-Object { $_.FullName -match '(^|/)(tests|__pycache__)/|\.(lib|pdb|pyi)$' }).Count
    }
    finally {
        $Archive.Dispose()
    }
    if ($Unstripped) {
        Write-Warning "$(Split-Path -Leaf $Source) still has $Unstripped files Python never loads. Run: python tools/vendor_wheels.py"
    }
    $WheelNames += $Relative
}

# Blender: the given one, else 5.2 (what the tests use), else blender on PATH.
if ([string]::IsNullOrWhiteSpace($BlenderPath)) {
    if (Test-Path -LiteralPath $DefaultBlender -PathType Leaf) {
        $BlenderPath = $DefaultBlender
    } else {
        $Command = Get-Command blender -ErrorAction SilentlyContinue
        if ($null -eq $Command) {
            throw "Could not find Blender at $DefaultBlender or on PATH. Pass -BlenderPath."
        }
        $BlenderPath = $Command.Source
    }
} elseif (-not (Test-Path -LiteralPath $BlenderPath -PathType Leaf)) {
    throw "Blender not found: $BlenderPath"
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path $RepositoryRoot 'dist'
}
# Relative to the PowerShell location, which can differ from the process's.
$OutputDirectory = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputDirectory)
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$ZipPath = Join-Path $OutputDirectory "$ExtensionId-$Version.zip"

# Blender writes a temporary file and renames it, so a failed build keeps the previous zip.
Write-Host "Building $ExtensionId $Version with $BlenderPath"
$Result = Invoke-Native $BlenderPath @(
    '--command', 'extension', 'build', '--source-dir', $SourceDirectory, '--output-dir', $OutputDirectory)
if ($Result.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $ZipPath -PathType Leaf)) {
    $Result.Output | Write-Host
    throw "Blender could not build the package (exit code $($Result.ExitCode))."
}

$Archive = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)
try {
    $Names = @($Archive.Entries | ForEach-Object { $_.FullName })
}
finally {
    $Archive.Dispose()
}
$Problems = @()
foreach ($Name in @('blender_manifest.toml', '__init__.py') + $WheelNames) {
    if ($Names -notcontains $Name) {
        $Problems += "missing $Name"
    }
}
$Bytecode = @($Names | Where-Object { $_ -match '(^|/)__pycache__/|\.py[co]$' })
if ($Bytecode) {
    $Problems += "bytecode included: $($Bytecode[0])" + $(if ($Bytecode.Count -gt 1) { " and $($Bytecode.Count - 1) more" } else { '' })
}
if ($Problems) {
    throw ("The package $ZipPath has problems:`n  " + ($Problems -join "`n  "))
}
$ZipInfo = Get-Item -LiteralPath $ZipPath
Write-Host ("Built {0} ({1}, {2} files, {3} wheels)" -f $ZipInfo.FullName, (Format-Size $ZipInfo.Length), $Names.Count, $WheelNames.Count)

if (-not $Check) {
    return
}

# Blender's own Python runs check_install.py; any Python 3 would do.
$Python = Get-ChildItem -Path (Join-Path (Split-Path -Parent $BlenderPath) '*\python\bin\python.exe') -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $Python) {
    $Command = Get-Command python -ErrorAction SilentlyContinue
    if ($null -eq $Command) {
        throw "-Check needs Python: none next to Blender or on PATH."
    }
    $Python = $Command.Source
}
Write-Host 'Installing into a throwaway profile to check it...'
$Result = Invoke-Native $Python @((Join-Path $ScriptRoot 'check_install.py'), $ZipPath, $BlenderPath)
$Result.Output | Write-Host
if ($Result.ExitCode -ne 0) {
    throw "check_install.py failed (exit code $($Result.ExitCode))."
}
