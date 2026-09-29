<#
.SYNOPSIS
    Builds the Blender extension repository of Magic Fit: MagicFit.zip and its index.

.DESCRIPTION
    Builds the package with tools/bundle-addon.ps1 (with its version and wheel
    checks), puts it in <RepositoryPath> as MagicFit.zip, and has Blender write
    index.json (and index.html) for it. Blender reads index.json when the
    repository is added under Preferences > Get Extensions > Repositories.

    The repository in blender_repo/ is committed, so the README's
    raw.githubusercontent.com link installs it; the release workflow builds its
    own copy for GitHub Pages. Check a repository with
    tools/verify-blender-repository.ps1.

.PARAMETER RepositoryPath
    Folder for the repository. Defaults to blender_repo\ in the repository root.

.PARAMETER BlenderPath
    Blender executable, passed on to bundle-addon.ps1.

.EXAMPLE
    .\tools\generate-blender-repository.ps1
#>
[CmdletBinding()]
param(
    [string] $RepositoryPath = '',
    [string] $BlenderPath = ''
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'extension-package.ps1')

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
    $RepositoryPath = Join-Path $RepositoryRoot 'blender_repo'
}
$RepositoryPath = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($RepositoryPath)
$Package = Get-ExtensionPackage
$PackagePath = Join-Path $RepositoryPath $Package.Archive

$BuildDirectory = Join-Path ([System.IO.Path]::GetTempPath()) ("magic-fit-build-" + [guid]::NewGuid().ToString('N'))
try {
    & (Join-Path $PSScriptRoot 'bundle-addon.ps1') -BlenderPath $BlenderPath -OutputDirectory $BuildDirectory
    $Built = Join-Path $BuildDirectory "$($Package.Id)-$($Package.Version).zip"

    # server-generate lists every zip in the folder, so only this one may be there.
    New-Item -ItemType Directory -Path $RepositoryPath -Force | Out-Null
    Get-ChildItem -LiteralPath $RepositoryPath -Filter '*.zip' -File | Remove-Item -Force
    Move-Item -LiteralPath $Built -Destination $PackagePath
}
finally {
    Remove-Item -LiteralPath $BuildDirectory -Recurse -Force -ErrorAction SilentlyContinue
}

# The same Blender bundle-addon.ps1 used: the given one, else 5.2 in Program Files, else PATH.
if ([string]::IsNullOrWhiteSpace($BlenderPath)) {
    $DefaultBlender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
    $BlenderPath = if (Test-Path -LiteralPath $DefaultBlender -PathType Leaf) { $DefaultBlender } else { (Get-Command blender).Source }
}
& $BlenderPath --background --factory-startup --command extension server-generate "--repo-dir=$RepositoryPath" --html
if ($LASTEXITCODE -ne 0) {
    throw "Blender could not generate the repository index (exit code $LASTEXITCODE)."
}

Write-Host "Generated the Blender extension repository in ${RepositoryPath}: $($Package.Archive) ($($Package.Name) $($Package.Version))"
