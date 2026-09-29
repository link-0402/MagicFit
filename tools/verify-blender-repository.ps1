<#
.SYNOPSIS
    Checks a Blender extension repository made by generate-blender-repository.ps1.

.DESCRIPTION
    Checks that index.json lists only Magic Fit, with the id, version and minimum
    Blender of magic_fit/blender_manifest.toml, and that it describes the zip next
    to it (size and hash). Blender refuses to install a package whose size or hash
    differs from its index entry.

.PARAMETER RepositoryPath
    Folder of the repository. Defaults to blender_repo\ in the repository root.
#>
[CmdletBinding()]
param(
    [string] $RepositoryPath = ''
)

$ErrorActionPreference = 'Stop'

. (Join-Path $PSScriptRoot 'extension-package.ps1')

if ([string]::IsNullOrWhiteSpace($RepositoryPath)) {
    $RepositoryPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'blender_repo'
}
$Package = Get-ExtensionPackage
$RepositoryPath = (Resolve-Path -LiteralPath $RepositoryPath).Path
$ArchivePath = Join-Path $RepositoryPath $Package.Archive
$IndexPath = Join-Path $RepositoryPath 'index.json'
foreach ($Path in $ArchivePath, $IndexPath) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Missing $Path. Run tools/generate-blender-repository.ps1."
    }
}

# A second entry means a zip from an earlier build is still there.
$Entries = @((Get-Content -LiteralPath $IndexPath -Raw | ConvertFrom-Json).data)
if ($Entries.Count -ne 1 -or $Entries[0].id -ne $Package.Id) {
    $Listed = ($Entries | ForEach-Object { "$($_.id) ($($_.archive_url))" }) -join ', '
    throw "The index should list only $($Package.Id), found: $Listed. Run tools/generate-blender-repository.ps1."
}
$Entry = $Entries[0]
$Problems = @()
if ($Entry.archive_url -ne "./$($Package.Archive)") {
    $Problems += "archive: index has $($Entry.archive_url), expected ./$($Package.Archive)"
}
if ($Entry.version -ne $Package.Version) {
    $Problems += "version: index has $($Entry.version), the manifest $($Package.Version)"
}
if ($Entry.blender_version_min -ne $Package.BlenderMinimum) {
    $Problems += "minimum Blender: index has $($Entry.blender_version_min), the manifest $($Package.BlenderMinimum)"
}
$Archive = Get-Item -LiteralPath $ArchivePath
if ([int64] $Entry.archive_size -ne $Archive.Length) {
    $Problems += "size: index has $($Entry.archive_size), the zip is $($Archive.Length)"
}
$Hash = 'sha256:' + (Get-FileHash -LiteralPath $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
if (([string] $Entry.archive_hash).ToLowerInvariant() -ne $Hash) {
    $Problems += "hash: index has $($Entry.archive_hash), the zip $Hash"
}
if ($Problems) {
    throw ("The repository in $RepositoryPath doesn't match (run tools/generate-blender-repository.ps1):`n  " + ($Problems -join "`n  "))
}

Write-Host "Verified the Blender repository: $($Package.Archive), $($Package.Name) $($Entry.version), $($Archive.Length) bytes, $Hash"
