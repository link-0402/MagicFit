# Shared by generate-blender-repository.ps1 and verify-blender-repository.ps1, so both name the
# package the same way. The id, name, version and minimum Blender come from
# magic_fit/blender_manifest.toml; the repository's zip keeps one name across versions.

function Get-ExtensionPackage {
    $ManifestPath = Join-Path (Join-Path (Split-Path -Parent $PSScriptRoot) 'magic_fit') 'blender_manifest.toml'
    # Only top-level keys; they come before the first [section].
    $TopLevel = ((Get-Content -LiteralPath $ManifestPath -Raw) -split '(?m)^\[', 2)[0]
    $Values = @{}
    foreach ($Key in 'id', 'name', 'version', 'blender_version_min') {
        $Match = [regex]::Match($TopLevel, ('(?m)^{0}\s*=\s*"([^"]+)"' -f $Key))
        if (-not $Match.Success) {
            throw "No $Key found in $ManifestPath"
        }
        $Values[$Key] = $Match.Groups[1].Value
    }

    [pscustomobject]@{
        Id              = $Values['id']
        Name            = $Values['name']
        Version         = $Values['version']
        BlenderMinimum  = $Values['blender_version_min']
        Archive         = 'MagicFit.zip'
    }
}
