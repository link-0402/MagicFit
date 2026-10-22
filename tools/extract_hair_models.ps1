<#
.SYNOPSIS
Copies every race's vanilla hair models out of the game's data, for tools/make_scalp_map.py.

.DESCRIPTION
Reads chara/human/<race>/obj/hair/h<NNNN>/model/<race>h<NNNN>_hir.mdl (hair 1 to 999) of the 18 races of
tools/make_hair_reference.py with Lumina (Dalamud's dev copy) and writes the ones that exist to
<OutputDirectory>\<race>\<race>h<NNNN>_hir.mdl. The game folder is only read. Keep the output outside the
repository: game files are never committed.

.EXAMPLE
pwsh tools/extract_hair_models.ps1 -OutputDirectory "$env:TEMP\magic_fit_hair_models"
#>
param(
    [string]$Sqpack = "E:\SquareEnix\FINAL FANTASY XIV - A Realm Reborn\game\sqpack",
    [Parameter(Mandatory = $true)][string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$Races = @("c0101", "c0201", "c0301", "c0401", "c0501", "c0601", "c0701", "c0801", "c0901", "c1001",
           "c1101", "c1201", "c1301", "c1401", "c1501", "c1601", "c1701", "c1801")

Add-Type -Path "$env:APPDATA\XIVLauncher\addon\Hooks\dev\Lumina.dll"
$game = [Lumina.GameData]::new($Sqpack)

foreach ($race in $Races) {
    $folder = Join-Path $OutputDirectory $race
    New-Item -ItemType Directory -Force -Path $folder | Out-Null
    $count = 0
    foreach ($id in 1..999) {
        $name = "{0}h{1:D4}_hir.mdl" -f $race, $id
        $path = "chara/human/{0}/obj/hair/h{1:D4}/model/{2}" -f $race, $id, $name
        $file = $null
        try { $file = $game.GetFile($path) } catch { $file = $null }
        if ($null -eq $file) { continue }
        [System.IO.File]::WriteAllBytes((Join-Path $folder $name), $file.Data)
        $count++
    }
    Write-Output ("{0} {1} hair models" -f $race, $count)
}
