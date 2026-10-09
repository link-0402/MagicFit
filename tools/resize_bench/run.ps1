<#
.SYNOPSIS
    Runs the resize bench on every garment in parallel Blender processes, then summarizes it.

.DESCRIPTION
    Starts -Shards headless Blenders running bench.py on the bench data (see README.md), each on every Nth
    garment, waits for them and prints summarize.py's totals (against -Ref when given). Each shard's output
    goes to <Out>\shard<N>.log; run.json records the checkout and its commit. Blender runs with
    --factory-startup and a profile of its own in <Data>\profile, so yours isn't touched.

.PARAMETER Out
    Folder for the results. Defaults to <Data>\runs\<checkout folder name>-<time>.

.PARAMETER Checkout
    The Magic Fit checkout to measure (a worktree, say). Defaults to this repository.

.PARAMETER Data
    The bench data. Defaults to bench_data in this repository.

.PARAMETER Ref
    A previous run's folder to compare with.

.PARAMETER Targets
    near (default: Small and Omoi from each garment's From size) or all (every size).

.PARAMETER RoundTrip
    near (default), all or none: the sizes resized back to the From size.

.PARAMETER Only
    Garment tags to run.

.PARAMETER PerVertex
    Also write per-vertex arrays (Out\pv).

.EXAMPLE
    .\tools\resize_bench\run.ps1 -Out bench_data\runs\base

.EXAMPLE
    .\tools\resize_bench\run.ps1 -Checkout .claude\worktrees\my-change -Ref bench_data\runs\base
#>
[CmdletBinding()]
param(
    [string] $Out = '',
    [string] $Checkout = '',
    [string] $Data = '',
    [string] $Ref = '',
    [int] $Shards = 8,
    [ValidateSet('near', 'all')] [string] $Targets = 'near',
    [ValidateSet('near', 'all', 'none')] [string] $RoundTrip = 'near',
    [string[]] $Only = @(),
    [switch] $PerVertex,
    [string] $BlenderPath = ''
)

$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot
$root = Split-Path -Parent (Split-Path -Parent $here)
if (-not $Checkout) { $Checkout = $root }
if (-not $Data) { $Data = Join-Path $root 'bench_data' }
$Checkout = (Resolve-Path $Checkout).Path
$Data = (Resolve-Path $Data).Path
if (-not $Out) {
    $Out = Join-Path $Data ('runs\{0}-{1}' -f (Split-Path -Leaf $Checkout), (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
if (-not $BlenderPath) {
    $default = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
    $BlenderPath = if (Test-Path $default) { $default } else { (Get-Command blender).Source }
}
$python = Get-ChildItem (Split-Path -Parent $BlenderPath) -Recurse -Filter python.exe |
    Where-Object { $_.FullName -match '\\python\\bin\\' } | Select-Object -First 1

$commit = (& git -C $Checkout rev-parse --short HEAD) 2>$null
# New files not added yet count too: a module of the change can be among them.
$dirty = (& git -C $Checkout status --porcelain) 2>$null
@{ checkout = $Checkout; commit = $commit; uncommitted = [bool]$dirty; targets = $Targets; round_trip = $RoundTrip
   started = (Get-Date -Format o) } | ConvertTo-Json | ForEach-Object {
    [IO.File]::WriteAllText((Join-Path $Out 'run.json'), $_) }

$env:BLENDER_USER_RESOURCES = Join-Path $Data 'profile'
New-Item -ItemType Directory -Force $env:BLENDER_USER_RESOURCES | Out-Null
$quote = { param($s) '"' + $s + '"' }
$started = Get-Date
$processes = @()
for ($i = 0; $i -lt $Shards; $i++) {
    $arguments = @('-b', '--factory-startup', '--python', (& $quote (Join-Path $here 'bench.py')), '--',
                   (& $quote $Data), (& $quote $Out), '--checkout', (& $quote $Checkout), '--targets', $Targets,
                   '--round-trip', $RoundTrip, '--shard', "$i/$Shards")
    if ($Only.Count) { $arguments += @('--only', ($Only -join ',')) }
    if ($PerVertex) { $arguments += '--pv' }
    $process = Start-Process -FilePath $BlenderPath -ArgumentList $arguments -NoNewWindow -PassThru `
        -RedirectStandardOutput (Join-Path $Out "shard$i.log") -RedirectStandardError (Join-Path $Out "shard$i.err")
    $null = $process.Handle  # keeps ExitCode readable once it exits
    $processes += $process
}
$processes | Wait-Process
$failed = @($processes | Where-Object { $_.ExitCode -ne 0 }).Count
Write-Host ('{0} shards in {1:N0} s, {2} with failures (see shard*.log)' -f $Shards, ((Get-Date) - $started).TotalSeconds,
            $failed)

$summary = @((Join-Path $here 'summarize.py'), $Out)
if ($Ref) { $summary += @('--ref', (Resolve-Path $Ref).Path) }
& $python.FullName @summary
if ($failed) { exit 1 }
