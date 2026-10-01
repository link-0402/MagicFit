<#
.SYNOPSIS
    Runs Magic Fit's core and UI tests without taking over your desktop.

.DESCRIPTION
    Runs every tests/test_*.py: the core tests headless, the UI tests in a
    Blender window that doesn't take focus (--no-window-focus), optionally
    placed with -Position so it sits on another screen or in a corner while
    you work. Prints one line per test and a summary, and exits non-zero if
    any test failed. The last lines of a failing test's output are shown.

    If Blender crashes, the script closes the "Blender has stopped working"
    dialog itself (it kills the process once the crash log is written), marks
    that test failed and goes on.

    Tests load the add-on from the repository with --factory-startup, so your
    Blender profile isn't touched.

.PARAMETER BlenderPath
    Blender executable. Defaults to Blender 5.2 in Program Files, else blender
    on PATH.

.PARAMETER Position
    X, Y, width and height of the UI test window, like Blender's -p. Y is
    measured upward from the bottom of the whole desktop. Defaults to a small
    window; leave it out to let Blender choose.

.PARAMETER Only
    Test names (without test_) to run, for example fit_core, fit_ui.

.PARAMETER SkipUi
    Runs the core tests only.

.PARAMETER SkipFace
    Leaves out the face tests, for computers without the game.

.EXAMPLE
    .\tools\run-tests.ps1

.EXAMPLE
    .\tools\run-tests.ps1 -Position 1920,0,1280,720

.EXAMPLE
    .\tools\run-tests.ps1 -Only fit_core,fit_ui
#>
[CmdletBinding()]
param(
    [string] $BlenderPath = '',
    [int[]] $Position = @(),
    [string[]] $Only = @(),
    [switch] $SkipUi,
    [switch] $SkipFace
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$crashLog = Join-Path $env:TEMP 'blender.crash.txt'

if (-not $BlenderPath) {
    $default = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
    $BlenderPath = if (Test-Path $default) { $default } else { (Get-Command blender).Source }
}
if ($Position.Count -ne 0 -and $Position.Count -ne 4) { throw '-Position needs X, Y, width and height.' }

$core = 'core','fit_core','move_core','straighten_core','skirt_core','heels_core','smooth_core','hair_core',
        'face_core','relax_core','resize_core','lineup_core','transfer_core','cplus_core'
$ui = 'ui','fit_ui','move_ui','fit_deformed_ui','straighten_ui','skirt_ui','heels_ui','smooth_ui','hair_ui',
      'face_ui','relax_ui','resize_ui','lineup_ui','clipping_ui','sidebar_ui','transfer_ui','cplus_ui'

$tests = @($core | ForEach-Object { [pscustomobject]@{ Name = $_; Ui = $false } })
if (-not $SkipUi) { $tests += $ui | ForEach-Object { [pscustomobject]@{ Name = $_; Ui = $true } } }
if ($SkipFace) { $tests = $tests | Where-Object { $_.Name -notlike 'face_*' } }
if ($Only.Count) { $tests = $tests | Where-Object { $Only -contains $_.Name } }

Push-Location $root
try {
    $failed = @()
    foreach ($t in $tests) {
        $blenderArgs = @('--factory-startup')
        if ($t.Ui) {
            $blenderArgs += '--no-window-focus'
            if ($Position.Count) { $blenderArgs += '-p'; $blenderArgs += $Position }
            $blenderArgs += '--enable-event-simulate'
        } else {
            $blenderArgs = @('-b') + $blenderArgs
        }
        $blenderArgs += '--python', "tests/test_$($t.Name).py"

        # Blender writes its crash log, then waits behind a "Blender has stopped
        # working" dialog. Watch for the log and kill the process so nobody has
        # to click the dialog away.
        $started = Get-Date
        $outFile = Join-Path $env:TEMP 'magic_fit_test.out'
        $errFile = Join-Path $env:TEMP 'magic_fit_test.err'
        $process = Start-Process -FilePath $BlenderPath -ArgumentList $blenderArgs -NoNewWindow -PassThru `
            -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $null = $process.Handle  # keeps ExitCode readable after WaitForExit(timeout)
        $crashed = $false
        while (-not $process.WaitForExit(500)) {
            $log = Get-Item $crashLog -ErrorAction SilentlyContinue
            if ($log -and $log.LastWriteTime -ge $started) {
                Start-Sleep -Seconds 1
                $process.Kill()
                $process.WaitForExit()
                $crashed = $true
                break
            }
        }
        $exitCode = if ($crashed) { 'crashed' } else { $process.ExitCode }
        if ($exitCode -eq 0) {
            'ok      {0}' -f $t.Name
        } else {
            'FAILED  {0} ({1})' -f $t.Name, $(if ($crashed) { 'Blender crashed, see ' + $crashLog } else { "exit $exitCode" })
            Get-Content $outFile, $errFile -ErrorAction SilentlyContinue | Select-Object -Last 15 |
                ForEach-Object { '         ' + $_.TrimEnd() }
            $failed += $t.Name
        }
    }
    if ($failed) {
        "FAILED: $($failed -join ', ')"
        exit 1
    }
    "All $($tests.Count) tests passed."
} finally {
    Pop-Location
}
