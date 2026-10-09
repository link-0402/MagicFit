<#
.SYNOPSIS
    Runs Magic Fit's core and UI tests without taking over your desktop.

.DESCRIPTION
    Runs every tests/test_*.py: the core tests headless, the UI tests in a
    Blender window that doesn't take focus (--no-window-focus), optionally
    on your other screen (-OtherScreen) while you work. Prints one line per
    test and a summary, and exits non-zero if any test failed. The last lines
    of a failing test's output are shown.

    The brush tests measure their brush in screen pixels against a model
    framed to fit the window, so they expect a full-screen window like
    Blender's own: in a smaller one the brush reaches further over the model
    and checks like "only under the brush" fail.

    If Blender crashes, the script closes the "Blender has stopped working"
    dialog itself (it kills the process once a crash log is written), marks
    that test failed and goes on. So it does with a test still running after
    -TimeoutMinutes, and a test script that raises an error counts as failed.

    Tests load the add-on from the repository with --factory-startup, so your
    Blender profile isn't touched.

.PARAMETER BlenderPath
    Blender executable. Defaults to Blender 5.2 in Program Files, else blender
    on PATH.

.PARAMETER OtherScreen
    Opens the UI test window full size on the first screen that isn't the
    primary one, out of your way.

.PARAMETER Position
    X, Y, width and height of the UI test window, like Blender's -p: the
    window's top lands at (height of the whole desktop - Y - height). Left
    out, Blender fills the primary screen. Brush tests fail in small windows
    (see above).

.PARAMETER Only
    Test names (without test_) to run, for example fit_core, fit_ui.

.PARAMETER SkipUi
    Runs the core tests only.

.PARAMETER SkipFace
    Leaves out the face tests, for computers without the game.

.PARAMETER TimeoutMinutes
    How long a test may run before it's stopped and counted as failed.

.EXAMPLE
    .\tools\run-tests.ps1

.EXAMPLE
    .\tools\run-tests.ps1 -OtherScreen

.EXAMPLE
    .\tools\run-tests.ps1 -Only fit_core,fit_ui
#>
[CmdletBinding()]
param(
    [string] $BlenderPath = '',
    [switch] $OtherScreen,
    [int[]] $Position = @(),
    [string[]] $Only = @(),
    [switch] $SkipUi,
    [switch] $SkipFace,
    [int] $TimeoutMinutes = 30
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
# Blender names its crash log after the open file: blender.crash.txt without one.
$crashLogs = Join-Path $env:TEMP '*.crash.txt'
# Each run's output files of its own, so runs at the same time don't mix them up.
$outFile = Join-Path $env:TEMP "magic_fit_test_$PID.out"
$errFile = Join-Path $env:TEMP "magic_fit_test_$PID.err"

if (-not $BlenderPath) {
    $default = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
    $BlenderPath = if (Test-Path $default) { $default } else { (Get-Command blender).Source }
}
if ($Position.Count -ne 0 -and $Position.Count -ne 4) { throw '-Position needs X, Y, width and height.' }
if ($OtherScreen) {
    Add-Type -AssemblyName System.Windows.Forms
    $screen = [System.Windows.Forms.Screen]::AllScreens | Where-Object { -not $_.Primary } | Select-Object -First 1
    if (-not $screen) { throw 'There is no screen besides the primary one.' }
    $area = $screen.WorkingArea
    $desktop = [System.Windows.Forms.SystemInformation]::VirtualScreen
    # Blender puts a window's top (below its title bar) at desktop height - Y - height, in screen coordinates.
    # A window as wide as the screen is maximized there, the size Blender opens at on the primary screen.
    $titleBar = [System.Windows.Forms.SystemInformation]::CaptionHeight
    $height = $area.Height - $titleBar
    $Position = @($area.X, ($desktop.Height - $height - ($area.Y + $titleBar)), $area.Width, $height)
}

# Every tests/test_*.py: the core tests first, then the UI ones (test_ui.py and test_*_ui.py), which need a window.
$names = Get-ChildItem (Join-Path $root 'tests') -Filter 'test_*.py' | Sort-Object Name |
    ForEach-Object { $_.BaseName.Substring(5) }
$tests = @($names | Where-Object { $_ -ne 'ui' -and $_ -notlike '*_ui' } |
    ForEach-Object { [pscustomobject]@{ Name = $_; Ui = $false } })
if (-not $SkipUi) {
    $tests += $names | Where-Object { $_ -eq 'ui' -or $_ -like '*_ui' } |
        ForEach-Object { [pscustomobject]@{ Name = $_; Ui = $true } }
}
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
        # An error the test script doesn't catch fails the test (Blender would exit with 0).
        $blenderArgs += '--python-exit-code', '1', '--python', "tests/test_$($t.Name).py"

        # Blender writes its crash log, then waits behind a "Blender has stopped
        # working" dialog. Watch for the log and kill the process so nobody has
        # to click the dialog away; so too for a test that runs too long.
        $started = Get-Date
        $process = Start-Process -FilePath $BlenderPath -ArgumentList $blenderArgs -NoNewWindow -PassThru `
            -RedirectStandardOutput $outFile -RedirectStandardError $errFile
        $null = $process.Handle  # keeps ExitCode readable after WaitForExit(timeout)
        $crashed = $null
        $timedOut = $false
        while (-not $process.WaitForExit(500)) {
            $crashed = Get-ChildItem $crashLogs -ErrorAction SilentlyContinue |
                Where-Object { $_.LastWriteTime -ge $started } | Select-Object -First 1
            $timedOut = ((Get-Date) - $started).TotalMinutes -ge $TimeoutMinutes
            if ($crashed -or $timedOut) {
                if ($crashed) { Start-Sleep -Seconds 1 }
                $process.Kill()
                $process.WaitForExit()
                break
            }
        }
        $exitCode = if ($crashed -or $timedOut) { 'stopped' } else { $process.ExitCode }
        if ($exitCode -eq 0) {
            'ok      {0}' -f $t.Name
        } else {
            $why = if ($crashed) { 'Blender crashed, see ' + $crashed.FullName }
                   elseif ($timedOut) { "still running after $TimeoutMinutes minutes" } else { "exit $exitCode" }
            'FAILED  {0} ({1})' -f $t.Name, $why
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
    Remove-Item $outFile, $errFile -ErrorAction SilentlyContinue
}
