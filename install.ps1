# Capture Front/Back And Bake - Marmoset Toolbag plugin installer (Windows)
#
# One-liner install (no clone needed):
#   irm https://raw.githubusercontent.com/ji-eee/Marmoset-script/main/install.ps1 | iex
#
# Or from a cloned repo:
#   double-click install.bat
#   powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1 [-PluginDir <path>]
#
# What it does:
#   1. Uses the repo files next to this script if present; otherwise clones the
#      repo (installing git via winget when missing) or falls back to a ZIP
#      download that needs no git at all.
#   2. Finds the Marmoset Toolbag user plugin folder automatically
#      (%LOCALAPPDATA%\Marmoset Toolbag 5\plugins etc.); asks for a path if it
#      cannot be found.
#   3. Copies CaptureFrontBackBake.py + projbake/ into
#      <plugins>\CaptureFrontBackBake\ (replacing any previous install).
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 misreads UTF-8
# without BOM, which would garble non-ASCII text (see docs/marmoset-api-notes.md).

param(
    [string]$PluginDir = "",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"
$RepoUrl    = "https://github.com/ji-eee/Marmoset-script.git"
$ZipUrl     = "https://github.com/ji-eee/Marmoset-script/archive/refs/heads/$Branch.zip"
$PluginName = "CaptureFrontBackBake"

function Write-Step([string]$msg) { Write-Host "==> $msg" -ForegroundColor Cyan }
function Write-Ok([string]$msg)   { Write-Host " OK  $msg" -ForegroundColor Green }
function Write-Note([string]$msg) { Write-Host " !   $msg" -ForegroundColor Yellow }

function Test-SourceDir([string]$dir) {
    if (-not $dir) { return $false }
    return (Test-Path (Join-Path $dir "CaptureFrontBackBake.py")) -and
           (Test-Path (Join-Path $dir "projbake"))
}

function Get-LocalSourceDir {
    # running from a cloned repo? ($PSScriptRoot is empty under 'irm | iex')
    foreach ($c in @($PSScriptRoot, (Get-Location).Path)) {
        if (Test-SourceDir $c) { return $c }
    }
    return $null
}

function Confirm-Git {
    if (Get-Command git -ErrorAction SilentlyContinue) { return $true }
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-Note "git not found and winget unavailable; will fall back to ZIP download."
        return $false
    }
    Write-Step "git not found - installing via winget (a UAC prompt may appear)..."
    try {
        winget install --id Git.Git -e --source winget `
            --accept-package-agreements --accept-source-agreements | Out-Host
        # refresh PATH for this session so 'git' resolves without a new shell
        $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                    [Environment]::GetEnvironmentVariable("Path", "User")
        if (Get-Command git -ErrorAction SilentlyContinue) {
            Write-Ok "git installed."
            return $true
        }
    } catch {
        Write-Note "winget git install failed: $($_.Exception.Message)"
    }
    Write-Note "could not install git; will fall back to ZIP download."
    return $false
}

function Get-RemoteSourceDir {
    $dest = Join-Path $env:TEMP ("Marmoset-script-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
    if (Confirm-Git) {
        Write-Step "Cloning $RepoUrl (branch: $Branch)..."
        git clone --depth 1 --branch $Branch $RepoUrl $dest 2>&1 | Out-Host
        if (Test-SourceDir $dest) { return $dest }
        Write-Note "clone did not produce the expected files; trying ZIP download."
    }
    Write-Step "Downloading repository ZIP..."
    $zip = "$dest.zip"
    Invoke-WebRequest -Uri $ZipUrl -OutFile $zip -UseBasicParsing
    Expand-Archive -Path $zip -DestinationPath $dest -Force
    Remove-Item $zip -Force -ErrorAction SilentlyContinue
    # the archive contains a single top-level folder (Marmoset-script-<branch>)
    $inner = Get-ChildItem -Path $dest -Directory | Select-Object -First 1
    if ($inner -and (Test-SourceDir $inner.FullName)) { return $inner.FullName }
    throw "Downloaded archive does not contain the plugin files."
}

function Find-MarmosetPluginDir {
    # existing user plugin folders first (Toolbag 5 preferred over 4)
    $pluginCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Marmoset Toolbag 5\plugins"),
        (Join-Path $env:LOCALAPPDATA "Marmoset Toolbag 4\plugins")
    )
    foreach ($c in $pluginCandidates) {
        if (Test-Path $c) { return $c }
    }
    # app data folder exists but plugins/ not created yet -> create it
    $appCandidates = @(
        (Join-Path $env:LOCALAPPDATA "Marmoset Toolbag 5"),
        (Join-Path $env:LOCALAPPDATA "Marmoset Toolbag 4")
    )
    foreach ($a in $appCandidates) {
        if (Test-Path $a) {
            $p = Join-Path $a "plugins"
            New-Item -ItemType Directory -Path $p -Force | Out-Null
            return $p
        }
    }
    # installation-side plugin folders (may require admin rights to write)
    $installCandidates = @(
        "C:\Program Files\Marmoset\Toolbag 5\data\plugins",
        "C:\Program Files\Marmoset\Toolbag 4\data\plugins"
    )
    foreach ($c in $installCandidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "Capture Front/Back And Bake - Marmoset plugin installer" -ForegroundColor White
Write-Host "--------------------------------------------------------"

# 1) locate the plugin source files
$cleanupDir = $null
$src = Get-LocalSourceDir
if ($src) {
    Write-Ok "Using local repo files: $src"
} else {
    $src = Get-RemoteSourceDir
    $cleanupDir = $src
    Write-Ok "Fetched repo to: $src"
}

# 2) resolve the Marmoset plugin folder
if (-not $PluginDir) {
    $PluginDir = Find-MarmosetPluginDir
    if ($PluginDir) {
        Write-Ok "Marmoset plugin folder: $PluginDir"
    } else {
        Write-Note "Could not find a Marmoset Toolbag plugin folder automatically."
        Write-Note "(usually: $env:LOCALAPPDATA\Marmoset Toolbag 5\plugins)"
        $PluginDir = Read-Host "Enter your Marmoset plugin folder path"
        if (-not $PluginDir) { throw "No plugin folder given - aborting." }
    }
} else {
    Write-Ok "Using plugin folder from -PluginDir: $PluginDir"
}
if (-not (Test-Path $PluginDir)) {
    New-Item -ItemType Directory -Path $PluginDir -Force | Out-Null
    Write-Ok "Created plugin folder: $PluginDir"
}

# 3) copy the plugin (replace any previous install)
$target = Join-Path $PluginDir $PluginName
Write-Step "Installing to $target"
try {
    if (Test-Path $target) { Remove-Item $target -Recurse -Force }
    New-Item -ItemType Directory -Path $target -Force | Out-Null
    Copy-Item (Join-Path $src "CaptureFrontBackBake.py") -Destination $target
    Copy-Item (Join-Path $src "projbake") -Destination (Join-Path $target "projbake") -Recurse
    Get-ChildItem -Path $target -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
} catch [System.UnauthorizedAccessException] {
    throw ("Access denied writing to '$target'. Re-run this script as Administrator, " +
           "or pass a user-writable folder: install.ps1 -PluginDir <path>")
}

# 4) tidy up any temp checkout
if ($cleanupDir) {
    Remove-Item $cleanupDir -Recurse -Force -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Ok "Installed: $target"
Write-Host ""
Write-Host "Next steps in Marmoset Toolbag:" -ForegroundColor White
Write-Host "  1. Edit > Plugins > Refresh"
Write-Host "  2. Edit > Plugins > $PluginName"
Write-Host ""
