# =============================================================================
#  setup.ps1  —  Volatility Forecasting Environment Setup
#  Python 3.12 · PyTorch CUDA 12.8 · TensorFlow GPU · all requirements
# =============================================================================
#  Usage:
#    powershell -ExecutionPolicy Bypass -File setup.ps1
#    powershell -ExecutionPolicy Bypass -File setup.ps1 -Force   # skip prompts
# =============================================================================
param(
    [switch]$Force   # skip confirmation prompts
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Colour helpers ─────────────────────────────────────────────────────────────
function Write-Header  { param($msg) Write-Host "`n══════════════════════════════════════════════" -ForegroundColor Cyan
                                     Write-Host "  $msg" -ForegroundColor Cyan
                                     Write-Host "══════════════════════════════════════════════" -ForegroundColor Cyan }
function Write-Step    { param($msg) Write-Host "`n▶  $msg" -ForegroundColor Yellow }
function Write-OK      { param($msg) Write-Host "  ✔  $msg" -ForegroundColor Green }
function Write-Info    { param($msg) Write-Host "  ℹ  $msg" -ForegroundColor DarkCyan }
function Write-Warn    { param($msg) Write-Host "  ⚠  $msg" -ForegroundColor DarkYellow }
function Write-Fail    { param($msg) Write-Host "`n  ✘  $msg" -ForegroundColor Red }

function Exit-Error {
    param($msg)
    Write-Fail $msg
    Write-Host ""
    exit 1
}

# ── Banner ─────────────────────────────────────────────────────────────────────
Clear-Host
Write-Host @"

  ╔══════════════════════════════════════════════════════╗
  ║   Volatility Forecasting  —  Environment Setup       ║
  ║   Python 3.12  ·  CUDA 12.8  ·  TF + PyTorch GPU    ║
  ╚══════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

$VENV_DIR  = ".venv"
$PYTHON312 = $null   # resolved below
$SCRIPT_DIR = $PSScriptRoot

# ══════════════════════════════════════════════════════════════════════════════
# 1. PRE-FLIGHT CHECKS
# ══════════════════════════════════════════════════════════════════════════════
Write-Header "1 / 6  Pre-flight checks"

# ── Python 3.12 ────────────────────────────────────────────────────────────────
Write-Step "Locating Python 3.12 ..."
$candidates = @(
    { py -3.12 -c "import sys; print(sys.executable)" 2>$null },
    { python3.12 -c "import sys; print(sys.executable)" 2>$null }
)
foreach ($c in $candidates) {
    try {
        $path = & $c
        if ($LASTEXITCODE -eq 0 -and $path -and (Test-Path $path.Trim())) {
            $PYTHON312 = $path.Trim()
            break
        }
    } catch {}
}
if (-not $PYTHON312) {
    Exit-Error "Python 3.12 not found. Install from https://python.org and ensure it is on PATH or registered with the py launcher."
}
$pyVer = & $PYTHON312 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-OK "Python $pyVer  ->  $PYTHON312"

# ── NVIDIA GPU ─────────────────────────────────────────────────────────────────
Write-Step "Detecting NVIDIA GPU ..."
try {
    $smiOut = nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $parts = ($smiOut -join "") -split ","
        Write-OK "GPU     : $($parts[0].Trim())"
        Write-OK "Driver  : $($parts[1].Trim())"
        Write-OK "VRAM    : $($parts[2].Trim())"
    } else {
        Write-Warn "nvidia-smi failed — CUDA packages will still be installed."
    }
} catch {
    Write-Warn "nvidia-smi not found — CUDA packages will still be installed."
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. CLEAN OLD ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════
Write-Header "2 / 6  Clean old environment"

$venvPath = Join-Path $SCRIPT_DIR $VENV_DIR
if (Test-Path $venvPath) {
    if (-not $Force) {
        Write-Warn "Existing virtual environment found at: $venvPath"
        $answer = Read-Host "  Delete it and start fresh? [Y/n]"
        if ($answer -match "^[Nn]") {
            Write-Info "Keeping existing environment. Exiting."
            exit 0
        }
    }
    Write-Step "Removing old virtual environment ..."
    Remove-Item -Recurse -Force $venvPath
    Write-OK "Removed $venvPath"
} else {
    Write-Info "No existing virtual environment found — nothing to clean."
}

Write-Step "Removing __pycache__ directories ..."
Get-ChildItem -Path $SCRIPT_DIR -Filter "__pycache__" -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\.git\*" } |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName }
Write-OK "Done."

# ══════════════════════════════════════════════════════════════════════════════
# 3. CREATE VIRTUAL ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════
Write-Header "3 / 6  Create virtual environment"

Write-Step "Creating .venv with Python 3.12 ..."
& $PYTHON312 -m venv $venvPath
if ($LASTEXITCODE -ne 0) { Exit-Error "Failed to create virtual environment." }
Write-OK "Created: $venvPath"

$PY = Join-Path $venvPath "Scripts\python.exe"

Write-Step "Upgrading pip, wheel, setuptools ..."
& $PY -m pip install --upgrade pip wheel setuptools --quiet
if ($LASTEXITCODE -ne 0) { Exit-Error "pip/wheel/setuptools upgrade failed." }
$pipVer = (& $PY -m pip --version) -replace "\s+", " "
Write-OK $pipVer

# ══════════════════════════════════════════════════════════════════════════════
# 4. INSTALL PYTORCH  (CUDA 12.8)
# ══════════════════════════════════════════════════════════════════════════════
Write-Header "4 / 6  PyTorch + CUDA 12.8"
Write-Info "Downloading torch (~2.8 GB) — this will take a few minutes ..."

Write-Step "Installing: torch torchvision torchaudio ..."
& $PY -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Exit-Error "PyTorch install failed." }
Write-OK "PyTorch installed."

Write-Step "Verifying PyTorch CUDA ..."
$torchCheck = & $PY -c "import torch; avail=torch.cuda.is_available(); print(f'torch={torch.__version__}'); print(f'cuda={avail}'); print(f'gpu={torch.cuda.get_device_name(0) if avail else None}')" 2>&1
$torchCheck | ForEach-Object { Write-Info $_ }
if (($torchCheck -join "") -match "cuda=True") { Write-OK "PyTorch sees the GPU" } else { Write-Warn "CUDA not detected by PyTorch — check driver." }

# ══════════════════════════════════════════════════════════════════════════════
# 5. INSTALL TENSORFLOW
# ══════════════════════════════════════════════════════════════════════════════
Write-Header "5 / 6  TensorFlow"
Write-Warn "Note: TensorFlow dropped native Windows GPU support after v2.10."
Write-Warn "      TF models (LSTM/GRU/Transformer) will run on CPU on Windows."
Write-Info  "      For TF GPU support use WSL2 (Linux) instead."
Write-Info  "      PyTorch GPU (CUDA 12.8) is fully supported and verified above."

Write-Step "Installing tensorflow 2.19.1 + CUDA runtime libraries ..."
& $PY -m pip install tensorflow==2.19.1 `
    nvidia-cublas-cu12==12.5.3.2 `
    nvidia-cudnn-cu12==9.3.0.75 `
    nvidia-cuda-runtime-cu12==12.5.82 `
    nvidia-cuda-nvcc-cu12==12.5.82 `
    nvidia-cusolver-cu12==11.6.3.83 `
    nvidia-cusparse-cu12==12.5.1.3 `
    nvidia-cufft-cu12==11.2.3.61 `
    nvidia-curand-cu12==10.3.6.82
if ($LASTEXITCODE -ne 0) { Exit-Error "TensorFlow install failed." }
Write-OK "TensorFlow 2.19.1 installed."

# Register CUDA DLL dirs so PyTorch (and any C extension) can find them at runtime
Write-Step "Writing sitecustomize.py to register CUDA DLL directories ..."
$sitePackages = (& $PY -c "import site; print(site.getsitepackages()[0])").Trim()
$cudaBins = Get-ChildItem -Path (Join-Path $sitePackages "nvidia") -Filter "bin" -Recurse -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
$lines  = "# Auto-generated by setup.ps1 -- registers pip CUDA DLLs`nimport os, pathlib`n_nvidia_bins = [`n"
$lines += ($cudaBins | ForEach-Object { "    r`"$_`"," }) -join "`n"
$lines += "`n]`nfor _p in _nvidia_bins:`n    if pathlib.Path(_p).is_dir():`n        os.add_dll_directory(_p)`n"
$lines | Set-Content -Path (Join-Path $sitePackages "sitecustomize.py") -Encoding UTF8
Write-OK "sitecustomize.py written to $sitePackages"

Write-Step "Verifying TensorFlow install ..."
$tfVer = & $PY -c "import os; os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; import tensorflow as tf; print(f'tensorflow  : {tf.__version__}'); info=tf.sysconfig.get_build_info(); print(f'CUDA build  : {info[\"is_cuda_build\"]}')" 2>&1
$tfVer | ForEach-Object { Write-Info $_ }
Write-Warn "TF CUDA build = False on Windows (expected). PyTorch handles GPU training."

# ══════════════════════════════════════════════════════════════════════════════
# 6. INSTALL REMAINING REQUIREMENTS
# ══════════════════════════════════════════════════════════════════════════════
Write-Header "6 / 6  Project requirements"

$reqFile = Join-Path $SCRIPT_DIR "requirements.txt"
if (-not (Test-Path $reqFile)) { Exit-Error "requirements.txt not found at: $reqFile" }

# Filter out torch/tensorflow lines already installed with GPU builds
$filteredReqs = Get-Content $reqFile |
    Where-Object { $_ -notmatch "^\s*#" -and $_ -notmatch "^\s*$" } |
    Where-Object { $_ -notmatch "^(torch|torchvision|torchaudio|tensorflow)" }

Write-Info "Remaining packages: $($filteredReqs -join ', ')"
$i = 0
$total = $filteredReqs.Count
$filteredReqs | ForEach-Object {
    $i++
    Write-Step "[$i/$total]  pip install $_"
    & $PY -m pip install $_ --quiet
    if ($LASTEXITCODE -ne 0) { Write-Warn "  Failed: $_ — continuing..." } else { Write-OK "  $_ installed" }
}

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "  ║                  Setup Complete  OK                  ║" -ForegroundColor Green
Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "  Activate the environment:" -ForegroundColor White
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor DarkCyan
Write-Host ""
Write-Host "  Verify GPU:" -ForegroundColor White
Write-Host "    python -c `"import torch; print(torch.cuda.is_available())`"" -ForegroundColor DarkCyan
Write-Host "    python -c `"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))`"" -ForegroundColor DarkCyan
Write-Host ""
