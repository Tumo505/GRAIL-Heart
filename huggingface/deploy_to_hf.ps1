<#
.SYNOPSIS
    Deploy GRAIL-Heart to Hugging Face Spaces (Windows PowerShell)

.DESCRIPTION
    Prepares assets, creates the HF Space, and pushes all files.

.USAGE
    # Step 1: Generate all assets (model, demo data, source modules)
    python huggingface/prepare_demo_data.py

    # Step 2: Deploy to HF Spaces
    .\huggingface\deploy_to_hf.ps1 -HfUsername "Tumo505"

.PREREQUISITES
    - pip install huggingface_hub
    - huggingface-cli login
    - git lfs install
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$HfUsername,

    [Parameter(Mandatory=$false)]
    [string]$HfToken
)

$ErrorActionPreference = "Stop"
$SpaceName = "grail-heart"
$SpaceId = "$HfUsername/$SpaceName"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TempDir = Join-Path $env:TEMP "hf-grail-heart-deploy"

Write-Host "=" * 50
Write-Host "Deploying GRAIL-Heart to HF Space: $SpaceId"
Write-Host "=" * 50

# ── 1. Verify assets ──────────────────────────────────────────────
Write-Host "`n[1/5] Checking assets ..."

$requiredFiles = @(
    "$ScriptDir\app.py",
    "$ScriptDir\README.md",
    "$ScriptDir\requirements.txt",
    "$ScriptDir\.gitattributes",
    "$ScriptDir\Dockerfile",
    "$ScriptDir\model\best.pt"
)

foreach ($f in $requiredFiles) {
    if (-not (Test-Path $f)) {
        Write-Host "  ERROR: Missing file: $f" -ForegroundColor Red
        Write-Host "  Run first: python huggingface\prepare_demo_data.py"
        exit 1
    }
}
Write-Host "  All required files present." -ForegroundColor Green

# ── 2. Create the HF Space ────────────────────────────────────────
Write-Host "`n[2/5] Creating HF Space ..."
try {
    python -c "from huggingface_hub import HfApi; api = HfApi(token='$HfToken'); api.create_repo('$SpaceId', repo_type='space', space_sdk='docker', exist_ok=True)"
    Write-Host "  Space created." -ForegroundColor Green
} catch {
    Write-Host "  Space already exists (OK)" -ForegroundColor Yellow
}

# ── 3. Clone ──────────────────────────────────────────────────────
Write-Host "`n[3/5] Cloning Space repo ..."
if (Test-Path $TempDir) { Remove-Item -Recurse -Force $TempDir }

# Read HF token so git can authenticate
if (-not $HfToken) {
    # Try reading from the standard HF token cache file
    $tokenFile = Join-Path $env:USERPROFILE ".cache\huggingface\token"
    if (Test-Path $tokenFile) {
        $HfToken = (Get-Content $tokenFile -Raw).Trim()
    }
}

if ($HfToken) {
    git clone "https://${HfUsername}:${HfToken}@huggingface.co/spaces/$SpaceId" $TempDir
} else {
    Write-Host "  WARNING: No HF token found. Trying without auth..." -ForegroundColor Yellow
    git clone "https://huggingface.co/spaces/$SpaceId" $TempDir
}
Push-Location $TempDir

# ── 4. Copy files ─────────────────────────────────────────────────
Write-Host "`n[4/5] Copying files ..."
git lfs install

# Top-level
Copy-Item "$ScriptDir\app.py" -Destination .
Copy-Item "$ScriptDir\README.md" -Destination .
Copy-Item "$ScriptDir\requirements.txt" -Destination .
Copy-Item "$ScriptDir\.gitattributes" -Destination .
Copy-Item "$ScriptDir\Dockerfile" -Destination .

# Model
New-Item -ItemType Directory -Force -Path "model" | Out-Null
Copy-Item "$ScriptDir\model\best.pt" -Destination "model\"

# Demo data
New-Item -ItemType Directory -Force -Path "demo_data\tables" | Out-Null
Get-ChildItem "$ScriptDir\demo_data\*.csv" -ErrorAction SilentlyContinue |
    Copy-Item -Destination "demo_data\"
Get-ChildItem "$ScriptDir\demo_data\*.h5ad" -ErrorAction SilentlyContinue |
    Copy-Item -Destination "demo_data\"
Get-ChildItem "$ScriptDir\demo_data\tables\*.csv" -ErrorAction SilentlyContinue |
    Copy-Item -Destination "demo_data\tables\"

# Source modules
if (Test-Path "$ScriptDir\src") {
    Copy-Item -Recurse -Force "$ScriptDir\src" -Destination "src"
}

Write-Host "  Files copied." -ForegroundColor Green
Get-ChildItem -Recurse | Where-Object { -not $_.PSIsContainer } |
    Select-Object @{N='File';E={$_.FullName.Replace($TempDir + '\', '')}},
                  @{N='Size';E={
                      if ($_.Length -gt 1MB) { "{0:N1} MB" -f ($_.Length / 1MB) }
                      else { "{0:N0} KB" -f ($_.Length / 1KB) }
                  }} | Format-Table -AutoSize

# ── 5. Push ───────────────────────────────────────────────────────
Write-Host "`n[5/5] Pushing to Hugging Face ..."
git add .
git commit -m "Deploy GRAIL-Heart Streamlit app with model and demo data"
git push

Pop-Location

Write-Host "`n" + "=" * 50
Write-Host "Deployment complete!" -ForegroundColor Green
Write-Host "View at: https://huggingface.co/spaces/$SpaceId"
Write-Host "=" * 50
