$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$OutputDir = Join-Path $ProjectRoot "data\shdb_af"
$ManifestPath = Join-Path $OutputDir "download_manifest.txt"
$LogPath = Join-Path $OutputDir "download.log"
$BaseUrl = "https://physionet.org/files/shdb-af/1.0.1"

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if (-not (Test-Path $ManifestPath)) {
    throw "Missing manifest: $ManifestPath. Run scripts/download_shdb_af.py first."
}

"SHDB-AF download started: $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append

Get-Content $ManifestPath | ForEach-Object {
    $FileName = $_.Trim()
    if (-not $FileName) {
        return
    }

    $Target = Join-Path $OutputDir $FileName
    $Url = "$BaseUrl/$FileName`?download"
    $RemoteSize = $null
    try {
        $Headers = & curl.exe -L -s -I $Url
        $LengthLine = $Headers | Where-Object { $_ -match "content-length:" } | Select-Object -Last 1
        if ($LengthLine -match "content-length:\s*(\d+)") {
            $RemoteSize = [int64]$Matches[1]
        }
    }
    catch {
        $RemoteSize = $null
    }

    if ((Test-Path $Target) -and $RemoteSize) {
        $LocalSize = (Get-Item $Target).Length
        if ($LocalSize -eq $RemoteSize) {
            "Skipping complete $FileName ($LocalSize bytes)" | Tee-Object -FilePath $LogPath -Append
            return
        }
    }

    "Downloading $FileName : $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append

    & curl.exe -L --fail --retry 20 --retry-delay 5 --connect-timeout 30 -C - -o $Target $Url --silent --show-error 2>&1 |
        Tee-Object -FilePath $LogPath -Append

    if ($LASTEXITCODE -ne 0) {
        "FAILED $FileName with exit code $LASTEXITCODE" | Tee-Object -FilePath $LogPath -Append
        exit $LASTEXITCODE
    }

    "Finished $FileName : $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append
}

"SHDB-AF download completed: $(Get-Date -Format o)" | Tee-Object -FilePath $LogPath -Append
