<#
.SYNOPSIS
    E7 driver (H2 dense oracle): capture FFN activations of the dense 31B in Docker, then
    decompose -> oracle -> markdown natively on Windows, with sampling knobs that keep it under
    one hour.

.DESCRIPTION
    pc/h2_run.ps1 1.0.0 - PowerShell 5.1, ASCII only (5.1 reads BOM-less scripts as ANSI).

    Steps (wall time printed per step, verdict line printed at the end):
      map      python -m pigemma.gguf_map build <model> --out traces/h2/map31.json --reuse
               (cached; header only, the model is NOT loaded; run first because the layer
               range check and the disk projection need n_layer / n_embd / n_ff)
      corpus   python -m pigemma.quality corpus [--check]  (data/calibration/all.txt; only when
               the capture runs)
      acts     (skipped with -SkipActs or when a usable dump exists)
               bash pc/docker_run.sh --no-stdin -- python -m pigemma.paths lock --hold
                 --min-free-gb 20 -- tools/build-linux-x86/bin/pigemma-probe --mode act-dump
                 -m /work/<31B> -f /work/data/calibration/all.txt --chunk-tokens 384 --gen 128
                 -nr -t 12 -c 1024 -b 512 -ub 128 --layers <L>
                 --kinds ffn_norm,ffn_geglu,ffn_out --tokens <T> --act-out /work/traces/h2/acts/31b
               (prefill 384 + 128 greedy tokens per chunk until T tokens; both phases recorded)
      decompose python -m pigemma.h2.decompose --map traces/h2/map31.json --layers <L>
                 --mats gate,up,down --bases <B> --group <G> --res-bits <R>
                 --out traces/h2/decomp/ --lock
      oracle   python -m pigemma.h2.oracle --decomp traces/h2/decomp/ --acts traces/h2/acts/31b/
                 --tokens <T> --eps 0.01,0.03,0.10 --out results/h2/oracle.json
                 --md docs/results/H2_RESULTS.md --layers <L> --mats <M> --bases <B>
                 --map traces/h2/map31.json --lock

    Decisions made by this driver (not in the manifest text; listed in its open issues):
      * The oracle also gets --layers/--mats/--bases/--map so that npz files left in
        traces/h2/decomp/ by other runs never leak into this run's verdict, and the Q4_0
        reference matrix is found through the map.
      * "Acts exist" means: meta.json complete, same GGUF basename, n_tokens >= -Tokens, every
        requested layer present with all three kinds. Otherwise the probe-owned files of the
        acts directory (L*.npy, phase.u8, tokens.bin, gen.txt, meta.json) are deleted and the
        dump is redone (trace.read_acts_dir would otherwise mix stale L*.npy files in).
      * -Tokens below 512 scales the chunking so both phases are present: chunk = round(0.75*T),
        gen = T - chunk (-Quick: 96 prefill + 32 generated). -c grows when chunk+gen > 1024.
      * -Quick writes to its own locations (traces/h2/acts/31b-quick, results/h2-quick/,
        docs/results/H2_RESULTS_quick.md) so it never overwrites the full E7 run; the decomp
        directory is shared (npz files are keyed by layer/matrix/base and re-validated).
      * A run record <out dir>/h2_run.json (no 'experiment' key, so results summarize skips it)
        keeps the argv, the step wall times and the verdict.
      * The capture writes into <acts>.partial; the previous dump is replaced only after the new
        one is complete and verified, so a failed or refused capture never destroys it.
      * After a fresh oracle run, older E7 envelopes next to oracle.json that cite the same
        oracle.json are deleted (they describe the overwritten document); -KeepOldEnvelopes
        keeps them.
      * The ETA printed is a formula: decompose + oracle ~60 s per (layer, matrix, base) on the
        8400F (manifest, ASSUMED; the -Quick run of 2026-10-05 MEASURED ~15 s decompose + ~15 s
        oracle per q2 case at 128 tokens, so the formula is conservative for q2 and short runs);
        act-dump = 17.65 GB / 16.4 MB/s first read through the Docker Desktop bind mount
        (MEASURED 2026-10-05, two cold -Quick runs: t_load 929.8 s and 1077.1 s; less when the VM
        page cache still holds the file) + prefill tokens / 5.61 tok/s + generated tokens /
        1.63 tok/s (MEASURED in Docker at -t 6 with the cb_eval copy, 96-token chunks; the
        second run gave 6.47 / 2.06, the slower pair is used).
      * To hold the bench lock across ALL steps, run the driver under the lock:
          python -m pigemma.paths lock --hold -- powershell -NoProfile -File pc\h2_run.ps1 ...
        (PIGEMMA_LOCK_HELD is inherited by the Docker step and by --lock of decompose/oracle).

    Exit codes:
      0  all steps done; oracle pipeline VALID (or UNKNOWN: no sanity pair, warned)
      1  a step failed (including oracle pipeline INVALID: outputs are still written and the
         verdict is printed as VOID), or an unexpected error
      2  usage (bad parameter value, unknown parameter or stray argument)
      3  missing input (model, probe binary, Git Bash, Python with numpy, map, acts with
         -SkipActs)
      4  refused: > 8 layers or > 1024 tokens without -Force, bench lock held, not enough disk,
         or a step refused (lock busy / low RAM inside the step)
      130 a step was interrupted
    A step's own exit code 1/2/3/4/130 is forwarded; any other code (docker 125/126/127, ...)
    becomes 1. The failing step and its real code are always printed.

.PARAMETER Layers
    Layers to sample (comma list, ranges a-b allowed). Default 5,20,35,50; -Quick: 20,45.
.PARAMETER Tokens
    Tokens to capture and analyse (prefill + generated). Default 512; -Quick: 128.
.PARAMETER Group
    Input columns per residual group. Default 128.
.PARAMETER Bases
    Base approximations (q2, q3, lowrank256, q2+lowrank128, ...). Default q2,lowrank256;
    -Quick: q2.
.PARAMETER ResBits
    Residual bits per element: 2,3,4,5,6,8 or 16. Default 4.
.PARAMETER Force
    Allow more than 8 layers or more than 1024 tokens.
.PARAMETER SkipActs
    Never run the Docker capture; use the existing acts directory (exit 3 if it is missing).
.PARAMETER Quick
    Layers 20,45; tokens 128; bases q2; group 128 (explicit parameters still win).
.PARAMETER Mats
    FFN matrices (subset of gate,up,down). Default all three.
.PARAMETER Eps
    Relative output-error targets for the oracle. Default 0.01,0.03,0.10.
.PARAMETER Model
    Registry id or GGUF path of the dense model. Default 31b-q4.
.PARAMETER Corpus
    Text file (inside the repo) fed to the probe with -f. Default data/calibration/all.txt
    (rebuilt with python -m pigemma.quality corpus when missing or out of date). Its first
    chunk is C++ source, so the greedy 'gen' phase of short runs continues code; pass e.g.
    data/calibration/en_tech_moe_streaming.txt to probe prose generation instead.
.PARAMETER Threads
    Probe threads in Docker. Default 12.
.PARAMETER ChunkTokens
    Prefill tokens per chunk (0 = auto: 384, or round(0.75*Tokens) when Tokens < 512).
.PARAMETER Gen
    Greedy tokens generated per chunk (-1 = auto: 128, or Tokens - chunk when Tokens < 512).
.PARAMETER Ctx
    Context size for the probe (0 = auto: max(1024, chunk+gen rounded up to 256)).
.PARAMETER ActsDir
    Activation dump directory, inside the repo (default traces/h2/acts/31b[-quick]).
.PARAMETER DecompDir
    Decomposition output directory (default traces/h2/decomp).
.PARAMETER MapPath
    map.json of the model (default traces/h2/map31.json for 31b-q4).
.PARAMETER Out
    oracle.json path (default results/h2/oracle.json; -Quick: results/h2-quick/oracle.json).
.PARAMETER Md
    Markdown report (default docs/results/H2_RESULTS.md; -Quick: docs/results/H2_RESULTS_quick.md).
.PARAMETER RedoActs
    Re-capture the activations even if a usable dump exists.
.PARAMETER Recompute
    Pass --force to decompose (recompute npz files that would otherwise be reused).
.PARAMETER BuildProbe
    Build tools/build-linux-x86/bin/pigemma-probe first if it is missing
    (bash pc/docker_run.sh build-tools --targets pigemma-probe).
.PARAMETER KeepOldEnvelopes
    Keep E7 envelopes of earlier runs next to oracle.json. By default, after a fresh oracle run,
    older E7 envelopes in that directory whose inputs[0] is the same oracle.json (which this run
    overwrote) are deleted so results summaries never mix superseded projections.
.PARAMETER NoLock
    Do not take results/.bench.lock (debugging only; never while a benchmark runs).
.PARAMETER MinFreeGb
    Available-RAM gate in GB for the bench lock of the Docker capture (default 20).
.PARAMETER MinFreeDiskGb
    Disk that must stay free after the projected outputs (default 2).
.PARAMETER Python
    Python interpreter (default $env:PIGEMMA_PYTHON or 'python').
.PARAMETER DryRun
    Print gates, ETA and every command (docker_run.sh --dry-run, decompose --dry-run);
    load no model, write nothing.
.PARAMETER Help
    Print the usage text and exit 0.

.EXAMPLE
    .\pc\h2_run.ps1 -Quick
.EXAMPLE
    .\pc\h2_run.ps1 -Layers 5,20,35,50 -Tokens 512 -Group 128 -Bases q2,lowrank256
.EXAMPLE
    .\pc\h2_run.ps1 -SkipActs -Bases q3,lowrank512 -DryRun
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string[]]$Layers = @('5', '20', '35', '50'),
    [int]$Tokens = 512,
    [int]$Group = 128,
    [string[]]$Bases = @('q2', 'lowrank256'),
    [int]$ResBits = 4,
    [switch]$Force,
    [switch]$SkipActs,
    [switch]$Quick,
    [string[]]$Mats = @('gate', 'up', 'down'),
    [string[]]$Eps = @('0.01', '0.03', '0.10'),
    [string]$Model = '31b-q4',
    [string]$Corpus = 'data/calibration/all.txt',
    [int]$Threads = 12,
    [int]$ChunkTokens = 0,
    [int]$Gen = -1,
    [int]$Ctx = 0,
    [string]$ActsDir = '',
    [string]$DecompDir = 'traces/h2/decomp',
    [string]$MapPath = '',
    [string]$Out = '',
    [string]$Md = '',
    [switch]$RedoActs,
    [switch]$Recompute,
    [switch]$BuildProbe,
    [switch]$NoLock,
    [switch]$KeepOldEnvelopes,
    [double]$MinFreeGb = 20,
    [double]$MinFreeDiskGb = 2,
    [string]$Python = '',
    [switch]$DryRun,
    [switch]$Help,
    # Catches unknown parameters / stray arguments so they exit 2 (usage) instead of
    # PowerShell's own binding error (exit 1 or -1 depending on the host).
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$UnknownArgs
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# ------------------------------------------------------------------------------------------
# constants
# ------------------------------------------------------------------------------------------
$H2RunVersion = '1.0.0'
$MaxLayersWithoutForce = 8
$MaxTokensWithoutForce = 1024
$EtaSecondsPerCase = 60          # manifest: ~60 s per (layer, matrix, base) on the 8400F (ASSUMED)
$Model31BBytes = 17651001568     # 31b-q4 file size (paths.MODELS)
$BindMountMBps = 16.4            # MEASURED 2026-10-05 (-Quick, 2 cold runs): 17.65 GB mmap first read in 929.8 s and 1077.1 s through the Docker Desktop bind mount (slower one used)
$PrefillTokS31B = 5.61           # MEASURED 2026-10-05 (-Quick): probe prefill of the 31B in Docker, -t 6, 96-token chunk, cb_eval copy included
$GenTokS31B = 1.63               # MEASURED 2026-10-05 (-Quick): probe greedy generation of the 31B in Docker, -t 6, cb_eval copy included
$ActKinds = @('ffn_norm', 'ffn_geglu', 'ffn_out')
$ProbeRel = 'tools/build-linux-x86/bin/pigemma-probe'
$DefaultCorpusRel = 'data/calibration/all.txt'   # NB: PS variable names are case-insensitive
$ValidMats = @('gate', 'up', 'down')
$ValidResBits = @(2, 3, 4, 5, 6, 8, 16)
$Inv = [System.Globalization.CultureInfo]::InvariantCulture

$UsageText = @'
pc/h2_run.ps1 - E7 driver: 31B activation capture in Docker, then decompose -> oracle -> markdown

Usage:
  .\pc\h2_run.ps1 [-Layers 5,20,35,50] [-Tokens 512] [-Group 128] [-Bases q2,lowrank256]
                  [-ResBits 4] [-Force] [-SkipActs] [-Quick]
                  [-Mats gate,up,down] [-Eps 0.01,0.03,0.10] [-Model 31b-q4] [-Threads 12]
                  [-Corpus data/calibration/all.txt]
                  [-ChunkTokens N] [-Gen N] [-Ctx N] [-ActsDir DIR] [-DecompDir DIR]
                  [-MapPath F] [-Out oracle.json] [-Md REPORT.md] [-RedoActs] [-Recompute]
                  [-BuildProbe] [-NoLock] [-KeepOldEnvelopes] [-MinFreeGb 20] [-MinFreeDiskGb 2]
                  [-Python EXE]
                  [-DryRun] [-Help]

  -Quick      layers 20,45; tokens 128 (96 prefill + 32 generated); bases q2; group 128;
              outputs under traces/h2/acts/31b-quick, results/h2-quick/,
              docs/results/H2_RESULTS_quick.md
  -Force      allow more than 8 layers or more than 1024 tokens
              ETA formula (ASSUMED, Ryzen 5 8400F): decompose + oracle ~60 s per
              (layer, matrix, base); act-dump = 17.65 GB / 16.4 MB/s first read through the
              bind mount (~18 min, MEASURED) + prefill/5.61 tok/s + generated/1.63 tok/s
              (MEASURED in Docker at -t 6)
  -SkipActs   use the existing activation dump (exit 3 if missing)
  -DryRun     print gates, ETA and commands; load nothing, write nothing

Steps: map (gguf_map build --reuse) -> corpus -> acts (Docker, bench lock) -> decompose (--lock)
       -> oracle (--lock); the wall time of each step and the verdict line are printed at the end.
Exit: 0 ok | 1 step failed / pipeline INVALID | 2 usage | 3 missing input | 4 refused | 130 interrupted
'@

# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------
function Write-Log([string]$Message) {
    Write-Host ('h2_run: ' + $Message)
}

function Write-Warn([string]$Message) {
    Write-Host ('h2_run: WARNING: ' + $Message) -ForegroundColor Yellow
}

function Write-Err([string]$Message) {
    [Console]::Error.WriteLine('h2_run: ' + $Message)
}

function Exit-H2([int]$Code, [string]$Message) {
    # Unwinds to the top-level handler, which prints $Message and exits with $Code.
    $ex = New-Object System.Exception $Message
    $ex.Data['h2_exit'] = $Code
    throw $ex
}

function Format-Num($Value, [string]$Fmt = '0.000') {
    if ($null -eq $Value) { return 'n/a' }
    return ([double]$Value).ToString($Fmt, $Inv)
}

function Format-Gb($Bytes) {
    if ($null -eq $Bytes) { return 'n/a' }
    return (([double]$Bytes) / 1e9).ToString('0.00', $Inv) + ' GB'
}

function Format-Duration([double]$Seconds) {
    if ($Seconds -lt 120) { return $Seconds.ToString('0', $Inv) + ' s' }
    return ($Seconds / 60).ToString('0.0', $Inv) + ' min'
}

function Get-Prop($Object, [string]$Name, $Default = $null) {
    # StrictMode-safe property read on ConvertFrom-Json objects.
    if ($null -eq $Object) { return $Default }
    $p = $Object.PSObject.Properties[$Name]
    if ($null -eq $p) { return $Default }
    return $p.Value
}

function Read-JsonFile([string]$Path) {
    $text = [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
    return ($text | ConvertFrom-Json)
}

function Write-JsonFile([string]$Path, $Object) {
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $json = $Object | ConvertTo-Json -Depth 8
    $tmp = $Path + '.tmp'
    [System.IO.File]::WriteAllText($tmp, ($json -replace "`r`n", "`n") + "`n", (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Split-List([string[]]$Values) {
    $out = New-Object System.Collections.Generic.List[string]
    foreach ($v in $Values) {
        if ($null -eq $v) { continue }
        foreach ($tok in ([string]$v -split '[,\s]+')) {
            if ($tok -ne '' -and -not $out.Contains($tok)) { $out.Add($tok) }
        }
    }
    return , $out.ToArray()
}

function ConvertTo-LayerList([string[]]$Values) {
    $out = New-Object System.Collections.Generic.List[int]
    foreach ($tok in (Split-List $Values)) {
        if ($tok -match '^(\d+)-(\d+)$') {
            $a = [int]$Matches[1]; $b = [int]$Matches[2]
            if ($b -lt $a) { Exit-H2 2 "-Layers: empty range '$tok'" }
            for ($i = $a; $i -le $b; $i++) { if (-not $out.Contains($i)) { $out.Add($i) } }
        } elseif ($tok -match '^\d+$') {
            $i = [int]$tok
            if (-not $out.Contains($i)) { $out.Add($i) }
        } else {
            Exit-H2 2 "-Layers: '$tok' is not a layer index or a range a-b"
        }
    }
    if ($out.Count -eq 0) { Exit-H2 2 '-Layers is empty' }
    return , $out.ToArray()
}

function Quote-Arg([string]$Arg) {
    if ($Arg -eq '') { return "''" }
    if ($Arg -match '[\s"''$`|&;<>()*?]') { return "'" + ($Arg -replace "'", "'\''") + "'" }
    return $Arg
}

function Format-Command([string]$Exe, [string[]]$ArgList) {
    $parts = @($Exe) + @($ArgList | ForEach-Object { Quote-Arg ([string]$_) })
    return ($parts -join ' ')
}

function Invoke-Native([string]$Exe, [string[]]$ArgList) {
    # Runs a native program with its stdout streamed to the host (not captured into the
    # function's output) and stderr straight to the console. PS 5.1 wraps native stderr into
    # error records when ErrorActionPreference is Stop, so it is relaxed here.
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $code = 127
    try {
        & $Exe @ArgList | Out-Host
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
    } catch [System.Management.Automation.CommandNotFoundException] {
        Write-Err ("not found: " + $Exe)
        $code = 127
    } finally {
        $ErrorActionPreference = $old
    }
    return [int]$code
}

function Invoke-Capture([string]$Exe, [string[]]$ArgList) {
    # Like Invoke-Native but returns @{Code; Text} with stdout captured (stderr to the console).
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $text = ''
    $code = 127
    try {
        $lines = & $Exe @ArgList
        $code = $LASTEXITCODE
        if ($null -eq $code) { $code = 0 }
        if ($null -ne $lines) { $text = (@($lines) | ForEach-Object { [string]$_ }) -join "`n" }
    } catch [System.Management.Automation.CommandNotFoundException] {
        $code = 127
    } finally {
        $ErrorActionPreference = $old
    }
    return @{ Code = [int]$code; Text = $text }
}

function ConvertTo-ExitCode([int]$Code) {
    if (@(0, 1, 2, 3, 4, 130) -contains $Code) { return $Code }
    return 1
}

$script:Steps = New-Object System.Collections.Generic.List[object]

function Add-StepRecord([string]$Name, [string]$Status, [double]$Seconds, $Code, [string]$Command) {
    $script:Steps.Add([pscustomobject]@{
            step    = $Name
            status  = $Status
            seconds = [math]::Round($Seconds, 1)
            exit    = $Code
            command = $Command
        })
}

function Invoke-Step([string]$Name, [string]$Exe, [string[]]$ArgList, [string]$Display = '') {
    if ($Display -eq '') { $Display = Format-Command $Exe $ArgList }
    Write-Log ("---- step {0}: {1}" -f $Name, $Display)
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    $code = Invoke-Native $Exe $ArgList
    $sw.Stop()
    $secs = $sw.Elapsed.TotalSeconds
    $status = 'ok'
    if ($code -ne 0) { $status = 'failed' }
    Add-StepRecord $Name $status $secs $code $Display
    Write-Log ("---- step {0}: exit {1} after {2}" -f $Name, $code, (Format-Duration $secs))
    return $code
}

function Find-GitBash {
    $cands = New-Object System.Collections.Generic.List[string]
    if ($env:PIGEMMA_BASH) { $cands.Add($env:PIGEMMA_BASH) }
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($null -ne $git -and $git.Source) {
        $root = Split-Path -Parent (Split-Path -Parent $git.Source)
        $cands.Add((Join-Path $root 'bin\bash.exe'))
        $cands.Add((Join-Path $root 'usr\bin\bash.exe'))
    }
    foreach ($base in @($env:ProgramFiles, $env:ProgramW6432, ${env:ProgramFiles(x86)}, (Join-Path ([string]$env:LOCALAPPDATA) 'Programs'))) {
        if ($base) { $cands.Add((Join-Path $base 'Git\bin\bash.exe')) }
    }
    foreach ($c in $cands) {
        if ($c -and (Test-Path -LiteralPath $c -PathType Leaf)) { return (Resolve-Path -LiteralPath $c).Path }
    }
    # Last resort: a bash on PATH that is not WSL's System32\bash.exe or a WindowsApps stub.
    $sysroot = [string]$env:SystemRoot
    foreach ($b in @(Get-Command bash -All -ErrorAction SilentlyContinue)) {
        $src = [string]$b.Source
        if ($src -and -not ($sysroot -and $src.StartsWith($sysroot, [System.StringComparison]::OrdinalIgnoreCase)) -and $src -notmatch 'WindowsApps') {
            return $src
        }
    }
    return $null
}

function Get-FreeBytes([string]$Path) {
    try {
        $full = [System.IO.Path]::GetFullPath($Path)
        $root = [System.IO.Path]::GetPathRoot($full)
        $di = New-Object System.IO.DriveInfo($root)
        return [int64]$di.AvailableFreeSpace
    } catch {
        return $null
    }
}

function Get-RepoRelative([string]$Root, [string]$Path) {
    # Absolute path -> repo-relative with forward slashes, or $null when outside the repo.
    $full = [System.IO.Path]::GetFullPath($Path).TrimEnd('\', '/')
    $r = $Root.TrimEnd('\', '/')
    if ($full.Equals($r, [System.StringComparison]::OrdinalIgnoreCase)) { return '' }
    if ($full.StartsWith($r + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        return $full.Substring($r.Length + 1).Replace('\', '/')
    }
    return $null
}

function Resolve-RepoPath([string]$Root, [string]$Path) {
    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
    return [System.IO.Path]::GetFullPath((Join-Path $Root $Path))
}

function Get-ChunkPlan([int]$T, [int]$Chunk, [int]$GenPer) {
    # Mirrors the probe loop: chunks of $Chunk prefill tokens (BOS included) + $GenPer generated,
    # stopping once $T tokens are recorded.
    $pre = 0; $gen = 0; $n = 0; $left = $T
    while ($left -gt 0) {
        $p = [math]::Min($Chunk, $left); $pre += $p; $left -= $p; $n++
        if ($left -le 0) { break }
        $g = [math]::Min($GenPer, $left); $gen += $g; $left -= $g
        if ($Chunk -le 0) { break }
    }
    return @{ Prefill = $pre; Gen = $gen; Chunks = $n }
}

function Test-Acts([string]$Dir, [int[]]$WantLayers, [int]$WantTokens, [string]$GgufBasename, [int64]$GgufSize, [string]$CorpusCtr = '') {
    $r = [ordered]@{ Exists = $false; Usable = $false; Reason = ''; NTokens = $null; Layers = @(); Meta = $null }
    $metaPath = Join-Path $Dir 'meta.json'
    if (-not (Test-Path -LiteralPath $metaPath -PathType Leaf)) {
        $r.Reason = "no meta.json in $Dir"
        return [pscustomobject]$r
    }
    $r.Exists = $true
    try {
        $m = Read-JsonFile $metaPath
    } catch {
        $r.Reason = "meta.json is not valid JSON: $($_.Exception.Message)"
        return [pscustomobject]$r
    }
    $r.Meta = $m
    $nt = Get-Prop $m 'n_tokens'
    $r.NTokens = $nt
    $have = @(Get-Prop $m 'layers' @()) | ForEach-Object { [int]$_ }
    $r.Layers = $have
    $reasons = New-Object System.Collections.Generic.List[string]
    if ((Get-Prop $m 'complete') -ne $true) { $reasons.Add('dump incomplete (meta.complete is not true; error: ' + [string](Get-Prop $m 'error' 'n/a') + ')') }
    $model = [string](Get-Prop $m 'model' '')
    if ($GgufBasename -and $model -ne $GgufBasename) { $reasons.Add("captured from '$model', not '$GgufBasename'") }
    $size = Get-Prop $m 'gguf_size_bytes'
    if ($null -ne $size -and $GgufSize -gt 0 -and [int64]$size -ne $GgufSize) { $reasons.Add("GGUF size $size differs from $GgufSize") }
    if ($null -eq $nt -or [int]$nt -lt $WantTokens) { $reasons.Add("n_tokens $nt < $WantTokens") }
    $flags = @(Get-Prop $m 'flags' @())
    if ($CorpusCtr -and $flags.Count -gt 1) {
        $fi = [array]::IndexOf([object[]]$flags, '-f')
        if ($fi -ge 0 -and $fi + 1 -lt $flags.Count -and [string]$flags[$fi + 1] -ne $CorpusCtr) {
            $reasons.Add("captured from corpus '$($flags[$fi + 1])', not '$CorpusCtr'")
        }
    }
    $missing = @($WantLayers | Where-Object { $have -notcontains $_ })
    if ($missing.Count -gt 0) { $reasons.Add('layers missing: ' + ($missing -join ',')) }
    $kinds = Get-Prop $m 'kinds'
    foreach ($k in $ActKinds) {
        if ($null -eq (Get-Prop $kinds $k)) { $reasons.Add("kind $k missing") }
    }
    if ($missing.Count -eq 0) {
        foreach ($il in $WantLayers) {
            foreach ($k in $ActKinds) {
                $f = Join-Path $Dir ("L{0}.{1}.npy" -f $il, $k)
                if (-not (Test-Path -LiteralPath $f -PathType Leaf)) { $reasons.Add("file missing: L$il.$k.npy") }
            }
        }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Dir 'phase.u8') -PathType Leaf)) { $reasons.Add('phase.u8 missing') }
    if ($reasons.Count -eq 0) {
        $r.Usable = $true
        $r.Reason = "usable: $nt tokens, layers " + ($have -join ',')
    } else {
        $r.Reason = ($reasons -join '; ')
    }
    return [pscustomobject]$r
}

function Remove-ProbeFiles([string]$Dir) {
    if (-not (Test-Path -LiteralPath $Dir -PathType Container)) { return }
    $victims = @(Get-ChildItem -LiteralPath $Dir -File | Where-Object {
            $_.Name -match '^L\d+\.[A-Za-z0-9_]+\.npy$' -or @('phase.u8', 'tokens.bin', 'gen.txt', 'meta.json') -contains $_.Name
        })
    if ($victims.Count -gt 0) {
        Write-Log ("removing {0} stale probe file(s) from {1} before the new dump: {2}" -f $victims.Count, $Dir, (($victims | ForEach-Object { $_.Name }) -join ' '))
        $victims | Remove-Item -Force
    }
}

function Remove-SupersededEnvelopes([string]$Dir, [string]$OracleRel, [DateTime]$Since) {
    # E7 envelopes are projections derived from oracle.json; the oracle step just overwrote it, so
    # envelopes of earlier runs that cite the same oracle.json (inputs[0]) are stale.
    $cut = $Since.ToString('yyyy-MM-ddTHH:mm:ssZ', $Inv)
    $want = $OracleRel.Replace([char]92, [char]47)   # backslash -> slash
    foreach ($f in @(Get-ChildItem -LiteralPath $Dir -Filter 'E7_*.json' -File -ErrorAction SilentlyContinue)) {
        try { $e = Read-JsonFile $f.FullName } catch { continue }
        if ([string](Get-Prop $e 'experiment' '') -ne 'E7') { continue }
        $inputs = @(Get-Prop $e 'inputs' @())
        if ($inputs.Count -eq 0 -or ([string]$inputs[0]).Replace([char]92, [char]47) -ne $want) { continue }
        $created = [string](Get-Prop $e 'created_at' '')
        if ($created -and [string]::CompareOrdinal($created, $cut) -lt 0) {
            Write-Log ("retiring superseded envelope {0} (created {1}, cites {2})" -f $f.Name, $created, $want)
            Remove-Item -LiteralPath $f.FullName -Force
        }
    }
}

function Get-AvailableRamBytes {
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        return [int64]$os.FreePhysicalMemory * 1024
    } catch {
        return $null
    }
}

# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------
function Invoke-H2Run {
    $repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
    Set-Location -LiteralPath $repo
    $startedAt = [DateTime]::UtcNow
    $wall = [System.Diagnostics.Stopwatch]::StartNew()

    # ---- -Quick defaults (explicit parameters win) ----
    $bound = $script:BoundParams
    $quickNotes = New-Object System.Collections.Generic.List[string]
    $selLayers = $Layers; $T = $Tokens; $selBases = $Bases; $G = $Group   # NB: PS names are case-insensitive ($b loops below)
    if ($Quick) {
        if (-not $bound.ContainsKey('Layers')) { $selLayers = @('20', '45') } else { $quickNotes.Add('-Layers') }
        if (-not $bound.ContainsKey('Tokens')) { $T = 128 } else { $quickNotes.Add('-Tokens') }
        if (-not $bound.ContainsKey('Bases')) { $selBases = @('q2') } else { $quickNotes.Add('-Bases') }
        if (-not $bound.ContainsKey('Group')) { $G = 128 } else { $quickNotes.Add('-Group') }
        if ($quickNotes.Count -gt 0) { Write-Log ('-Quick: explicit ' + ($quickNotes -join ', ') + ' kept') }
    }

    # ---- validate parameters ----
    $layerList = ConvertTo-LayerList $selLayers
    $baseList = Split-List $selBases
    if ($baseList.Count -eq 0) { Exit-H2 2 '-Bases is empty' }
    foreach ($b in $baseList) {
        if ($b -notmatch '^(none|q[2-8]|lowrank[1-9]\d*|q[2-8]\+lowrank[1-9]\d*)$') {
            Exit-H2 2 "-Bases: unknown base '$b' (q2..q8, lowrank<r>, q<b>+lowrank<r>, none)"
        }
    }
    $matList = Split-List $Mats
    if ($matList.Count -eq 0) { Exit-H2 2 '-Mats is empty' }
    foreach ($m in $matList) { if ($ValidMats -notcontains $m) { Exit-H2 2 "-Mats: unknown matrix '$m' (gate, up, down)" } }
    $epsList = Split-List $Eps
    foreach ($e in $epsList) {
        $d = 0.0
        if (-not [double]::TryParse($e, [System.Globalization.NumberStyles]::Float, $Inv, [ref]$d) -or $d -le 0 -or $d -ge 1) {
            Exit-H2 2 "-Eps: '$e' is not a number in (0, 1)"
        }
    }
    if ($epsList.Count -eq 0) { Exit-H2 2 '-Eps is empty' }
    if ($T -lt 2) { Exit-H2 2 "-Tokens must be >= 2 (got $T)" }
    if ($G -lt 1) { Exit-H2 2 "-Group must be >= 1 (got $G)" }
    if ($ValidResBits -notcontains $ResBits) { Exit-H2 2 "-ResBits must be one of $($ValidResBits -join ',') (got $ResBits)" }
    if ($Threads -lt 1) { Exit-H2 2 "-Threads must be >= 1 (got $Threads)" }
    if ($MinFreeGb -lt 0 -or $MinFreeDiskGb -lt 0) { Exit-H2 2 '-MinFreeGb / -MinFreeDiskGb must be >= 0' }
    if ($SkipActs -and $RedoActs) { Exit-H2 2 '-SkipActs and -RedoActs are mutually exclusive' }

    # chunking: 384 + 128 per chunk, scaled down for short runs so both phases are present
    $chunk = $ChunkTokens; $genPer = $Gen
    if ($chunk -eq 0) {
        if ($T -ge 512) { $chunk = 384 } else { $chunk = [int][math]::Max(2, [math]::Round($T * 0.75)) }
    }
    if ($genPer -lt 0) {
        if ($T -ge 512) { $genPer = 128 } else { $genPer = [int][math]::Max(0, $T - $chunk) }
    }
    if ($chunk -lt 2) { Exit-H2 2 "-ChunkTokens must be >= 2 (got $chunk)" }
    $ctxEff = $Ctx
    if ($ctxEff -eq 0) { $ctxEff = [int][math]::Max(1024, [math]::Ceiling(($chunk + $genPer) / 256.0) * 256) }
    if ($ctxEff -lt $chunk) { Exit-H2 2 "-Ctx $ctxEff is smaller than one chunk ($chunk tokens)" }
    $plan = Get-ChunkPlan $T $chunk $genPer

    # ---- ETA (formula, ASSUMED) and the size limits ----
    $nCases = $layerList.Count * $matList.Count * $baseList.Count
    $etaNum = $EtaSecondsPerCase * $nCases
    $etaLoad = $Model31BBytes / ($BindMountMBps * 1e6)
    $etaActs = $etaLoad + $plan.Prefill / $PrefillTokS31B + $plan.Gen / $GenTokS31B
    $etaFormula = ("ETA formula (Ryzen 5 8400F): decompose + oracle ~{0} s x {1} layers x {2} matrices x {3} bases = {4} (ASSUMED, manifest); " +
        "act-dump = 17.65 GB / {5} MB/s first read through the bind mount ({6}; less if the Docker VM page cache still holds the file) + " +
        "{7} prefill tok / {8} tok/s + {9} generated tok / {10} tok/s = {11} (rates MEASURED 2026-10-05 in Docker at -t 6; -t 12 untested)") -f `
        $EtaSecondsPerCase, $layerList.Count, $matList.Count, $baseList.Count, (Format-Duration $etaNum),
    $BindMountMBps.ToString($Inv), (Format-Duration $etaLoad), $plan.Prefill, $PrefillTokS31B.ToString($Inv), $plan.Gen,
    $GenTokS31B.ToString($Inv), (Format-Duration $etaActs)
    $over = New-Object System.Collections.Generic.List[string]
    if ($layerList.Count -gt $MaxLayersWithoutForce) { $over.Add("$($layerList.Count) layers > $MaxLayersWithoutForce") }
    if ($T -gt $MaxTokensWithoutForce) { $over.Add("$T tokens > $MaxTokensWithoutForce") }
    if ($over.Count -gt 0 -and -not $Force) {
        Write-Log $etaFormula
        Exit-H2 4 ('REFUSED: ' + ($over -join ' and ') + ' without -Force (the run would exceed the one-hour budget; see the ETA formula above)')
    }

    # ---- python ----
    $py = $Python
    if (-not $py) { $py = $env:PIGEMMA_PYTHON }
    if (-not $py) { $py = 'python' }
    $probePy = Invoke-Capture $py @('-c', 'import sys, numpy, pigemma.paths; print(sys.version.split()[0])')
    if ($probePy.Code -ne 0) {
        Exit-H2 3 "missing: '$py' cannot import numpy + pigemma.paths from $repo (exit $($probePy.Code)); pass -Python or set PIGEMMA_PYTHON"
    }

    # ---- model ----
    $mres = Invoke-Capture $py @('-m', 'pigemma.paths', '--log-level', 'WARNING', 'model', $Model, '--require')
    if ($mres.Code -ne 0) {
        Exit-H2 (ConvertTo-ExitCode $mres.Code) "missing: model '$Model' not found or wrong size (python -m pigemma.paths model $Model --require exit $($mres.Code))"
    }
    $modelPath = ($mres.Text -split "`n" | Where-Object { $_.Trim() -ne '' } | Select-Object -Last 1).Trim()
    $modelItem = Get-Item -LiteralPath $modelPath
    $modelBase = $modelItem.Name
    $modelSize = [int64]$modelItem.Length
    $tag = $Model
    if ($tag -match '[\\/]' -or $tag -match '\.gguf$') { $tag = [System.IO.Path]::GetFileNameWithoutExtension($modelBase) }
    if ($tag -eq '31b-q4' -or $modelBase -eq 'gemma-4-31B_q4_0-it.gguf') { $tag = '31b' }

    # ---- default locations ----
    $actsRel = $ActsDir
    if (-not $actsRel) {
        $actsRel = "traces/h2/acts/$tag"
        if ($Quick) { $actsRel += '-quick' }
    }
    $mapRel = $MapPath
    if (-not $mapRel) {
        if ($tag -eq '31b') { $mapRel = 'traces/h2/map31.json' } else { $mapRel = "traces/h2/map_$tag.json" }
    }
    $outRel = $Out
    if (-not $outRel) { if ($Quick) { $outRel = 'results/h2-quick/oracle.json' } else { $outRel = 'results/h2/oracle.json' } }
    $mdRel = $Md
    if (-not $mdRel) { if ($Quick) { $mdRel = 'docs/results/H2_RESULTS_quick.md' } else { $mdRel = 'docs/results/H2_RESULTS.md' } }
    $actsAbs = Resolve-RepoPath $repo $actsRel
    $decompAbs = Resolve-RepoPath $repo $DecompDir
    $mapAbs = Resolve-RepoPath $repo $mapRel
    $outAbs = Resolve-RepoPath $repo $outRel
    $mdAbs = Resolve-RepoPath $repo $mdRel
    # python gets repo-relative forward-slash paths when possible (cwd = repo root)
    $actsArg = Get-RepoRelative $repo $actsAbs; if ($null -eq $actsArg) { $actsArg = $actsAbs }
    $decompArg = Get-RepoRelative $repo $decompAbs; if ($null -eq $decompArg) { $decompArg = $decompAbs }
    $mapArg = Get-RepoRelative $repo $mapAbs; if ($null -eq $mapArg) { $mapArg = $mapAbs }
    $outArg = Get-RepoRelative $repo $outAbs; if ($null -eq $outArg) { $outArg = $outAbs }
    $mdArg = Get-RepoRelative $repo $mdAbs; if ($null -eq $mdArg) { $mdArg = $mdAbs }

    $corpusAbs = Resolve-RepoPath $repo $Corpus
    $corpusRepoRel = Get-RepoRelative $repo $corpusAbs
    $corpusCtr = ''
    if ($null -ne $corpusRepoRel -and $corpusRepoRel -ne '') { $corpusCtr = '/work/' + $corpusRepoRel }
    $isDefaultCorpus = ($corpusRepoRel -eq $DefaultCorpusRel)

    $layersCsv = ($layerList -join ',')
    $basesCsv = ($baseList -join ',')
    $matsCsv = ($matList -join ',')
    $epsCsv = ($epsList -join ',')
    $minFreeGbStr = $MinFreeGb.ToString($Inv)

    # ---- header ----
    $mode = 'RUN'
    if ($DryRun) { $mode = 'DRY RUN (nothing is loaded or written)' }
    Write-Log ("pc/h2_run.ps1 {0} - E7 H2 dense oracle - {1} - {2}" -f $H2RunVersion, $startedAt.ToString('yyyy-MM-ddTHH:mm:ssZ', $Inv), $mode)
    Write-Log ("python  {0} (Python {1})" -f $py, $probePy.Text.Trim())
    Write-Log ("model   {0} ({1}, {2})" -f $Model, $modelPath, (Format-Gb $modelSize))
    $quickLabel = ''
    if ($Quick) { $quickLabel = ' [-Quick]' }
    Write-Log ("sample  layers {0}; mats {1}; bases {2}; group {3}; res_bits {4}; tokens {5}; eps {6}{7}" -f $layersCsv, $matsCsv, $basesCsv, $G, $ResBits, $T, $epsCsv, $quickLabel)
    Write-Log ("capture -f {7}; chunk {0} prefill + {1} generated per chunk -> {2} chunk(s), {3} prefill + {4} generated tokens; -t {5} -c {6} -b 512 -ub 128 -nr" -f $chunk, $genPer, $plan.Chunks, $plan.Prefill, $plan.Gen, $Threads, $ctxEff, $Corpus)
    Write-Log ("paths   acts {0} | map {1} | decomp {2} | out {3} | md {4}" -f $actsArg, $mapArg, $decompArg, $outArg, $mdArg)
    $freeRepo = Get-FreeBytes $repo
    $sysDrive = [string]$env:SystemDrive
    if (-not $sysDrive) { $sysDrive = 'C:' }
    $freeSys = Get-FreeBytes ($sysDrive + '\')
    $ramAvail = Get-AvailableRamBytes
    Write-Log ("free    disk {0} {1} (repo) | disk {2} {3} (Docker Desktop VM disk by default) | RAM available {4}" -f `
            ([System.IO.Path]::GetPathRoot($repo)), (Format-Gb $freeRepo), $sysDrive, (Format-Gb $freeSys), (Format-Gb $ramAvail))
    Write-Log $etaFormula
    if ($over.Count -gt 0) { Write-Warn ('-Force: ' + ($over -join ' and ') + ' accepted') }

    # ---- bench lock pre-check ----
    $lockState = 'unknown'
    $lres = Invoke-Capture $py @('-m', 'pigemma.paths', '--log-level', 'WARNING', 'lock', '--status')
    $lj = $null
    try { $lj = $lres.Text | ConvertFrom-Json } catch { $lj = $null }
    if ($null -eq $lj) {
        Write-Warn ("cannot read the bench lock status (python -m pigemma.paths lock --status exit {0}); the steps still take the lock themselves" -f $lres.Code)
    } else {
        $lockState = [string](Get-Prop $lj 'state' 'unknown')
        if ($lockState -eq 'held') {
            $h = Get-Prop $lj 'holder'
            $msg = ("bench lock {0} is held by pid {1} on {2} since {3}: {4}" -f (Get-Prop $lj 'path'), (Get-Prop $h 'pid'), (Get-Prop $h 'host'), (Get-Prop $h 'started'), (Get-Prop $h 'cmd'))
            if ($NoLock) {
                Write-Warn ($msg + ' (-NoLock: continuing; results may be distorted)')
            } elseif ($DryRun) {
                Write-Warn ($msg + ' (a real run would be REFUSED, exit 4)')
            } else {
                Exit-H2 4 ('REFUSED: ' + $msg + '; another model-loading job is running - retry when it finishes')
            }
        } else {
            Write-Log ("lock    {0} ({1})" -f $lockState, (Get-Prop $lj 'reason' ''))
        }
    }

    # ---- step: map (cached, header only) ----
    $mapArgs = @('-m', 'pigemma.gguf_map', 'build', $Model, '--out', $mapArg, '--reuse')
    if ($DryRun) {
        Write-Log ('[dry-run] ' + (Format-Command $py $mapArgs))
        Add-StepRecord 'map' 'dry-run' 0 $null (Format-Command $py $mapArgs)
    } else {
        $rc = Invoke-Step 'map' $py $mapArgs
        if ($rc -ne 0) { Exit-H2 (ConvertTo-ExitCode $rc) "step map failed (exit $rc)" }
    }
    $nLayer = $null; $nEmbd = $null; $nFf = $null
    if (Test-Path -LiteralPath $mapAbs -PathType Leaf) {
        $map = Read-JsonFile $mapAbs
        $nLayer = Get-Prop $map 'n_layer'; $nEmbd = Get-Prop $map 'n_embd'; $nFf = Get-Prop $map 'n_ff'
        $mapBase = [string](Get-Prop $map 'gguf_basename' '')
        if ($mapBase -and $mapBase -ne $modelBase) { Write-Warn "map $mapArg describes '$mapBase', not '$modelBase'" }
    } elseif (-not $DryRun) {
        Exit-H2 3 "missing: map $mapArg was not written"
    } else {
        Write-Warn "map $mapArg does not exist yet: layer range and disk projection not checked in the dry run"
    }
    if ($null -ne $nLayer) {
        $bad = @($layerList | Where-Object { $_ -ge [int]$nLayer })
        if ($bad.Count -gt 0) { Exit-H2 2 ("-Layers {0} out of range (the model has {1} layers: 0..{2})" -f ($bad -join ','), $nLayer, ([int]$nLayer - 1)) }
    }

    # ---- acts decision ----
    $acts = Test-Acts $actsAbs $layerList $T $modelBase $modelSize $corpusCtr
    $runActs = $true
    if ($SkipActs) {
        $runActs = $false
        if (-not $acts.Exists) { Exit-H2 3 "missing: -SkipActs but no activation dump at $actsArg ($($acts.Reason))" }
        if (-not $acts.Usable) { Write-Warn "-SkipActs: the existing dump does not match the request ($($acts.Reason)); the oracle uses what is there" }
        else { Write-Log "acts    $($acts.Reason) (-SkipActs)" }
    } elseif ($acts.Usable -and -not $RedoActs) {
        $runActs = $false
        Write-Log "acts    $($acts.Reason) -> capture skipped (use -RedoActs to re-capture)"
    } else {
        if ($RedoActs) { Write-Log "acts    -RedoActs: re-capturing ($($acts.Reason))" }
        else { Write-Log "acts    capture needed: $($acts.Reason)" }
    }

    # ---- disk projection (acts + decomposition npz, approximation of decompose's own estimate) ----
    $needActs = [int64]0
    if ($runActs -and $null -ne $nEmbd -and $null -ne $nFf) {
        $needActs = [int64]$T * (2 * [int64]$nEmbd + [int64]$nFf) * 2 * $layerList.Count + 4096
    }
    $needDecomp = [int64]0; $todo = 0
    if ($null -ne $nEmbd -and $null -ne $nFf) {
        foreach ($il in $layerList) {
            foreach ($m in $matList) {
                if ($m -eq 'down') { $nOut = [int64]$nEmbd; $nIn = [int64]$nFf } else { $nOut = [int64]$nFf; $nIn = [int64]$nEmbd }
                foreach ($b in $baseList) {
                    $npz = Join-Path $decompAbs ("L{0}.{1}.{2}.npz" -f $il, $m, $b)
                    if ((Test-Path -LiteralPath $npz -PathType Leaf) -and -not $Recompute) { continue }
                    $todo++
                    $gMax = [int64][math]::Min($G, $nIn)
                    $nGroups = [int64][math]::Ceiling($nIn / [double]$G)
                    $resItem = 2; if ($ResBits -eq 16) { $resItem = 4 }
                    $needDecomp += 2 * $nOut * $nIn + $nGroups * $nOut * $gMax * $resItem + 65536
                }
            }
        }
    }
    $freeOut = Get-FreeBytes $decompAbs
    Write-Log ("disk    projected new output: acts {0} + decomposition {1} ({2} new npz of {3}) = {4}; free {5}; keep >= {6} GB free" -f `
            (Format-Gb $needActs), (Format-Gb $needDecomp), $todo, $nCases, (Format-Gb ($needActs + $needDecomp)), (Format-Gb $freeOut), $MinFreeDiskGb.ToString($Inv))
    if ($null -ne $freeOut -and ($freeOut - $needActs - $needDecomp) -lt ($MinFreeDiskGb * 1e9)) {
        $msg = ("not enough disk: outputs need ~{0} but only {1} is free (keeping >= {2} GB free); choose fewer layers/bases or free space" -f (Format-Gb ($needActs + $needDecomp)), (Format-Gb $freeOut), $MinFreeDiskGb.ToString($Inv))
        if ($DryRun) { Write-Warn ($msg + ' (a real run would be REFUSED, exit 4)') } else { Exit-H2 4 ('REFUSED: ' + $msg) }
    }

    # ---- step: acts (Docker) ----
    $bash = $null
    if ($runActs) {
        $bash = Find-GitBash
        if ($null -eq $bash) {
            $msg = 'missing: Git Bash (bash.exe of Git for Windows) is needed for pc/docker_run.sh; set PIGEMMA_BASH'
            if ($DryRun) { Write-Warn $msg } else { Exit-H2 3 $msg }
        } else {
            Write-Log "bash    $bash"
        }
        $actsCtrRel = Get-RepoRelative $repo $actsAbs
        if ($null -eq $actsCtrRel -or $actsCtrRel -eq '') { Exit-H2 2 "-ActsDir must be a directory inside the repo (the container only sees /work): $actsAbs" }
        $stageAbs = $actsAbs.TrimEnd([char]92, [char]47) + '.partial'   # strip trailing \ and /
        $stageCtr = '/work/' + $actsCtrRel + '.partial'
        $modelRel = Get-RepoRelative $repo $modelPath
        $volumeArgs = @()
        if ($null -ne $modelRel -and $modelRel -ne '') {
            $modelCtr = '/work/' + $modelRel
        } else {
            $modelDir = (Split-Path -Parent $modelPath).Replace('\', '/')
            $volumeArgs = @('--volume', ($modelDir + ':/h2model:ro'))
            $modelCtr = '/h2model/' + $modelBase
            Write-Log "model is outside the repo: mounted read-only at /h2model"
        }

        # probe binary
        $probeAbs = Join-Path $repo $ProbeRel
        if (-not (Test-Path -LiteralPath $probeAbs -PathType Leaf)) {
            $buildArgs = @('pc/docker_run.sh', 'build-tools', '--targets', 'pigemma-probe')
            if ($DryRun) {
                $w = "probe binary $ProbeRel is missing"
                if ($BuildProbe) { $w += '; -BuildProbe would run: bash ' + ($buildArgs -join ' ') }
                else { $w += '; a real run exits 3 (build it with: bash pc/docker_run.sh build-tools, or pass -BuildProbe)' }
                Write-Warn $w
                Add-StepRecord 'build-probe' 'dry-run' 0 $null ('bash ' + ($buildArgs -join ' '))
            } elseif ($BuildProbe) {
                $rc = Invoke-Step 'build-probe' $bash $buildArgs ('bash ' + ($buildArgs -join ' '))
                if ($rc -ne 0) { Exit-H2 (ConvertTo-ExitCode $rc) "step build-probe failed (exit $rc)" }
                if (-not (Test-Path -LiteralPath $probeAbs -PathType Leaf)) { Exit-H2 3 "missing: $ProbeRel after build-tools" }
            } else {
                Exit-H2 3 "missing: probe binary $ProbeRel; build it with 'bash pc/docker_run.sh build-tools' (Git Bash) or pass -BuildProbe"
            }
        }

        # corpus: the default all.txt goes through python -m pigemma.quality corpus (--check first,
        # rebuild when missing/out of date); any other -Corpus file must exist inside the repo
        if (-not $corpusCtr) { Exit-H2 2 "-Corpus must be a file inside the repo (the container only sees /work): $corpusAbs" }
        $cchk = @{ Code = 0; Text = '' }
        if ($isDefaultCorpus) { $cchk = Invoke-Capture $py @('-m', 'pigemma.quality', 'corpus', '--check') }
        if (-not $isDefaultCorpus) {
            if (-not (Test-Path -LiteralPath $corpusAbs -PathType Leaf)) {
                if ($DryRun) { Write-Warn "corpus $corpusRepoRel does not exist (a real run exits 3)" } else { Exit-H2 3 "missing: -Corpus $corpusRepoRel" }
            } else {
                Write-Log "corpus  $corpusRepoRel (-Corpus)"
            }
            Add-StepRecord 'corpus' 'cached' 0 0 ''
        } elseif ($cchk.Code -eq 0) {
            Write-Log "corpus  $DefaultCorpusRel is up to date"
            Add-StepRecord 'corpus' 'cached' 0 0 (Format-Command $py @('-m', 'pigemma.quality', 'corpus', '--check'))
        } elseif ($DryRun) {
            Write-Warn "$DefaultCorpusRel is missing or out of date; a real run rebuilds it with: python -m pigemma.quality corpus"
            Add-StepRecord 'corpus' 'dry-run' 0 $null (Format-Command $py @('-m', 'pigemma.quality', 'corpus'))
        } else {
            if (Test-Path -LiteralPath (Join-Path $repo $DefaultCorpusRel)) { Write-Warn "$DefaultCorpusRel is out of date (calibration sources changed): rebuilding it" }
            $rc = Invoke-Step 'corpus' $py @('-m', 'pigemma.quality', 'corpus')
            if ($rc -ne 0) { Exit-H2 (ConvertTo-ExitCode $rc) "step corpus failed (exit $rc)" }
        }

        $inner = @()
        if (-not $NoLock) { $inner += @('python', '-m', 'pigemma.paths', 'lock', '--hold', '--min-free-gb', $minFreeGbStr, '--') }
        $inner += @($ProbeRel, '--mode', 'act-dump', '-m', $modelCtr, '-f', $corpusCtr,
            '--chunk-tokens', [string]$chunk, '--gen', [string]$genPer, '-nr', '-t', [string]$Threads,
            '-c', [string]$ctxEff, '-b', '512', '-ub', '128', '--layers', $layersCsv,
            '--kinds', ($ActKinds -join ','), '--tokens', [string]$T, '--act-out', $stageCtr)
        $dockerArgs = @('pc/docker_run.sh', '--no-stdin') + $volumeArgs + @('--') + $inner
        $display = 'bash ' + (Format-Command 'pc/docker_run.sh' (@('--no-stdin') + $volumeArgs + @('--') + $inner))
        if ($DryRun) {
            Write-Log ('[dry-run] ' + $display)
            if ($null -ne $bash) {
                Write-Log '[dry-run] docker command (pc/docker_run.sh --dry-run):'
                [void](Invoke-Native $bash (@('pc/docker_run.sh', '--no-stdin', '--dry-run') + $volumeArgs + @('--') + $inner))
            }
            Add-StepRecord 'acts' 'dry-run' 0 $null $display
        } else {
            # The probe writes into the staging dir <acts>.partial; an existing dump in <acts> is only
            # replaced after the new one finished and verified, so a failed capture never destroys it.
            Remove-ProbeFiles $stageAbs
            $rc = Invoke-Step 'acts' $bash $dockerArgs $display
            if ($rc -ne 0) {
                $hint = ''
                if ($rc -eq 4) { $hint = ' (refused: bench lock busy or < ' + $minFreeGbStr + ' GB RAM available in the Docker VM)' }
                Exit-H2 (ConvertTo-ExitCode $rc) "step acts (Docker act-dump) failed with exit $rc$hint; the previous dump in $actsArg (if any) is untouched; partial output (if any) in $($actsArg).partial"
            }
            $staged = Test-Acts $stageAbs $layerList $T $modelBase $modelSize $corpusCtr
            if (-not $staged.Usable) { Exit-H2 1 "step acts finished but the new dump in $($actsArg).partial is not usable: $($staged.Reason); the previous dump in $actsArg (if any) is untouched" }
            Remove-ProbeFiles $actsAbs
            if (-not (Test-Path -LiteralPath $actsAbs -PathType Container)) { New-Item -ItemType Directory -Force -Path $actsAbs | Out-Null }
            Get-ChildItem -LiteralPath $stageAbs -File | Move-Item -Destination $actsAbs -Force
            if (@(Get-ChildItem -LiteralPath $stageAbs -Force).Count -eq 0) { Remove-Item -LiteralPath $stageAbs -Force }
            $acts = Test-Acts $actsAbs $layerList $T $modelBase $modelSize $corpusCtr
            if (-not $acts.Usable) { Exit-H2 1 "the new dump could not be moved into place: $($acts.Reason)" }
            Write-Log "acts    $($acts.Reason)"
        }
    } else {
        Add-StepRecord 'acts' 'skipped' 0 $null ''
    }

    # ---- step: decompose ----
    $decArgs = @('-m', 'pigemma.h2.decompose', '--map', $mapArg, '--layers', $layersCsv, '--mats', $matsCsv,
        '--bases', $basesCsv, '--group', [string]$G, '--res-bits', [string]$ResBits, '--out', ($decompArg.TrimEnd('/') + '/'))
    if ($Recompute) { $decArgs += '--force' }
    if (-not $NoLock) { $decArgs += '--lock' }
    if ($DryRun) {
        $dry = @($decArgs | Where-Object { $_ -ne '--lock' }) + @('--dry-run')
        Write-Log ('[dry-run] ' + (Format-Command $py $decArgs))
        if (Test-Path -LiteralPath $mapAbs -PathType Leaf) {
            $rc = Invoke-Native $py $dry
            if ($rc -ne 0) { Exit-H2 (ConvertTo-ExitCode $rc) "decompose --dry-run failed (exit $rc)" }
        }
        Add-StepRecord 'decompose' 'dry-run' 0 $null (Format-Command $py $decArgs)
    } else {
        $rc = Invoke-Step 'decompose' $py $decArgs
        if ($rc -ne 0) { Exit-H2 (ConvertTo-ExitCode $rc) "step decompose failed (exit $rc)" }
    }

    # ---- step: oracle ----
    $orcArgs = @('-m', 'pigemma.h2.oracle', '--decomp', ($decompArg.TrimEnd('/') + '/'), '--acts', ($actsArg.TrimEnd('/') + '/'),
        '--tokens', [string]$T, '--eps', $epsCsv, '--out', $outArg, '--md', $mdArg,
        '--layers', $layersCsv, '--mats', $matsCsv, '--bases', $basesCsv, '--map', $mapArg)
    if (-not $NoLock) { $orcArgs += '--lock' }
    $verdict = $null
    $oracleCode = 0
    if ($DryRun) {
        Write-Log ('[dry-run] ' + (Format-Command $py $orcArgs))
        Add-StepRecord 'oracle' 'dry-run' 0 $null (Format-Command $py $orcArgs)
    } else {
        $t0 = [DateTime]::UtcNow
        $oracleCode = Invoke-Step 'oracle' $py $orcArgs
        $fresh = (Test-Path -LiteralPath $outAbs -PathType Leaf) -and ((Get-Item -LiteralPath $outAbs).LastWriteTimeUtc -ge $t0.AddSeconds(-2))
        if ($fresh) {
            try { $verdict = (Read-JsonFile $outAbs) } catch { Write-Warn "cannot parse $outArg : $($_.Exception.Message)" }
        }
        if ($oracleCode -ne 0 -and $null -eq $verdict) {
            Exit-H2 (ConvertTo-ExitCode $oracleCode) "step oracle failed (exit $oracleCode) and wrote no oracle.json"
        }
        if ($fresh -and -not $KeepOldEnvelopes) { Remove-SupersededEnvelopes (Split-Path -Parent $outAbs) $outArg $t0 }
    }

    $wall.Stop()
    return @{
        Code       = (ConvertTo-ExitCode $oracleCode)
        OracleCode = $oracleCode
        Doc        = $verdict
        OutAbs     = $outAbs
        OutArg     = $outArg
        MdArg      = $mdArg
        Repo       = $repo
        Started    = $startedAt
        Wall       = $wall.Elapsed.TotalSeconds
        Config     = [ordered]@{
            model = $Model; model_path = $modelPath; corpus = $Corpus; layers = $layerList; mats = $matList; bases = $baseList
            group = $G; res_bits = $ResBits; tokens = $T; eps = $epsList; quick = [bool]$Quick
            chunk_tokens = $chunk; gen = $genPer; ctx = $ctxEff; threads = $Threads
            acts_dir = $actsArg; decomp_dir = $decompArg; map = $mapArg; out = $outArg; md = $mdArg
            lock = (-not $NoLock); min_free_gb = $MinFreeGb; acts_captured = [bool]$runActs
        }
    }
}

function Write-StepTable {
    if ($script:Steps.Count -eq 0) { return }
    Write-Log 'wall time per step:'
    foreach ($s in $script:Steps) {
        $code = 'n/a'
        if ($null -ne $s.exit) { $code = [string]$s.exit }
        Write-Host ('  {0,-12} {1,-8} {2,10}  exit {3}' -f $s.step, $s.status, (Format-Duration ([double]$s.seconds)), $code)
    }
}

function Write-Verdict($Result) {
    $doc = $Result.Doc
    if ($null -eq $doc) {
        Write-Log 'E7 VERDICT: none (oracle.json was not produced)'
        return $null
    }
    $s = Get-Prop $doc 'summary'
    $v = [string](Get-Prop $s 'verdict' 'n/a')
    $pipe = [string](Get-Prop $s 'pipeline' 'UNKNOWN')
    $imp = Get-Prop $s 'implied_bytes_per_token'
    $reasons = @(Get-Prop $s 'pipeline_reasons' @()) -join '; '
    $line = ("E7 VERDICT: {0} (best base {1}; pipeline {2}) | frac@3% median {3} | cert eff median {4} | implied {5}/token (PROJECTED) | {6} s/token @850 MB/s (PROJECTED) | {7} tok/s @10 GB/s (PROJECTED)" -f `
            $v, (Get-Prop $s 'best_base' 'n/a'), $pipe, (Format-Num (Get-Prop $s 'frac_0_03_median')), (Format-Num (Get-Prop $s 'cert_eff_median')),
        (Format-Gb (Get-Prop $imp 'total')), (Format-Num (Get-Prop $s 's_per_token_nvme_850mbps') '0.0'), (Format-Num (Get-Prop $s 'tok_s_in_ram_10gbps') '0.00'))
    if ($pipe -eq 'INVALID') {
        $line = "E7 VERDICT: VOID (pipeline INVALID: $reasons) - raw " + $line.Substring('E7 VERDICT: '.Length)
    }
    foreach ($a in @(Get-Prop $s 'always_reported' @())) { Write-Log ('always reported: ' + [string]$a) }
    if ($pipe -eq 'UNKNOWN') { Write-Warn "pipeline validity UNKNOWN: $reasons" }
    Write-Log ("outputs: {0}, {1} (+ E7 envelopes next to oracle.json)" -f $Result.OutArg, $Result.MdArg)
    Write-Host ''
    Write-Host ('h2_run: ' + $line)
    return $line
}

# ------------------------------------------------------------------------------------------
# entry
# ------------------------------------------------------------------------------------------
if ($Help) {
    Write-Host $UsageText
    exit 0
}
if ($null -ne $UnknownArgs -and $UnknownArgs.Count -gt 0) {
    [Console]::Error.WriteLine('h2_run: ERROR (exit 2): unknown parameter(s) or argument(s): ' + ($UnknownArgs -join ' ') + ' (see -Help)')
    exit 2
}

$script:BoundParams = @{} + $PSBoundParameters
[void]$script:BoundParams.Remove('UnknownArgs')
$exitCode = 1
$prevLocation = Get-Location
$prevOutEnc = $null
$envBackup = @{}
foreach ($k in @('PYTHONUTF8', 'PYTHONIOENCODING')) { $envBackup[$k] = [Environment]::GetEnvironmentVariable($k, 'Process') }
try {
    try { $prevOutEnc = [Console]::OutputEncoding; [Console]::OutputEncoding = New-Object System.Text.UTF8Encoding($false) } catch { $prevOutEnc = $null }
    $env:PYTHONUTF8 = '1'
    $env:PYTHONIOENCODING = 'utf-8'
    $result = Invoke-H2Run
    Write-StepTable
    if ($DryRun) {
        Write-Log 'dry run complete: nothing was loaded or written'
        $exitCode = 0
    } else {
        $line = Write-Verdict $result
        $exitCode = $result.Code
        try {
            $rec = [ordered]@{
                record      = 'pigemma.h2_run'
                driver      = 'pc/h2_run.ps1'
                version     = $H2RunVersion
                started_at  = $result.Started.ToString('yyyy-MM-ddTHH:mm:ssZ', $Inv)
                finished_at = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ', $Inv)
                wall_s      = [math]::Round($result.Wall, 1)
                host        = [Environment]::MachineName
                argv        = @($script:BoundParams.GetEnumerator() | ForEach-Object { '-' + $_.Key + ' ' + ((@($_.Value) | ForEach-Object { [string]$_ }) -join ',') })
                config      = $result.Config
                steps       = $script:Steps.ToArray()   # not @(List): PS 5.1 ConvertTo-Json fails on a List[object]
                exit_code   = $exitCode
                verdict     = $line
            }
            $recPath = Join-Path (Split-Path -Parent $result.OutAbs) 'h2_run.json'
            Write-JsonFile $recPath $rec
            $recShow = Get-RepoRelative $result.Repo $recPath
            if (-not $recShow) { $recShow = $recPath }
            Write-Log ('run record: ' + $recShow)
        } catch {
            Write-Warn ('cannot write the run record: ' + $_.Exception.Message)
        }
        if ($result.OracleCode -ne 0) { Write-Err ("oracle exited {0}; see its messages above" -f $result.OracleCode) }
    }
} catch {
    $ex = $_.Exception
    if ($null -ne $ex -and $ex.Data.Contains('h2_exit')) {
        $exitCode = [int]$ex.Data['h2_exit']
        Write-StepTable
        if ($ex.Message.StartsWith('REFUSED')) { Write-Err $ex.Message }
        else { Write-Err ('ERROR (exit ' + $exitCode + '): ' + $ex.Message) }
    } else {
        $exitCode = 1
        Write-StepTable
        Write-Err ('ERROR (exit 1): unexpected ' + $_.Exception.GetType().Name + ': ' + $_.Exception.Message + ' at ' + $_.InvocationInfo.PositionMessage)
    }
} finally {
    if ($null -ne $prevOutEnc) { try { [Console]::OutputEncoding = $prevOutEnc } catch { } }
    foreach ($k in $envBackup.Keys) { [Environment]::SetEnvironmentVariable($k, $envBackup[$k], 'Process') }
    Set-Location -LiteralPath $prevLocation.Path
}
exit $exitCode
