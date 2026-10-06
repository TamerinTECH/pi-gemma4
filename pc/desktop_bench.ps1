<#
.SYNOPSIS
    Desktop driver for E0 (llama-bench matrix + repack gate), E1 (speculative decoding through
    llama-server) and E2 (quality: KLD + Q&A) with the native Windows llama.cpp binaries.
    Resumable and time-boxed.

.DESCRIPTION
    Runs, strictly one model job at a time, the desktop experiments of docs/EXPERIMENT_PLAN.md
    through the Python wrappers (every model-loading step holds results/.bench.lock on the Python
    side; this script only sequences them):

      bench   (E0) membw (Docker, D5) -> results/<date>/membw_desktop.json;
                   python -m pigemma.bench matrix --model <id> --threads 4,6,12 --pp 128 --tg 32 -r 3
                       --variants default,norepack,hybrid_negctrl,nommap     (31B: -t 6,12, --tg 16, -r 2)
                   python -m pigemma.bench completion --model <id> -n 64 --threads 6 --variant <v>
                       --prompt-file data/calibration/en_tech_moe_streaming.txt   (per variant)
                   then evaluates the E0 repack gate and the speed band -> results/<date>/E0_GATE.md
      spec    (E1) python -m pigemma.bench server-spec --model <id> --variant plain,mtp-explicit --rounds 2
                       (interleaved A,B,A,B per docs/design/FACTS.md, both models), then one round each of
                       26B: mtp, dflash, ngram (+ dflash-explicit when mtp or dflash never drafted); 31B: mtp
                   -> results/<date>/E1_*.json and E1_SUMMARY.md (speedup vs plain, accept_rate,
                      identical_to_plain, spec_type_effective, one-word verdict per variant)
      quality (E2) python -m pigemma.quality corpus; kld-base (6 x 512, disk-gated by quality.py);
                   kld --variant norepack / hybrid_negctrl; qa (plain, plus every E1 variant that was
                   spec_active); with -AllowDownload: one lower quant per model (26b-ud-q3_k_m,
                   31b-ud-iq3_xxs) downloaded via huggingface_hub, kld + qa, deleted afterwards
      always  python -m pigemma.results summarize results/<date> --md docs/results/DESKTOP_<date>.md

    Resumable: a step is skipped when its result already exists (matching envelope in
    results/<date>/, valid reference logits, QA file with the current Q&A set) unless -Force.
    A step that is re-run renames its earlier envelopes to <name>.json.<UTC>.superseded first (kept
    on disk, ignored by results summarize). Rerun the same command to resume after an interruption,
    a refusal or the time box.

    Time-boxed: -MaxMinutes N stops starting new steps after N minutes (exit 5; a running step is
    allowed to finish); every child gets --timeout -StepTimeoutMin (llama-bench, llama-completion,
    llama-perplexity) or --request-timeout -RequestTimeoutS (llama-server requests).

    -DryRun prints every command (and, indented with '|', the exact llama.cpp command lines from
    the Python tools' own --dry-run) and writes nothing.

.PARAMETER Stage
    bench, spec, quality (comma list; also E0/E1/E2 or all). Required. Run order is always
    bench -> spec -> quality.
.PARAMETER Models
    26b, 31b (default both), a registry/catalog id (26b-q4, 31b-q4, ...) or a GGUF path.
.PARAMETER Threads
    llama-bench thread list for every model (default 26B 4,6,12; 31B 6,12; -Quick 6).
.PARAMETER Quick
    Smoke-sized run into results/<date>-quick/: matrix 6 threads -r 1; E1 core only, 1 round,
    2 prompts, 64 tokens; KLD 2 chunks; QA 6 items (2 EN, 2 HE, 2 numeric); no lower quants.
.PARAMETER Force
    Re-run steps whose results exist (old envelopes are superseded; kld-base gets --force).
.PARAMETER AllowDownload
    E2: measure one lower quant per model (download when absent and free disk >= max(1.3 x its size,
    size + 4 GB)); a file downloaded by this run is deleted once its kld and qa steps finished
    (-KeepDownloads keeps it).
.PARAMETER DryRun
    Print the plan and every command; run nothing that loads a model; write nothing.
.PARAMETER Date
    Results date (yyyy-MM-dd, default today): results/<date>/ and docs/results/DESKTOP_<date>.md.
    Pass an earlier date to resume that day's run.
.PARAMETER OutDir
    Results directory instead of results/<date>/ (envelopes, E0_GATE.md, E1_SUMMARY.md, raw/).
.PARAMETER QualityDir
    Directory for the reference logits and QA files (default results/quality/; -Quick: <OutDir>/quality/).
.PARAMETER SummaryMd
    Rendered summary instead of docs/results/DESKTOP_<date>.md.
.PARAMETER MaxMinutes
    Time box: no new step starts after N minutes (0 = none). Exit 5; rerun the same command to resume.
.PARAMETER StepTimeoutMin
    --timeout for each llama-bench / llama-completion / llama-perplexity child (default 60).
.PARAMETER RequestTimeoutS
    --request-timeout for each llama-server request of E1 and the Q&A (default 900).
.PARAMETER RunThreads
    Threads of the completion, server, perplexity and Q&A runs (default 6).
.PARAMETER SpecRounds
    Rounds of the interleaved plain / mtp-explicit E1 core (default 2; -Quick 1).
.PARAMETER SpecTokens
    Tokens per E1 prompt (default 200; -Quick 64; 0 = each prompt's max_tokens).
.PARAMETER SpecPrompts
    Use only the first N prompts of data/prompts/gen_prompts.json (default all 8; -Quick 2).
.PARAMETER DraftNMax
    --spec-draft-n-max for the speculative variants (default: llama.cpp's own).
.PARAMETER KldChunks
    KLD reference chunks of 512 tokens (default 6; -Quick 2).
.PARAMETER MinFreeGB
    Bench-lock free-RAM gate passed to the Python tools (default 20; env PIGEMMA_MIN_FREE_GB wins).
.PARAMETER IdleCheckS
    Idle-machine check before every timed child (bench.py --idle-check-s, default 10; 0 disables).
    With -RequireIdle a busy machine is refused (exit 4); -IdleWaitS waits for an idle window.
.PARAMETER LowerQuant26
    Catalog id of the 26B lower quant measured with -AllowDownload (default 26b-ud-q3_k_m).
.PARAMETER LowerQuant31
    Catalog id of the 31B lower quant measured with -AllowDownload (default 31b-ud-iq3_xxs).
.PARAMETER QaHard
    Also run data/prompts/qa_hard.json (plain) into <QualityDir>/hard/.
.PARAMETER NoMembw
    Skip the Docker membw measurement (D5).

.EXAMPLE
    .\pc\desktop_bench.ps1 -Stage bench -Models 26b,31b -DryRun
.EXAMPLE
    .\pc\desktop_bench.ps1 -Stage bench,spec,quality -Models 26b -MaxMinutes 120
.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File pc\desktop_bench.ps1 -Stage quality -Models 26b -AllowDownload

.NOTES
    pc/desktop_bench.ps1 1.0.0 - Windows PowerShell 5.1 (also runs under PowerShell 7).
    From cmd / Git Bash use "powershell -File ..." (with -Command the exit code is lost).

    Exit codes (worst wins, in this order):
      4  refused: a step exited 4 (bench lock busy, < -MinFreeGB GB RAM, disk gate, busy machine with
         -RequireIdle, port in use) or the lock was held at start; the run stops there
      3  missing input: model / binary / python / prompt file (other models still run)
      1  a step failed (its failed envelope is still written) or an internal error
      6  every step ran but the E0 repack gate (or its negative control) is FAIL / INCOMPLETE for a
         model of this run: stop all streaming work (E5/E6/Pi) until the flags are fixed
      5  the -MaxMinutes time box stopped the run before all steps ran (rerun to resume)
      2  usage error
      0  everything requested ran or was already present (speed band, KLD and QA thresholds are
         results written to E0_GATE.md / the envelopes, not errors)

    Decisions (also listed in the run header and the design notes of the manifest entry):
      * E1 follows the 2026-10-05 FACTS update: MTP drafts only with an explicit --spec-type, so the
        core of E1 is plain vs mtp-explicit interleaved for -SpecRounds rounds (default 2) on both
        models; the manifest variants (mtp, dflash, ngram, dflash-explicit) follow with one round.
      * The optional lower quant for the 31B is 31b-ud-iq3_xxs (the quant the real Pi runs,
        EXPERIMENT_PLAN D2), not the manifest's UD-IQ2_XXS; override with -LowerQuant31.
      * QA files stay in results/quality/ (EXPERIMENT_PLAN E2 'Writes'); the summary step adds the
        ones of this run's models to the results/<date> inputs.
      * A failed membw step (Docker / Git Bash missing) is a warning, not a failure: D5 is an addition
        to E0 and the native E0 measurements do not depend on it.
      * E0_GATE.md is regenerated from every E0 envelope in the results dir on each bench run, so it
        stays complete when models are run in separate invocations.
#>
[CmdletBinding(PositionalBinding = $false)]
param(
    [string[]]$Stage,
    [string[]]$Models = @('26b', '31b'),
    [string[]]$Threads,
    [switch]$Quick,
    [switch]$Force,
    [switch]$AllowDownload,
    [switch]$DryRun,
    [string]$Date,
    [string]$OutDir,
    [string]$QualityDir,
    [string]$SummaryMd,
    [int]$RunThreads = 6,
    [int]$CompletionTokens = 64,
    [int]$Port = 18080,
    [int]$SpecRounds = 2,
    [int]$SpecTokens = 200,
    [int]$SpecPrompts = 0,
    [int]$DraftNMax = 0,
    [int]$QaTokens = 160,
    [int]$KldChunks = 6,
    [double]$MaxMinutes = 0,
    [double]$StepTimeoutMin = 60,
    [int]$RequestTimeoutS = 900,
    [double]$MinFreeGB = 20,
    [int]$IdleCheckS = 10,
    [double]$IdleMaxPct = 5,
    [int]$IdleWaitS = 0,
    [switch]$RequireIdle,
    [string]$LowerQuant26 = '26b-ud-q3_k_m',
    [string]$LowerQuant31 = '31b-ud-iq3_xxs',
    [switch]$KeepDownloads,
    [switch]$QaHard,
    [switch]$NoMembw,
    [switch]$NoSummary,
    [string]$Python,
    [switch]$Help,
    [switch]$Version,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$Rest
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0

# ------------------------------------------------------------------------------------------------
# Constants and script state
# ------------------------------------------------------------------------------------------------
$script:ToolName = 'pc/desktop_bench.ps1'
$script:ToolVersion = '1.0.0'
$script:Inv = [Globalization.CultureInfo]::InvariantCulture
$script:Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$script:Bound = $PSBoundParameters

$script:EXIT_OK = 0
$script:EXIT_ERROR = 1
$script:EXIT_USAGE = 2
$script:EXIT_MISSING = 3
$script:EXIT_REFUSED = 4
$script:EXIT_TIMEBOX = 5
$script:EXIT_GATE = 6

$script:E0_VARIANTS = @('default', 'norepack', 'hybrid_negctrl', 'nommap')
$script:E0_PP = 128
$script:E0_PROMPT = 'data/calibration/en_tech_moe_streaming.txt'
$script:GEN_PROMPTS = 'data/prompts/gen_prompts.json'
$script:QA_SET = 'data/prompts/qa_set.json'
$script:QA_HARD = 'data/prompts/qa_hard.json'
$script:CORPUS = 'data/calibration/all.txt'
$script:KLD_CTX = 512
$script:QA_CTX = 4096
$script:SPEC_CTX = 4096
$script:QUICK_QA_IDS = 'en-01,en-02,he-01,he-02,num-01,num-02'
$script:MEMBW_BIN = 'tools/build-linux-x86/bin/membw'
$script:MEMBW_ARGS = @('--size-mb', '512', '--threads', '1,2,4,6,12', '--json')
$script:DOWNLOAD_DISK_FACTOR = 1.3       # manifest: download only with >= 1.3 x the file size free
$script:DOWNLOAD_KEEP_FREE = 4e9          # added: and >= 4 GB left after the download (logits, results)
# E0 pre-registered thresholds (docs/EXPERIMENT_PLAN.md#e0-desktop-baseline-and-repack-gate)
$script:GATE_MAPPED_FRACTION = 0.95
$script:GATE_PRIVATE_PEAK_MB = 1500.0      # "private peak < 1.5 GB", MB = 1e6 bytes
$script:SPEED_BAND = 0.20
$script:SPEED_REFS = @{
    '26b-q4' = @(@{ T = 4; Ref = 15.18 }, @{ T = 6; Ref = 13.5 })
    '31b-q4' = @(@{ T = 6; Ref = 2.44 })
}
# E1 rule (docs/EXPERIMENT_PLAN.md#e1-speculative-decoding-via-llama-server)
$script:E1_USEFUL_SPEEDUP = 1.2
$script:E1_USEFUL_PROMPT_FRACTION = 0.75   # ">= 6 of 8 prompts"
$script:E1_IDENTICAL_MIN = 0.95

$script:RepoRoot = $null
$script:OrigCwd = $null
$script:PyExe = $null
$script:PyPre = @()
$script:PyDisplay = 'python'
$script:Catalog = $null
$script:ModelInfos = @()
$script:Opt = @{}
$script:StepLog = New-Object System.Collections.ArrayList
$script:EnvCache = @{}
$script:LogPath = $null
$script:StartTime = Get-Date
$script:Deadline = $null
$script:Stopped = $false
$script:StopReason = ''
$script:TimeBoxHit = $false
$script:GateFailed = $false
$script:UsageMissing = $false
$script:CorpusSha = $null
$script:HelperPath = $null
$script:FinalCode = 0

# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------
function Write-Line([string]$Text = '') { Write-Host $Text }

function Add-Log([string]$Text) {
    if (-not $script:LogPath) { return }
    $ts = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ', $script:Inv)
    try { [IO.File]::AppendAllText($script:LogPath, "$ts $Text`n", $script:Utf8NoBom) } catch { }
}

function Write-Note([string]$Text) { Write-Host "desktop_bench: $Text"; Add-Log $Text }
function Write-Warn([string]$Text) { Write-Host "desktop_bench: WARNING: $Text"; Add-Log "WARNING: $Text" }

function Fmt([object]$X, [int]$Digits = 2) {
    if ($null -eq $X) { return 'n/a' }
    try { return ([double]$X).ToString('F' + $Digits, $script:Inv) } catch { return [string]$X }
}

function Fmt-GB([object]$Bytes) {
    if ($null -eq $Bytes) { return 'n/a' }
    return (Fmt ([double]$Bytes / 1e9) 2) + ' GB'
}

function Get-Prop([object]$Obj, [string]$Path) {
    $cur = $Obj
    foreach ($part in $Path.Split('.')) {
        if ($null -eq $cur) { return $null }
        if ($cur -is [System.Collections.IDictionary]) {
            if ($cur.Contains($part)) { $cur = $cur[$part] } else { return $null }
            continue
        }
        if ($cur -is [string] -or $cur -is [ValueType]) { return $null }
        $p = $cur.PSObject.Properties[$part]
        if ($null -eq $p) { return $null }
        $cur = $p.Value
    }
    return $cur
}

function Get-PropNames([object]$Obj) {
    if ($null -eq $Obj) { return @() }
    if ($Obj -is [System.Collections.IDictionary]) { return @($Obj.Keys) }
    return @($Obj.PSObject.Properties | ForEach-Object { $_.Name })
}

function Split-List([string[]]$Values) {
    $out = New-Object System.Collections.ArrayList
    foreach ($v in @($Values)) {
        if ($null -eq $v) { continue }
        foreach ($p in ([string]$v -split '[,\s]+')) {
            if ($p.Trim() -ne '') { [void]$out.Add($p.Trim()) }
        }
    }
    return $out.ToArray()
}

function Format-Arg([string]$A) {
    if ($A -eq '') { return '""' }
    if ($A -match '[\s"&|<>^()]') { return '"' + ($A -replace '"', '\"') + '"' }
    return $A
}

function Format-Cmd([string[]]$PyArgs) {
    $parts = @($script:PyDisplay) + @($script:PyPre) + @($PyArgs)
    return (($parts | ForEach-Object { Format-Arg ([string]$_) }) -join ' ')
}

function Get-FullPathFrom([string]$Base, [string]$P) {
    if ([IO.Path]::IsPathRooted($P)) { return [IO.Path]::GetFullPath($P) }
    return [IO.Path]::GetFullPath((Join-Path $Base $P))
}

function Get-RelArg([string]$Abs) {
    # repo-relative with forward slashes when inside the repo, else the absolute path
    $root = $script:RepoRoot.TrimEnd('\', '/') + '\'
    $full = [IO.Path]::GetFullPath($Abs)
    if ($full.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) {
        return $full.Substring($root.Length).Replace('\', '/')
    }
    return $full
}

function Get-DirArg([string]$Abs) { return (Get-RelArg $Abs).TrimEnd('/', '\') + '/' }

function Get-PathKey([string]$P) {
    if ([string]::IsNullOrEmpty($P)) { return '' }
    $q = $P.Replace('/', '\')
    if (-not [IO.Path]::IsPathRooted($q)) { $q = Join-Path $script:RepoRoot $q }
    return ([IO.Path]::GetFullPath($q)).ToLowerInvariant()
}

function Write-TextFile([string]$Path, [string[]]$Lines) {
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path -LiteralPath $dir)) { $null = New-Item -ItemType Directory -Path $dir -Force }
    $tmp = "$Path.tmp$PID"
    [IO.File]::WriteAllText($tmp, (($Lines -join "`n") + "`n"), $script:Utf8NoBom)
    Move-Item -LiteralPath $tmp -Destination $Path -Force
}

function Read-JsonFile([string]$Path) {
    try {
        $raw = [IO.File]::ReadAllText($Path, [Text.Encoding]::UTF8)
        if ($raw.Length -gt 0 -and $raw[0] -eq [char]0xFEFF) { $raw = $raw.Substring(1) }
        return ($raw | ConvertFrom-Json)
    } catch {
        return $null
    }
}

function Get-FileSha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    return (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant()
}

function Get-FreeBytes([string]$Path) {
    try {
        $p = $Path
        while ($p -and -not (Test-Path -LiteralPath $p)) { $p = Split-Path -Parent $p }
        if (-not $p) { return $null }
        $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($p))
        return [int64](New-Object System.IO.DriveInfo($root)).AvailableFreeSpace
    } catch {
        return $null
    }
}

function Get-RamAvailBytes {
    try {
        $os = Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop
        return [int64]$os.FreePhysicalMemory * 1024
    } catch {
        return $null
    }
}

function Show-FreeDisk {
    $seen = @{}
    $parts = New-Object System.Collections.ArrayList
    $cands = @($script:RepoRoot, $script:Opt.ResultsDirAbs, $script:Opt.QualityDirAbs, $script:Opt.ModelsDirAbs)
    if ($env:OS -eq 'Windows_NT') { $cands += 'C:\' }
    foreach ($c in $cands) {
        if (-not $c) { continue }
        try { $root = [IO.Path]::GetPathRoot([IO.Path]::GetFullPath($c)) } catch { continue }
        if ($seen.ContainsKey($root.ToLowerInvariant())) { continue }
        $seen[$root.ToLowerInvariant()] = $true
        $free = Get-FreeBytes $c
        $label = $root.TrimEnd('\')
        if ($label -eq 'C:' -and $env:OS -eq 'Windows_NT') { $label = 'C: (Docker VM disk)' }
        [void]$parts.Add("$label $(Fmt-GB $free)")
    }
    $ram = Get-RamAvailBytes
    Write-Line ("   free disk: " + ($parts -join ', ') + "; RAM available: " + (Fmt-GB $ram))
    Add-Log ("free disk: " + ($parts -join ', ') + "; RAM available: " + (Fmt-GB $ram))
}

function Format-Elapsed {
    $s = ((Get-Date) - $script:StartTime).TotalSeconds
    return ('{0}m{1:00}s' -f [int][Math]::Floor($s / 60), [int]($s % 60))
}

# ------------------------------------------------------------------------------------------------
# Python
# ------------------------------------------------------------------------------------------------
function Find-Python {
    $cands = New-Object System.Collections.ArrayList
    if ($Python) { [void]$cands.Add(@{ Exe = $Python; Pre = @(); Display = $Python }) }
    elseif ($env:PIGEMMA_PYTHON) { [void]$cands.Add(@{ Exe = $env:PIGEMMA_PYTHON; Pre = @(); Display = $env:PIGEMMA_PYTHON }) }
    else {
        [void]$cands.Add(@{ Exe = 'python'; Pre = @(); Display = 'python' })
        [void]$cands.Add(@{ Exe = 'python3'; Pre = @(); Display = 'python3' })
        [void]$cands.Add(@{ Exe = 'py'; Pre = @('-3'); Display = 'py' })
    }
    foreach ($c in $cands) {
        $cmd = Get-Command $c.Exe -ErrorAction SilentlyContinue
        if ($null -eq $cmd) { continue }
        $old = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            $null = & $c.Exe @($c.Pre) '-c' 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'
            $rc = $LASTEXITCODE
        } catch {
            $rc = 1
        } finally {
            $ErrorActionPreference = $old
        }
        if ($rc -eq 0) {
            $script:PyExe = $c.Exe
            $script:PyPre = @($c.Pre)
            $script:PyDisplay = $c.Display
            return $true
        }
    }
    return $false
}

function Invoke-Py([string[]]$PyArgs) {
    $all = @($script:PyPre) + @($PyArgs)
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $lines = $null
    $rc = 1
    try {
        $lines = & $script:PyExe @all
        $rc = $LASTEXITCODE
    } catch {
        Write-Warn "cannot run $($script:PyDisplay): $_"
        $rc = 127
    } finally {
        $ErrorActionPreference = $old
    }
    if ($null -eq $rc) { $rc = 1 }
    $out = @()
    if ($null -ne $lines) { $out = @(@($lines) | ForEach-Object { [string]$_ }) }
    return [pscustomobject]@{ Rc = [int]$rc; Lines = $out }
}

# ------------------------------------------------------------------------------------------------
# Step bookkeeping
# ------------------------------------------------------------------------------------------------
function Add-StepRecord([string]$StageName, [string]$Name, [string]$Status, [object]$Rc, [double]$Seconds,
                        [string]$Note, [string[]]$Lines = @()) {
    $rec = [pscustomobject]@{
        Stage = $StageName; Name = $Name; Status = $Status; Rc = $Rc; Seconds = $Seconds; Note = $Note
        Lines = @($Lines)
    }
    [void]$script:StepLog.Add($rec)
    Add-Log ("step [{0}] {1}: {2} rc={3} {4}s {5}" -f $StageName, $Name, $Status, $Rc, (Fmt $Seconds 1), $Note)
    return $rec
}

function Move-Superseded([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return }
    $ts = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ', $script:Inv)
    $dest = "$Path.$ts.superseded"
    Move-Item -LiteralPath $Path -Destination $dest -Force
    $script:EnvCache.Remove($Path)
    Write-Line "   superseded: $(Get-RelArg $Path) -> $([IO.Path]::GetFileName($dest))"
    Add-Log "superseded $Path -> $dest"
}

function Copy-Superseded([string]$Path) {
    $ts = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ', $script:Inv)
    $dest = "$Path.$ts.superseded"
    Copy-Item -LiteralPath $Path -Destination $dest -Force
    Write-Line "   kept a copy of the earlier $(Get-RelArg $Path) as $([IO.Path]::GetFileName($dest))"
    Add-Log "copied $Path -> $dest"
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$StageName,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string[]]$PyArgs,
        [bool]$Done = $false,
        [string]$DoneWhy = 'result present',
        [string[]]$Supersede = @(),
        [string[]]$BackupFirst = @(),
        [switch]$Heavy,
        [switch]$NoInnerDryRun,
        [switch]$Soft,
        [switch]$IgnoreTimeBox,
        [string]$Note = ''
    )
    if ($script:Stopped) {
        return (Add-StepRecord $StageName $Name 'not-run' $null 0 $script:StopReason)
    }
    if ($Done -and -not $Force) {
        Write-Line ''
        Write-Line "== [$StageName] $Name"
        Write-Line "   skip: $DoneWhy (-Force re-runs it)"
        return (Add-StepRecord $StageName $Name 'skipped' $null 0 $DoneWhy)
    }
    if (-not $DryRun -and -not $IgnoreTimeBox -and $null -ne $script:Deadline -and (Get-Date) -ge $script:Deadline) {
        $script:Stopped = $true
        $script:TimeBoxHit = $true
        $script:StopReason = "time box of $(Fmt $MaxMinutes 2) min reached; rerun the same command to resume"
        Write-Line ''
        Write-Line "== [$StageName] $Name"
        Write-Line "   NOT STARTED: $($script:StopReason)"
        return (Add-StepRecord $StageName $Name 'timebox' $null 0 $script:StopReason)
    }
    $cmdText = Format-Cmd $PyArgs
    Write-Line ''
    Write-Line "== [$StageName] $Name   (elapsed $(Format-Elapsed))"
    if ($Note) { Write-Line "   note: $Note" }
    if ($Heavy) { Show-FreeDisk }
    Write-Line "   $cmdText"
    $supList = @($Supersede | Where-Object { $_ })
    $bakList = @($BackupFirst | Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) })
    if ($DryRun) {
        if ($supList.Count -gt 0) { Write-Line "   (would supersede $($supList.Count) earlier result file(s) after a successful re-run)" }
        if ($bakList.Count -gt 0) { Write-Line "   (would keep a copy of $($bakList.Count) earlier result file(s) as *.superseded)" }
        if (-not $NoInnerDryRun) {
            $r = Invoke-Py (@($PyArgs) + @('--dry-run'))
            foreach ($l in $r.Lines) { Write-Line "     | $l" }
            if ($r.Rc -ne 0) { Write-Line "     | (the tool's --dry-run exited $($r.Rc))" }
        }
        return (Add-StepRecord $StageName $Name 'dry-run' $null 0 $Note)
    }
    foreach ($f in $bakList) { Copy-Superseded $f }
    Add-Log "run: $cmdText"
    $t0 = Get-Date
    $r = Invoke-Py $PyArgs
    $secs = ((Get-Date) - $t0).TotalSeconds
    foreach ($l in $r.Lines) { Write-Line "   -> $l"; Add-Log "   -> $l" }
    $status = 'failed'
    if ($r.Rc -eq 0) { $status = 'done' }
    elseif ($r.Rc -eq 3) { $status = 'missing' }
    elseif ($r.Rc -eq 4) { $status = 'refused' }
    if ($Soft -and $status -ne 'done') { $status = 'warn' }
    if ($status -eq 'refused') {
        $script:Stopped = $true
        $script:StopReason = "stopped after the refusal in '$Name' (exit 4: bench lock busy, low RAM or disk, busy machine or port in use)"
    } elseif ($r.Rc -eq 130) {
        $script:Stopped = $true
        $script:StopReason = "stopped: '$Name' was interrupted"
    }
    Write-Line ("   {0} (exit {1}, {2} s)" -f $status.ToUpperInvariant(), $r.Rc, (Fmt $secs 1))
    if ($supList.Count -gt 0) {
        if ($r.Rc -eq 0) { foreach ($f in $supList) { Move-Superseded $f } }
        else { Write-Line "   kept the $($supList.Count) earlier result file(s) of this step: the re-run did not succeed" }
    }
    return (Add-StepRecord $StageName $Name $status $r.Rc $secs $Note $r.Lines)
}

function Add-Skip([string]$StageName, [string]$Name, [string]$Status, [string]$Why) {
    Write-Line ''
    Write-Line "== [$StageName] $Name"
    Write-Line "   $($Status.ToUpperInvariant()): $Why"
    return (Add-StepRecord $StageName $Name $Status $null 0 $Why)
}

# ------------------------------------------------------------------------------------------------
# Common python arguments
# ------------------------------------------------------------------------------------------------
function Get-TimeoutArgs { return @('--timeout', ([int][Math]::Round($StepTimeoutMin * 60)).ToString($script:Inv)) }
function Get-LockArgs { return @('--min-free-gb', (Fmt $MinFreeGB 1)) }
function Get-IdleArgs {
    $a = @('--idle-check-s', $IdleCheckS.ToString($script:Inv), '--idle-max-pct', (Fmt $IdleMaxPct 1))
    if ($IdleWaitS -gt 0) { $a += @('--idle-wait-s', $IdleWaitS.ToString($script:Inv)) }
    if ($RequireIdle) { $a += '--require-idle' }
    return $a
}

# ------------------------------------------------------------------------------------------------
# Catalog and models
# ------------------------------------------------------------------------------------------------
function Get-Catalog {
    if ($null -eq $script:Catalog) {
        $p = Join-Path $script:RepoRoot 'pigemma\data\catalog.json'
        $c = Read-JsonFile $p
        if ($null -eq $c) { throw "cannot read the catalog $p" }
        $script:Catalog = $c
    }
    return $script:Catalog
}

function Get-CatalogModel([string]$Id) {
    foreach ($m in @(Get-Prop (Get-Catalog) 'models')) {
        if ($null -ne $m -and [string](Get-Prop $m 'id') -eq $Id) { return $m }
    }
    return $null
}

function Get-CatalogCandidates([object]$Entry) {
    $out = New-Object System.Collections.ArrayList
    $lp = Get-Prop $Entry 'local_path'
    $file = [string](Get-Prop $Entry 'file')
    $repo = [string](Get-Prop $Entry 'repo')
    if ($lp) { [void]$out.Add((Join-Path $script:RepoRoot ([string]$lp).Replace('/', '\'))) }
    $dirs = New-Object System.Collections.ArrayList
    foreach ($v in @($env:PIGEMMA_MODELS_DIR, $env:MODEL_DIR)) { if ($v) { [void]$dirs.Add($v) } }
    [void]$dirs.Add((Join-Path $script:RepoRoot 'models'))
    foreach ($d in $dirs) {
        if ($repo -and $file) { [void]$out.Add((Join-Path (Join-Path $d ($repo.Replace('/', '__'))) $file.Replace('/', '\'))) }
        if ($file) { [void]$out.Add((Join-Path $d ([IO.Path]::GetFileName($file.Replace('/', '\'))))) }
    }
    $uniq = New-Object System.Collections.ArrayList
    $seen = @{}
    foreach ($c in $out) {
        $k = $c.ToLowerInvariant()
        if (-not $seen.ContainsKey($k)) { $seen[$k] = $true; [void]$uniq.Add($c) }
    }
    return $uniq.ToArray()
}

function Resolve-ModelSpec([string]$Spec) {
    $s = $Spec.Trim()
    if ($s -eq '26b') { $s = '26b-q4' }
    if ($s -eq '31b') { $s = '31b-q4' }
    $entry = Get-CatalogModel $s
    if ($null -ne $entry) {
        if ([string](Get-Prop $entry 'kind') -eq 'draft') { throw "model '$Spec' is a draft sidecar, not a runnable model" }
        $cands = @(Get-CatalogCandidates $entry)
        $path = $null
        foreach ($c in $cands) { if (Test-Path -LiteralPath $c -PathType Leaf) { $path = $c; break } }
        $present = $true
        if (-not $path) { $path = $cands[0]; $present = $false }
        return [pscustomobject]@{
            Spec = $s; Key = $s; Id = $s; Family = [string](Get-Prop $entry 'family')
            Kind = [string](Get-Prop $entry 'kind'); Path = $path; Present = $present
            Bytes = Get-Prop $entry 'bytes'; Entry = $entry
        }
    }
    $full = Get-FullPathFrom $script:OrigCwd $s
    if ((Test-Path -LiteralPath $full -PathType Leaf) -or $s -like '*.gguf') {
        $leaf = [IO.Path]::GetFileName($full)
        $id = $null
        $fam = $null
        $kind = 'unknown'
        foreach ($m in @(Get-Prop (Get-Catalog) 'models')) {
            if ([IO.Path]::GetFileName(([string](Get-Prop $m 'file')).Replace('/', '\')) -eq $leaf) {
                $id = [string](Get-Prop $m 'id'); $fam = [string](Get-Prop $m 'family'); $kind = [string](Get-Prop $m 'kind'); break
            }
        }
        if (-not $fam) { if ($leaf -match '31B') { $fam = '31b' } else { $fam = '26b' } }
        $key = $id
        if (-not $key) { $key = [IO.Path]::GetFileNameWithoutExtension($leaf) }
        $present = Test-Path -LiteralPath $full -PathType Leaf
        $bytes = $null
        if ($present) { $bytes = (Get-Item -LiteralPath $full).Length }
        return [pscustomobject]@{
            Spec = $full; Key = $key; Id = $id; Family = $fam; Kind = $kind; Path = $full
            Present = $present; Bytes = $bytes; Entry = $null
        }
    }
    throw "unknown model '$Spec' (use 26b, 31b, a catalog id or a .gguf path)"
}

function Get-ModelCfg([object]$M) {
    $q = [bool]$Quick
    if ($M.Family -eq '31b') {
        $cfg = @{ Threads = @(6, 12); Reps = 2; Tg = 16; Secondary = @('mtp'); HasDflash = $false; LowerQuant = $LowerQuant31 }
    } else {
        $cfg = @{ Threads = @(4, 6, 12); Reps = 3; Tg = 32; Secondary = @('mtp', 'dflash', 'ngram'); HasDflash = $true; LowerQuant = $LowerQuant26 }
    }
    if ($q) { $cfg.Threads = @(6); $cfg.Reps = 1 }
    if ($script:Opt.Threads.Count -gt 0) { $cfg.Threads = @($script:Opt.Threads) }
    if ($M.Id -ne '26b-q4' -and $M.Id -ne '31b-q4') { $cfg.LowerQuant = $null }
    return $cfg
}

# ------------------------------------------------------------------------------------------------
# Envelope index (results JSON envelopes in one directory, cached by mtime)
# ------------------------------------------------------------------------------------------------
function ConvertTo-EnvRecord([object]$D, [IO.FileInfo]$F) {
    $exp = Get-Prop $D 'experiment'
    $kind = Get-Prop $D 'kind'
    if ([string]::IsNullOrEmpty([string]$exp) -or [string]::IsNullOrEmpty([string]$kind)) { return $null }
    $mid = Get-Prop $D 'model.id'
    $mfile = Get-Prop $D 'model.file'
    $key = $null
    if ($mid) { $key = [string]$mid }
    elseif ($mfile) { $key = [IO.Path]::GetFileNameWithoutExtension(([string]$mfile).Replace('/', '\')) }
    $failed = $false
    $notes = Get-Prop $D 'notes'
    if ($null -ne $notes) { foreach ($n in @($notes)) { if (([string]$n).StartsWith('failed')) { $failed = $true } } }
    $req = Get-Prop $D 'extra.requests'
    $reqs = @()
    if ($null -ne $req) { $reqs = @($req) }
    return [pscustomobject]@{
        Path = $F.FullName; Name = $F.Name; Mtime = $F.LastWriteTimeUtc
        Experiment = [string]$exp; Kind = [string]$kind; CreatedAt = [string](Get-Prop $D 'created_at')
        ModelKey = $key; ModelBytes = Get-Prop $D 'model.bytes'
        Variant = [string](Get-Prop $D 'variant'); Threads = Get-Prop $D 'threads'
        Source = [string](Get-Prop $D 'extra.source'); Role = [string](Get-Prop $D 'extra.role')
        Test = [string](Get-Prop $D 'extra.test')
        ServerVariant = [string](Get-Prop $D 'extra.server_variant')
        LoadVariant = [string](Get-Prop $D 'extra.load_variant')
        Round = Get-Prop $D 'extra.round'; Rounds = Get-Prop $D 'extra.rounds'
        NPredict = Get-Prop $D 'extra.n_predict'; NRequests = Get-Prop $D 'extra.n_requests'
        Failed = $failed; Notes = $notes
        PP = Get-Prop $D 'pp.tok_s'; PPsd = Get-Prop $D 'pp.sd'; PPn = Get-Prop $D 'pp.n'
        TG = Get-Prop $D 'tg.tok_s'; TGsd = Get-Prop $D 'tg.sd'; TGn = Get-Prop $D 'tg.n'
        LoadS = Get-Prop $D 'load_s'; RssPeakMB = Get-Prop $D 'rss_peak_mb'
        PrivatePeakMB = Get-Prop $D 'private_peak_mb'; MappedMB = Get-Prop $D 'mapped_mb'
        BufferSplit = Get-Prop $D 'buffer_split'
        SpecActive = Get-Prop $D 'spec_active'; AcceptRate = Get-Prop $D 'accept_rate'
        DraftN = Get-Prop $D 'draft_n'; DraftNAccepted = Get-Prop $D 'draft_n_accepted'
        SpecTypeEffective = Get-Prop $D 'spec_type_effective'
        IdenticalToPlain = Get-Prop $D 'identical_to_plain'
        NDrafting = Get-Prop $D 'extra.n_drafting'; NOk = Get-Prop $D 'extra.n_ok'
        Requests = $reqs
        KldMean = Get-Prop $D 'kld.mean'; KldMax = Get-Prop $D 'kld.max'; KldP999 = Get-Prop $D 'kld.p999'
        KldSameTopP = Get-Prop $D 'kld.same_top_p'; PplBase = Get-Prop $D 'kld.ppl_base'
        Ppl = Get-Prop $D 'ppl'; BaseLogits = [string](Get-Prop $D 'extra.base.logits')
        QaScore = Get-Prop $D 'qa_score'; QaTotal = Get-Prop $D 'qa_total'
        QaSetSha = [string](Get-Prop $D 'extra.qa_set.sha256'); QaNItems = Get-Prop $D 'extra.qa_set.n_items'
        BgCpu = Get-Prop $D 'extra.background_cpu.mean_pct'
    }
}

function Get-Envelopes([string]$Dir) {
    if (-not $Dir -or -not (Test-Path -LiteralPath $Dir -PathType Container)) { return @() }
    $out = New-Object System.Collections.ArrayList
    foreach ($f in @(Get-ChildItem -LiteralPath $Dir -File -Filter '*.json' -ErrorAction SilentlyContinue)) {
        if ($f.Name -notlike '*.json' -or $f.Name.StartsWith('.')) { continue }
        $stamp = $f.LastWriteTimeUtc.Ticks
        $cached = $script:EnvCache[$f.FullName]
        if ($null -ne $cached -and $cached.Stamp -eq $stamp) {
            $rec = $cached.Rec
        } else {
            $d = Read-JsonFile $f.FullName
            $rec = $null
            if ($null -ne $d -and -not ($d -is [array])) { $rec = ConvertTo-EnvRecord $d $f }
            $script:EnvCache[$f.FullName] = @{ Stamp = $stamp; Rec = $rec }
        }
        if ($null -ne $rec) { [void]$out.Add($rec) }
    }
    return $out.ToArray()
}

function Select-Newest([object[]]$List) {
    $arr = @($List | Where-Object { $null -ne $_ })
    if ($arr.Count -eq 0) { return $null }
    return ($arr | Sort-Object -Property @{ Expression = { [string]$_.CreatedAt } }, @{ Expression = { $_.Mtime } } |
            Select-Object -Last 1)
}

function Get-Mean([object[]]$Values) {
    $v = @($Values | Where-Object { $null -ne $_ } | ForEach-Object { [double]$_ })
    if ($v.Count -eq 0) { return $null }
    return (($v | Measure-Object -Sum).Sum / $v.Count)
}

# ------------------------------------------------------------------------------------------------
# Stage bench (E0)
# ------------------------------------------------------------------------------------------------
function Find-GitBash {
    $cands = New-Object System.Collections.ArrayList
    if ($env:PIGEMMA_BASH) { [void]$cands.Add($env:PIGEMMA_BASH) }
    $git = Get-Command git -ErrorAction SilentlyContinue
    if ($null -ne $git -and $git.Source) {
        $gdir = Split-Path -Parent (Split-Path -Parent $git.Source)
        [void]$cands.Add((Join-Path $gdir 'bin\bash.exe'))
        [void]$cands.Add((Join-Path $gdir 'usr\bin\bash.exe'))
    }
    foreach ($base in @($env:ProgramFiles, $env:ProgramW6432, 'C:\Program Files')) {
        if ($base) { [void]$cands.Add((Join-Path $base 'Git\bin\bash.exe')) }
    }
    if ($env:OS -ne 'Windows_NT') {
        $b = Get-Command bash -ErrorAction SilentlyContinue
        if ($null -ne $b) { [void]$cands.Add($b.Source) }
    }
    foreach ($c in $cands) { if ($c -and (Test-Path -LiteralPath $c -PathType Leaf)) { return $c } }
    return $null
}

function Test-MembwJson([string]$Path) {
    $d = Read-JsonFile $Path
    if ($null -eq $d) { return $false }
    return ([string](Get-Prop $d 'tool') -eq 'membw' -and $null -ne (Get-Prop $d 'results'))
}

function Invoke-MembwStep {
    $name = 'membw (desktop memory bandwidth, Docker; D5)'
    $out = Join-Path $script:Opt.ResultsDirAbs 'membw_desktop.json'
    if ($NoMembw) { $null = Add-Skip 'bench' $name 'skipped' '-NoMembw given'; return }
    if ((Test-Path -LiteralPath $out -PathType Leaf) -and (Test-MembwJson $out) -and -not $Force) {
        $null = Add-Skip 'bench' $name 'skipped' "$(Get-RelArg $out) present (-Force re-runs it)"
        return
    }
    $bash = Find-GitBash
    $docker = Get-Command docker -ErrorAction SilentlyContinue
    $bin = Join-Path $script:RepoRoot ($script:MEMBW_BIN.Replace('/', '\'))
    $why = @()
    if (-not $bash) { $why += 'Git Bash not found (set PIGEMMA_BASH)' }
    if ($null -eq $docker) { $why += 'docker not on PATH' }
    if (-not (Test-Path -LiteralPath $bin -PathType Leaf)) { $why += "$($script:MEMBW_BIN) not built (bash pc/docker_run.sh build-tools)" }
    $pyArgs = @('-m', 'pigemma.paths', 'lock', '--hold', '--min-free-gb', '2', '--', $(if ($bash) { $bash } else { 'bash' }),
                'pc/docker_run.sh', '--no-stdin', '--', $script:MEMBW_BIN) + $script:MEMBW_ARGS
    if ($why.Count -gt 0) {
        if ($DryRun) { Write-Line ''; Write-Line "== [bench] $name"; Write-Line "   $(Format-Cmd $pyArgs) > $(Get-RelArg $out)" }
        $null = Add-Skip 'bench' $name 'warn' ("cannot measure: " + ($why -join '; ') + '; cache_sim keeps the assumed bandwidth')
        return
    }
    $rec = Invoke-Step -StageName 'bench' -Name $name -PyArgs $pyArgs -NoInnerDryRun -Soft `
        -Note "stdout -> $(Get-RelArg $out); held under the bench lock (no model job may run meanwhile)"
    if ($DryRun -or $rec.Status -ne 'done') { return }
    $text = (@($rec.Lines) -join "`n").Replace("`r", '')
    $obj = $null
    try { $obj = $text | ConvertFrom-Json } catch { $obj = $null }
    if ($null -eq $obj -or [string](Get-Prop $obj 'tool') -ne 'membw') {
        $rec.Status = 'warn'
        $rec.Note = 'membw output was not membw JSON; nothing written'
        Write-Warn $rec.Note
        return
    }
    Write-TextFile $out @($text.Trim())
    Write-Note "wrote $(Get-RelArg $out)"
}

function Get-E0MatrixState([object[]]$Envs, [object]$M, [string]$Variant, [object]$Cfg) {
    $mine = @($Envs | Where-Object {
            $_.Experiment -eq 'E0' -and $_.ModelKey -eq $M.Key -and $_.Variant -eq $Variant -and
            $_.Source -like 'llama-bench*' })
    $done = $true
    foreach ($t in @($Cfg.Threads)) {
        $okPp = @($mine | Where-Object { -not $_.Failed -and [string]$_.Threads -eq [string]$t -and $null -ne $_.PP -and
                $_.Test -eq "pp$($script:E0_PP)" -and ($null -eq $_.PPn -or [int]$_.PPn -ge $Cfg.Reps) })
        $okTg = @($mine | Where-Object { -not $_.Failed -and [string]$_.Threads -eq [string]$t -and $null -ne $_.TG -and
                $_.Test -eq "tg$($Cfg.Tg)" -and ($null -eq $_.TGn -or [int]$_.TGn -ge $Cfg.Reps) })
        if ($okPp.Count -eq 0 -or $okTg.Count -eq 0) { $done = $false }
    }
    return [pscustomobject]@{ Done = $done; Files = @($mine | ForEach-Object { $_.Path }) }
}

function Get-E0CompletionState([object[]]$Envs, [object]$M, [string]$Variant) {
    $mine = @($Envs | Where-Object {
            $_.Experiment -eq 'E0' -and $_.ModelKey -eq $M.Key -and $_.Variant -eq $Variant -and
            $_.Source -like 'llama-completion*' })
    $ok = @($mine | Where-Object { -not $_.Failed -and $null -ne $_.TG })
    return [pscustomobject]@{ Done = ($ok.Count -gt 0); Files = @($mine | ForEach-Object { $_.Path }) }
}

function Invoke-StageBench {
    Write-Line ''
    Write-Line '#### Stage bench (E0): llama-bench matrix, llama-completion per variant, repack gate'
    Invoke-MembwStep
    $resArg = Get-DirArg $script:Opt.ResultsDirAbs
    foreach ($m in $script:ModelInfos) {
        $cfg = Get-ModelCfg $m
        $tlist = (@($cfg.Threads) | ForEach-Object { [string]$_ }) -join ','
        if (-not $m.Present) {
            $null = Add-Skip 'bench' "matrix $($m.Key)" 'missing' "model file not found: $($m.Path)"
            foreach ($v in $script:E0_VARIANTS) { $null = Add-Skip 'bench' "completion $($m.Key) $v" 'missing' 'model file not found' }
            continue
        }
        $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
        $todo = New-Object System.Collections.ArrayList
        $sup = New-Object System.Collections.ArrayList
        foreach ($v in $script:E0_VARIANTS) {
            $st = Get-E0MatrixState $envs $m $v $cfg
            if (-not $st.Done -or $Force) {
                [void]$todo.Add($v)
                foreach ($f in $st.Files) { [void]$sup.Add($f) }
            }
        }
        $variants = $script:E0_VARIANTS -join ','
        if ($todo.Count -gt 0) { $variants = ($todo.ToArray() -join ',') }
        $pyArgs = @('-m', 'pigemma.bench', 'matrix', '--model', $m.Spec, '--threads', $tlist,
                    '--pp', [string]$script:E0_PP, '--tg', [string]$cfg.Tg, '-r', [string]$cfg.Reps,
                    '--variants', $variants, '--out', $resArg, '--experiment', 'E0') +
                  (Get-LockArgs) + (Get-TimeoutArgs) + (Get-IdleArgs)
        $null = Invoke-Step -StageName 'bench' -Name "matrix $($m.Key) [$variants] t=$tlist" -PyArgs $pyArgs -Heavy `
            -Done ($todo.Count -eq 0) -DoneWhy "E0 llama-bench envelopes for all variants and threads $tlist present" `
            -Supersede $sup.ToArray()
        foreach ($v in $script:E0_VARIANTS) {
            $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
            $st = Get-E0CompletionState $envs $m $v
            $pyArgs = @('-m', 'pigemma.bench', 'completion', '--model', $m.Spec, '--prompt-file', $script:E0_PROMPT,
                        '-n', [string]$CompletionTokens, '--threads', [string]$RunThreads, '--variant', $v,
                        '--out', $resArg, '--experiment', 'E0') + (Get-LockArgs) + (Get-TimeoutArgs) + (Get-IdleArgs)
            $sup2 = @()
            if (-not $st.Done -or $Force) { $sup2 = @($st.Files) }
            $null = Invoke-Step -StageName 'bench' -Name "completion $($m.Key) $v" -PyArgs $pyArgs -Heavy `
                -Done $st.Done -DoneWhy 'E0 llama-completion envelope present' -Supersede $sup2
        }
    }
    if ($DryRun) {
        Write-Line ''
        Write-Line "   (dry run) the E0 repack gate would be evaluated into $(Get-RelArg (Join-Path $script:Opt.ResultsDirAbs 'E0_GATE.md'))"
        return
    }
    Write-E0Gate
}

function Get-OrderedVariants([string[]]$Names) {
    # E0 variants first in the documented order, then any other variant alphabetically
    $u = @($Names | Where-Object { $_ } | Select-Object -Unique)
    $first = @($script:E0_VARIANTS | Where-Object { $u -contains $_ })
    $rest = @($u | Where-Object { $script:E0_VARIANTS -notcontains $_ } | Sort-Object)
    return @($first + $rest)
}

function Select-GateEnv([object[]]$Comp, [string]$Variant) {
    $c = @($Comp | Where-Object { $_.Variant -eq $Variant })
    $best = Select-Newest @($c | Where-Object { -not $_.Failed -and $null -ne $_.BufferSplit })
    if ($null -eq $best) { $best = Select-Newest @($c | Where-Object { -not $_.Failed }) }
    if ($null -eq $best) { $best = Select-Newest $c }
    return $best
}

function Test-HasRepack([object]$Bs) {
    foreach ($n in (Get-PropNames $Bs)) { if ($n -like 'CPU_REPACK*') { return $true } }
    return $false
}

function New-Check([string]$Model, [string]$Check, [string]$Value, [string]$Threshold, [string]$Result, [object]$Env) {
    $file = ''
    if ($null -ne $Env) { $file = Get-RelArg $Env.Path }
    return [pscustomobject]@{ Model = $Model; Check = $Check; Value = $Value; Threshold = $Threshold; Result = $Result; File = $file }
}

function Get-GateChecks([string]$Key, [object[]]$Envs, [object]$Bytes) {
    $comp = @($Envs | Where-Object { $_.Experiment -eq 'E0' -and $_.ModelKey -eq $Key -and $_.Source -like 'llama-completion*' })
    $nr = Select-GateEnv $comp 'norepack'
    $hy = Select-GateEnv $comp 'hybrid_negctrl'
    $df = Select-GateEnv $comp 'default'
    if ($null -eq $Bytes) {
        $b = Select-Newest @($comp | Where-Object { $null -ne $_.ModelBytes })
        if ($null -ne $b) { $Bytes = $b.ModelBytes }
    }
    $checks = New-Object System.Collections.ArrayList
    $fileMiB = $null
    $needMiB = $null
    if ($null -ne $Bytes) { $fileMiB = [double]$Bytes / 1048576.0; $needMiB = $script:GATE_MAPPED_FRACTION * $fileMiB }
    # 1. norepack keeps the whole file CPU_Mapped
    $thr = '>= 95% of the file'
    if ($null -ne $needMiB) { $thr = ">= $(Fmt $needMiB 1) MiB (95% of the $(Fmt $fileMiB 1) MiB file)" }
    if ($null -eq $nr) {
        [void]$checks.Add((New-Check $Key 'norepack: CPU_Mapped MiB' 'no norepack completion envelope' $thr 'NO DATA' $null))
        [void]$checks.Add((New-Check $Key 'norepack: no CPU_REPACK buffer line' 'no norepack completion envelope' 'absent' 'NO DATA' $null))
        [void]$checks.Add((New-Check $Key 'norepack: private peak MB' 'no norepack completion envelope' '< 1500 MB' 'NO DATA' $null))
    } else {
        $bs = $nr.BufferSplit
        $mapped = Get-Prop $bs 'CPU_Mapped_mib'
        if ($null -eq $bs) { $r = 'NO DATA'; $val = 'no load_tensors buffer lines parsed (needs -lv 4)' }
        elseif ($null -eq $mapped) { $r = 'FAIL'; $val = 'no CPU_Mapped line' }
        elseif ($null -eq $needMiB) { $r = 'NO DATA'; $val = "$(Fmt $mapped 1) MiB (file size unknown)" }
        elseif ([double]$mapped -ge $needMiB) { $r = 'PASS'; $val = "$(Fmt $mapped 1) MiB" }
        else { $r = 'FAIL'; $val = "$(Fmt $mapped 1) MiB" }
        [void]$checks.Add((New-Check $Key 'norepack: CPU_Mapped MiB' $val $thr $r $nr))
        if ($null -eq $bs) { $r = 'NO DATA'; $val = 'no load_tensors buffer lines parsed' }
        elseif (Test-HasRepack $bs) { $r = 'FAIL'; $val = "CPU_REPACK line present ($(Fmt (Get-Prop $bs 'CPU_REPACK_mib') 1) MiB)" }
        else { $r = 'PASS'; $val = 'absent' }
        [void]$checks.Add((New-Check $Key 'norepack: no CPU_REPACK buffer line' $val 'absent' $r $nr))
        $pp = $nr.PrivatePeakMB
        if ($null -eq $pp) { $r = 'NO DATA'; $val = 'not sampled' }
        elseif ([double]$pp -lt $script:GATE_PRIVATE_PEAK_MB) { $r = 'PASS'; $val = "$(Fmt $pp 1) MB" }
        else { $r = 'FAIL'; $val = "$(Fmt $pp 1) MB" }
        [void]$checks.Add((New-Check $Key 'norepack: private peak MB' $val '< 1500 MB (1.5 GB)' $r $nr))
    }
    # 2. negative control C1: -ot ffn_(gate_up|down)_exps=CPU still repacks
    if ($null -eq $hy) {
        [void]$checks.Add((New-Check $Key 'C1 hybrid_negctrl: CPU_REPACK line present' 'no hybrid_negctrl completion envelope' 'present' 'NO DATA' $null))
    } else {
        $bs = $hy.BufferSplit
        if ($null -eq $bs) { $r = 'NO DATA'; $val = 'no load_tensors buffer lines parsed' }
        elseif (Test-HasRepack $bs) { $r = 'PASS'; $val = "present ($(Fmt (Get-Prop $bs 'CPU_REPACK_mib') 1) MiB): -ot does not stream" }
        else { $r = 'FAIL'; $val = 'absent' }
        [void]$checks.Add((New-Check $Key 'C1 hybrid_negctrl: CPU_REPACK line present' $val 'present' $r $hy))
    }
    if ($null -ne $df -and $null -ne $df.BufferSplit) {
        $val = "CPU_REPACK $(Fmt (Get-Prop $df.BufferSplit 'CPU_REPACK_mib') 1) MiB, CPU_Mapped $(Fmt (Get-Prop $df.BufferSplit 'CPU_Mapped_mib') 1) MiB"
        [void]$checks.Add((New-Check $Key 'reference: default buffer split' $val 'none' 'INFO' $df))
    }
    return $checks.ToArray()
}

function Get-GateVerdict([object[]]$Checks) {
    $gate = @($Checks | Where-Object { $_.Result -ne 'INFO' })
    if (@($gate | Where-Object { $_.Result -eq 'FAIL' }).Count -gt 0) { return 'FAIL' }
    if (@($gate | Where-Object { $_.Result -eq 'NO DATA' }).Count -gt 0) { return 'INCOMPLETE' }
    return 'PASS'
}

function Write-E0Gate {
    $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
    $keys = New-Object System.Collections.ArrayList
    $seen = @{}
    $bytesByKey = @{}
    foreach ($m in $script:ModelInfos) {
        if (-not $seen.ContainsKey($m.Key.ToLowerInvariant())) { $seen[$m.Key.ToLowerInvariant()] = $true; [void]$keys.Add($m.Key); $bytesByKey[$m.Key] = $m.Bytes }
    }
    foreach ($e in @($envs | Where-Object { $_.Experiment -eq 'E0' -and $_.ModelKey })) {
        if (-not $seen.ContainsKey($e.ModelKey.ToLowerInvariant())) { $seen[$e.ModelKey.ToLowerInvariant()] = $true; [void]$keys.Add($e.ModelKey); $bytesByKey[$e.ModelKey] = $e.ModelBytes }
    }
    $runKeys = @{}
    foreach ($m in $script:ModelInfos) { if ($m.Present) { $runKeys[$m.Key.ToLowerInvariant()] = $true } }
    $now = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ', $script:Inv)
    $md = New-Object System.Collections.ArrayList
    [void]$md.Add("# E0 gate - $($script:Opt.Date)")
    [void]$md.Add('')
    [void]$md.Add("Generated $now by ``$($script:ToolName)`` $($script:ToolVersion) from the MEASURED envelopes in ``$(Get-DirArg $script:Opt.ResultsDirAbs)``.")
    [void]$md.Add('Thresholds are the pre-registered E0 rules of [docs/EXPERIMENT_PLAN.md](../../docs/EXPERIMENT_PLAN.md#e0-desktop-baseline-and-repack-gate):')
    [void]$md.Add('REPACK GATE PASS if the norepack variant logs CPU_Mapped >= 95% of the file size with no CPU_REPACK line and private peak < 1.5 GB,')
    [void]$md.Add('AND hybrid_negctrl logs a CPU_REPACK line (confirms -ot does not stream). If the gate fails, stop all streaming work (E5/E6/Pi) until the flags are fixed.')
    [void]$md.Add('Speed: 26B tg@4t within +-20% of 15.18 and tg@6t within +-20% of 13.5, 31B tg@6t within +-20% of 2.44 (idle-machine references). MiB = 2^20 bytes; MB = 1e6 bytes.')
    [void]$md.Add('')
    $verdicts = @{}
    Write-Line ''
    Write-Line '#### E0 gate'
    foreach ($k in $keys) {
        $checks = @(Get-GateChecks $k $envs $bytesByKey[$k])
        $comp = @($envs | Where-Object { $_.Experiment -eq 'E0' -and $_.ModelKey -eq $k })
        if ($comp.Count -eq 0) { continue }
        $verdict = Get-GateVerdict $checks
        $verdicts[$k] = $verdict
        if ($runKeys.ContainsKey($k.ToLowerInvariant()) -and $verdict -ne 'PASS') { $script:GateFailed = $true }
        Write-Line "E0 REPACK GATE $k : $verdict"
        foreach ($c in $checks) { Write-Line ("  [{0}] {1}: {2} (threshold {3})" -f $c.Result, $c.Check, $c.Value, $c.Threshold) }
        [void]$md.Add("## $k - repack gate: **$verdict**")
        [void]$md.Add('')
        [void]$md.Add('| Check | Value | Threshold | Result | Evidence |')
        [void]$md.Add('|---|---|---|---|---|')
        foreach ($c in $checks) { [void]$md.Add("| $($c.Check) | $($c.Value) | $($c.Threshold) | $($c.Result) | $($c.File) |") }
        [void]$md.Add('')
        # speed band
        $mat = @($envs | Where-Object { $_.Experiment -eq 'E0' -and $_.ModelKey -eq $k -and $_.Source -like 'llama-bench*' -and -not $_.Failed })
        $refs = $script:SPEED_REFS[$k]
        [void]$md.Add("### $k - speed vs the idle-machine reference (+-20%)")
        [void]$md.Add('')
        if ($null -eq $refs) {
            [void]$md.Add('_No pre-registered speed reference for this model._')
        } else {
            [void]$md.Add('| Check | Measured tg tok/s (sd) | Band | Result | Background CPU % | Evidence |')
            [void]$md.Add('|---|---|---|---|---|---|')
            foreach ($ref in $refs) {
                $lo = $ref.Ref * (1 - $script:SPEED_BAND); $hi = $ref.Ref * (1 + $script:SPEED_BAND)
                $e = Select-Newest @($mat | Where-Object { $_.Variant -eq 'default' -and [string]$_.Threads -eq [string]$ref.T -and $null -ne $_.TG })
                if ($null -eq $e) { $res = 'not run'; $val = 'not run'; $bg = 'n/a'; $file = '' }
                else {
                    $val = "$(Fmt $e.TG 2) ($(Fmt $e.TGsd 2))"
                    if ([double]$e.TG -ge $lo -and [double]$e.TG -le $hi) { $res = 'PASS' } else { $res = 'FAIL' }
                    $bg = Fmt $e.BgCpu 1; $file = Get-RelArg $e.Path
                }
                Write-Line ("E0 SPEED {0} tg@{1}t default: {2} tok/s, band {3}-{4}: {5}" -f $k, $ref.T, $val, (Fmt $lo 2), (Fmt $hi 2), $res)
                [void]$md.Add("| tg@$($ref.T)t default | $val | $(Fmt $lo 2)-$(Fmt $hi 2) | $res | $bg | $file |")
            }
        }
        [void]$md.Add('')
        # matrix table + norepack penalty
        if ($mat.Count -gt 0) {
            [void]$md.Add("### $k - llama-bench matrix (MEASURED; random token ids: engine throughput only)")
            [void]$md.Add('')
            [void]$md.Add('| Variant | Threads | pp tok/s (sd) | tg tok/s (sd) | tg / default | Background CPU % |')
            [void]$md.Add('|---|---|---|---|---|---|')
            $vars = @(Get-OrderedVariants @($mat | ForEach-Object { $_.Variant }))
            $thr = @($mat | ForEach-Object { [int]$_.Threads } | Sort-Object -Unique)
            foreach ($v in $vars) {
                foreach ($t in $thr) {
                    $p = Select-Newest @($mat | Where-Object { $_.Variant -eq $v -and [int]$_.Threads -eq $t -and $null -ne $_.PP })
                    $g = Select-Newest @($mat | Where-Object { $_.Variant -eq $v -and [int]$_.Threads -eq $t -and $null -ne $_.TG })
                    if ($null -eq $p -and $null -eq $g) { continue }
                    $d = Select-Newest @($mat | Where-Object { $_.Variant -eq 'default' -and [int]$_.Threads -eq $t -and $null -ne $_.TG })
                    $ratio = 'n/a'
                    if ($null -ne $g -and $null -ne $d -and [double]$d.TG -gt 0) { $ratio = Fmt ([double]$g.TG / [double]$d.TG) 3 }
                    $ps = 'not run'; if ($null -ne $p) { $ps = "$(Fmt $p.PP 2) ($(Fmt $p.PPsd 2))" }
                    $gs = 'not run'; $bg = 'n/a'
                    if ($null -ne $g) { $gs = "$(Fmt $g.TG 2) ($(Fmt $g.TGsd 2))"; $bg = Fmt $g.BgCpu 1 }
                    [void]$md.Add("| $v | $t | $ps | $gs | $ratio | $bg |")
                    if ($v -eq 'norepack' -and $ratio -ne 'n/a') {
                        Write-Line ("E0 NOREPACK PENALTY {0} @{1}t: tg(norepack)/tg(default) = {2} (reported, expected 0.60-0.90)" -f $k, $t, $ratio)
                    }
                }
            }
            [void]$md.Add('')
            [void]$md.Add('The norepack penalty is the `tg / default` column of the norepack rows (reported, not pass/fail; expected 0.60-0.90).')
            [void]$md.Add('')
        }
        # completion table
        $cl = @($comp | Where-Object { $_.Source -like 'llama-completion*' })
        if ($cl.Count -gt 0) {
            [void]$md.Add("### $k - llama-completion runs (load, memory, buffers)")
            [void]$md.Add('')
            [void]$md.Add('| Variant | load_s | tg tok/s | rss_peak_mb | private_peak_mb | mapped_mb | CPU_Mapped MiB | CPU_REPACK MiB | Status | Evidence |')
            [void]$md.Add('|---|---|---|---|---|---|---|---|---|---|')
            $vord = @(Get-OrderedVariants @($cl | ForEach-Object { $_.Variant }))
            foreach ($e in @($cl | Sort-Object -Property @{ Expression = { [array]::IndexOf($vord, $_.Variant) } }, CreatedAt)) {
                $st = 'ok'; if ($e.Failed) { $st = 'failed' }
                $rp = 'absent'; if (Test-HasRepack $e.BufferSplit) { $rp = Fmt (Get-Prop $e.BufferSplit 'CPU_REPACK_mib') 1 }
                if ($null -eq $e.BufferSplit) { $rp = 'n/a' }
                [void]$md.Add("| $($e.Variant) | $(Fmt $e.LoadS 2) | $(Fmt $e.TG 2) | $(Fmt $e.RssPeakMB 1) | $(Fmt $e.PrivatePeakMB 1) | $(Fmt $e.MappedMB 1) | $(Fmt (Get-Prop $e.BufferSplit 'CPU_Mapped_mib') 1) | $rp | $st | $(Get-RelArg $e.Path) |")
            }
            [void]$md.Add('')
        }
    }
    # membw (D5)
    $mb = Join-Path $script:Opt.ResultsDirAbs 'membw_desktop.json'
    [void]$md.Add('## Desktop memory bandwidth (D5)')
    [void]$md.Add('')
    if (Test-MembwJson $mb) {
        $d = Read-JsonFile $mb
        $best = $null; $bestT = $null
        foreach ($r in @(Get-Prop $d 'results')) {
            $v = Get-Prop $r 'read_gbps'
            if ($null -ne $v -and ($null -eq $best -or [double]$v -gt $best)) { $best = [double]$v; $bestT = Get-Prop $r 'threads' }
        }
        [void]$md.Add("MEASURED: membw best read $(Fmt $best 2) GB/s at $bestT threads (``$(Get-RelArg $mb)``, Docker on this desktop). ``python -m pigemma.cache_sim --e0-dir $(Get-DirArg $script:Opt.ResultsDirAbs)`` derives llama_efficiency from it.")
        Write-Line "E0 MEMBW best read $(Fmt $best 2) GB/s at $bestT threads"
    } else {
        [void]$md.Add('not run (no membw_desktop.json); cache_sim keeps the assumed desktop bandwidth.')
    }
    [void]$md.Add('')
    [void]$md.Add('## Verdict')
    [void]$md.Add('')
    if ($verdicts.Count -eq 0) {
        [void]$md.Add('_No E0 envelopes found._')
    } else {
        foreach ($k in $verdicts.Keys) { [void]$md.Add("- $k : REPACK GATE **$($verdicts[$k])**") }
        if (@($verdicts.Values | Where-Object { $_ -ne 'PASS' }).Count -gt 0) {
            [void]$md.Add('')
            [void]$md.Add('At least one gate is not PASS: stop all streaming work (E5/E6/Pi) until the flags are fixed (or the missing data is measured).')
        }
    }
    $gp = Join-Path $script:Opt.ResultsDirAbs 'E0_GATE.md'
    Write-TextFile $gp $md.ToArray()
    Write-Note "wrote $(Get-RelArg $gp)"
}

# ------------------------------------------------------------------------------------------------
# Stage spec (E1)
# ------------------------------------------------------------------------------------------------
function Get-E1Envs([object[]]$Envs, [string]$Key) {
    return @($Envs | Where-Object {
            $_.Experiment -eq 'E1' -and $_.ModelKey -eq $Key -and $_.Source -eq 'llama-server' -and
            ($_.LoadVariant -eq '' -or $_.LoadVariant -eq 'default') })
}

function Test-NPredictMatch([object]$E, [int]$N) {
    if ($N -le 0) { return ($null -eq $E.NPredict) }
    return ([string]$E.NPredict -eq [string]$N)
}

function Test-NRequestsMatch([object]$E) {
    if ($null -eq $script:Opt.SpecPromptCount -or $null -eq $E.NRequests) { return $true }
    return ([int]$E.NRequests -eq [int]$script:Opt.SpecPromptCount)
}

function Get-E1CoreState([object[]]$Envs, [object]$M) {
    # NB: PowerShell variable names are case-insensitive ($r and $R are one variable)
    $nRounds = $script:Opt.SpecRounds
    $mine = @(Get-E1Envs $Envs $M.Key | Where-Object { ($_.ServerVariant -eq 'plain' -or $_.ServerVariant -eq 'mtp-explicit') -and [string]$_.Rounds -eq [string]$nRounds })
    $done = $true
    foreach ($v in @('plain', 'mtp-explicit')) {
        for ($rnd = 1; $rnd -le $nRounds; $rnd++) {
            $ok = @($mine | Where-Object { $_.ServerVariant -eq $v -and [string]$_.Round -eq [string]$rnd -and -not $_.Failed -and
                    $null -ne $_.TG -and (Test-NPredictMatch $_ $script:Opt.SpecTokens) -and
                    (Test-NRequestsMatch $_) })
            if ($ok.Count -eq 0) { $done = $false }
        }
    }
    return [pscustomobject]@{ Done = $done; Files = @($mine | ForEach-Object { $_.Path }) }
}

function Get-E1VariantState([object[]]$Envs, [object]$M, [string]$Variant) {
    $mine = @(Get-E1Envs $Envs $M.Key | Where-Object { $_.ServerVariant -eq $Variant -and [string]$_.Rounds -eq '1' })
    $ok = @($mine | Where-Object { -not $_.Failed -and $null -ne $_.TG -and (Test-NPredictMatch $_ $script:Opt.SpecTokens) -and
            (Test-NRequestsMatch $_) })
    return [pscustomobject]@{ Done = ($ok.Count -gt 0); Files = @($mine | ForEach-Object { $_.Path }) }
}

function Get-SpecArgs([object]$M, [string]$Variants, [int]$Rounds) {
    $a = @('-m', 'pigemma.bench', 'server-spec', '--model', $M.Spec, '--variant', $Variants, '--rounds', [string]$Rounds,
           '--prompts', $script:GEN_PROMPTS, '-n', [string]$script:Opt.SpecTokens, '--threads', [string]$RunThreads,
           '-c', [string]$script:SPEC_CTX, '--port', [string]$Port, '--out', (Get-DirArg $script:Opt.ResultsDirAbs),
           '--experiment', 'E1', '--request-timeout', [string]$RequestTimeoutS)
    if ($script:Opt.SpecPrompts -gt 0) { $a += @('--max-prompts', [string]$script:Opt.SpecPrompts) }
    if ($DraftNMax -gt 0) { $a += @('--draft-n-max', [string]$DraftNMax) }
    return ($a + (Get-LockArgs) + (Get-IdleArgs))
}

function Test-VariantActive([object[]]$Envs, [string]$Key, [string]$Variant) {
    $e = @(Get-E1Envs $Envs $Key | Where-Object { $_.ServerVariant -eq $Variant -and $_.SpecActive -eq $true })
    return ($e.Count -gt 0)
}

function Invoke-StageSpec {
    Write-Line ''
    Write-Line '#### Stage spec (E1): llama-server speculative decoding (plain vs mtp-explicit interleaved, then the other drafts)'
    foreach ($m in $script:ModelInfos) {
        $cfg = Get-ModelCfg $m
        $nRounds = $script:Opt.SpecRounds
        if (-not $m.Present) {
            $null = Add-Skip 'spec' "server-spec $($m.Key) core" 'missing' "model file not found: $($m.Path)"
            continue
        }
        $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
        $st = Get-E1CoreState $envs $m
        $sup = @()
        if (-not $st.Done -or $Force) { $sup = @($st.Files) }
        $null = Invoke-Step -StageName 'spec' -Name "server-spec $($m.Key) plain,mtp-explicit x$nRounds rounds (interleaved)" `
            -PyArgs (Get-SpecArgs $m 'plain,mtp-explicit' $nRounds) -Heavy -Done $st.Done `
            -DoneWhy "E1 envelopes for plain and mtp-explicit, rounds 1..$nRounds present" -Supersede $sup
        if ($Quick) { continue }
        foreach ($v in @($cfg.Secondary)) {
            $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
            $st = Get-E1VariantState $envs $m $v
            $sup = @()
            if (-not $st.Done -or $Force) { $sup = @($st.Files) }
            $null = Invoke-Step -StageName 'spec' -Name "server-spec $($m.Key) $v" -PyArgs (Get-SpecArgs $m $v 1) -Heavy `
                -Done $st.Done -DoneWhy "E1 envelope for $v present" -Supersede $sup
        }
        if ($cfg.HasDflash) {
            $v = 'dflash-explicit'
            $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
            $st = Get-E1VariantState $envs $m $v
            $mtpA = Test-VariantActive $envs $m.Key 'mtp'
            $dfA = Test-VariantActive $envs $m.Key 'dflash'
            $need = (-not $mtpA) -or (-not $dfA)
            if ($DryRun) {
                $null = Invoke-Step -StageName 'spec' -Name "server-spec $($m.Key) $v (conditional)" -PyArgs (Get-SpecArgs $m $v 1) `
                    -Done $st.Done -DoneWhy "E1 envelope for $v present" `
                    -Note 'runs only when mtp or dflash never drafted (decided at run time from their envelopes)'
            } elseif ($need -or $st.Done) {
                $sup = @()
                if (-not $st.Done -or $Force) { $sup = @($st.Files) }
                $null = Invoke-Step -StageName 'spec' -Name "server-spec $($m.Key) $v" -PyArgs (Get-SpecArgs $m $v 1) -Heavy `
                    -Done $st.Done -DoneWhy "E1 envelope for $v present" -Supersede $sup `
                    -Note "mtp active: $mtpA; dflash active: $dfA (manifest: run dflash-explicit when a draft did not engage)"
            } else {
                $null = Add-Skip 'spec' "server-spec $($m.Key) $v" 'skipped' 'mtp and dflash both drafted: the explicit DFlash variant is not needed'
            }
        }
    }
    if (-not $DryRun) { Write-E1Summary }
}

function Get-E1Verdict([object]$E, [object]$Speedup, [int]$NFast, [int]$NCmp) {
    if ($E.ServerVariant -eq 'plain') { return 'reference' }
    if ($E.Failed -or $null -eq $E.TG) { return 'failed' }
    if ($E.SpecActive -ne $true) {
        if ($null -ne $E.NDrafting -and [int]$E.NDrafting -gt 0) { return "inactive (drafted on $($E.NDrafting)/$($E.NOk) requests)" }
        return 'inactive'
    }
    if ($null -ne $E.IdenticalToPlain -and [double]$E.IdenticalToPlain -lt 1.0) { return 'bug (active drafter changed the greedy output)' }
    if ($null -ne $Speedup -and [double]$Speedup -lt 1.0) { return 'slower' }
    $needFast = [int][Math]::Ceiling($script:E1_USEFUL_PROMPT_FRACTION * [Math]::Max(1, $NCmp))
    if ($NCmp -gt 0 -and $NFast -ge $needFast -and $null -ne $E.IdenticalToPlain -and [double]$E.IdenticalToPlain -ge $script:E1_IDENTICAL_MIN) { return 'useful' }
    return 'marginal (active and lossless, below the useful rule)'
}

function Write-E1Summary {
    $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
    $e1 = @($envs | Where-Object { $_.Experiment -eq 'E1' -and $_.Source -eq 'llama-server' })
    if ($e1.Count -eq 0) { return }
    $now = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ', $script:Inv)
    $md = New-Object System.Collections.ArrayList
    [void]$md.Add("# E1 summary - $($script:Opt.Date)")
    [void]$md.Add('')
    [void]$md.Add("Generated $now by ``$($script:ToolName)`` $($script:ToolVersion) from the MEASURED E1 envelopes in ``$(Get-DirArg $script:Opt.ResultsDirAbs)``.")
    [void]$md.Add('Speedup = tg tok/s / the mean tg of all plain envelopes of the model (same load variant); per-prompt speedups use the mean plain rate of each prompt.')
    [void]$md.Add('Verdict words follow [docs/EXPERIMENT_PLAN.md](../../docs/EXPERIMENT_PLAN.md#e1-speculative-decoding-via-llama-server): ACTIVE = spec_active;')
    [void]$md.Add('useful = per-prompt speedup >= 1.2 on >= 6 of 8 prompts AND identical_to_plain >= 0.95; inactive / slower are findings; an ACTIVE drafter with non-identical greedy output is a bug.')
    [void]$md.Add("'marginal' (added by this driver) = active and lossless but below the useful rule.")
    [void]$md.Add('')
    Write-Line ''
    Write-Line '#### E1 summary'
    $order = @{ 'plain' = 0; 'mtp-explicit' = 1; 'mtp' = 2; 'dflash' = 3; 'dflash-explicit' = 4; 'ngram' = 5 }
    $groups = @($e1 | Group-Object -Property { "$($_.ModelKey)|$($_.LoadVariant)" })
    foreach ($g in $groups) {
        $rows = @($g.Group)
        $key = $rows[0].ModelKey
        $plains = @($rows | Where-Object { $_.ServerVariant -eq 'plain' -and -not $_.Failed -and $null -ne $_.TG })
        $plainMean = Get-Mean @($plains | ForEach-Object { $_.TG })
        $perPrompt = @{}
        foreach ($p in $plains) {
            foreach ($q in @($p.Requests)) {
                $id = [string](Get-Prop $q 'id'); $rate = Get-Prop $q 'predicted_per_second'
                if ($id -and $null -ne $rate -and -not (Get-Prop $q 'error')) {
                    if (-not $perPrompt.ContainsKey($id)) { $perPrompt[$id] = New-Object System.Collections.ArrayList }
                    [void]$perPrompt[$id].Add([double]$rate)
                }
            }
        }
        $lv = $rows[0].LoadVariant; if (-not $lv) { $lv = 'default' }
        [void]$md.Add("## $key (load variant $lv)")
        [void]$md.Add('')
        [void]$md.Add('| Variant | Round | tg tok/s (sd, n) | Speedup vs plain | Prompts >= 1.2x | accept_rate | drafted / accepted | spec_active | spec_type_effective | identical_to_plain | Verdict | Background CPU % | Evidence |')
        [void]$md.Add('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        $sorted = @($rows | Sort-Object -Property @{ Expression = { $o = $order[$_.ServerVariant]; if ($null -eq $o) { 9 } else { $o } } },
                @{ Expression = { [string]$_.Round } }, @{ Expression = { $_.CreatedAt } })
        foreach ($e in $sorted) {
            $speed = $null
            if ($null -ne $e.TG -and $null -ne $plainMean -and $plainMean -gt 0) { $speed = [double]$e.TG / $plainMean }
            $nFast = 0; $nCmp = 0
            foreach ($q in @($e.Requests)) {
                $id = [string](Get-Prop $q 'id'); $rate = Get-Prop $q 'predicted_per_second'
                if ($id -and $null -ne $rate -and $perPrompt.ContainsKey($id)) {
                    $base = Get-Mean @($perPrompt[$id].ToArray())
                    if ($null -ne $base -and $base -gt 0) {
                        $nCmp++
                        if ([double]$rate / $base -ge $script:E1_USEFUL_SPEEDUP) { $nFast++ }
                    }
                }
            }
            $verdict = Get-E1Verdict $e $speed $nFast $nCmp
            $tg = 'failed'
            if ($null -ne $e.TG) { $tg = "$(Fmt $e.TG 2) ($(Fmt $e.TGsd 2), $($e.TGn))" }
            $fast = 'n/a'; if ($e.ServerVariant -ne 'plain' -and $nCmp -gt 0) { $fast = "$nFast/$nCmp" }
            $spd = 'n/a'; if ($null -ne $speed) { $spd = (Fmt $speed 3) + 'x' }
            $sa = 'n/a'; if ($null -ne $e.SpecActive) { $sa = ([string]$e.SpecActive).ToLowerInvariant() }
            $idt = 'n/a'; if ($null -ne $e.IdenticalToPlain) { $idt = Fmt $e.IdenticalToPlain 3 }
            [void]$md.Add("| $($e.ServerVariant) | $($e.Round)/$($e.Rounds) | $tg | $spd | $fast | $(Fmt $e.AcceptRate 3) | $($e.DraftN) / $($e.DraftNAccepted) | $sa | $($e.SpecTypeEffective) | $idt | $verdict | $(Fmt $e.BgCpu 1) | $(Get-RelArg $e.Path) |")
            Write-Line ("E1 {0} {1} round {2}/{3}: tg {4} tok/s, speedup {5}, accept {6}, spec_active {7}, identical {8} -> {9}" -f
                $key, $e.ServerVariant, $e.Round, $e.Rounds, (Fmt $e.TG 2), $spd, (Fmt $e.AcceptRate 3), $sa, $idt, $verdict)
        }
        [void]$md.Add('')
    }
    $p = Join-Path $script:Opt.ResultsDirAbs 'E1_SUMMARY.md'
    Write-TextFile $p $md.ToArray()
    Write-Note "wrote $(Get-RelArg $p)"
}

# ------------------------------------------------------------------------------------------------
# Stage quality (E2)
# ------------------------------------------------------------------------------------------------
function Get-QaExpected([string]$SetPath) {
    $abs = Join-Path $script:RepoRoot ($SetPath.Replace('/', '\'))
    $sha = Get-FileSha256 $abs
    $n = $null
    $items = Read-JsonFile $abs
    if ($null -ne $items) { $n = @($items).Count }
    if ($script:Opt.QaIds -and $SetPath -eq $script:QA_SET) { $n = @(Split-List @($script:Opt.QaIds)).Count }
    return @{ Sha = $sha; N = $n }
}

function Test-QaDone([string]$File, [hashtable]$Expected) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { return $false }
    $d = Read-JsonFile $File
    if ($null -eq $d) { return $false }
    $rec = ConvertTo-EnvRecord $d (Get-Item -LiteralPath $File)
    if ($null -eq $rec -or $rec.Failed -or $null -eq $rec.QaScore) { return $false }
    if ($Expected.Sha -and $rec.QaSetSha -and $rec.QaSetSha -ne $Expected.Sha) { return $false }
    if ($null -ne $Expected.N -and $null -ne $rec.QaNItems -and [int]$rec.QaNItems -ne [int]$Expected.N) { return $false }
    return $true
}

function Test-KldBaseDone([string]$Logits, [object]$M) {
    if (-not (Test-Path -LiteralPath $Logits -PathType Leaf)) { return $false }
    $meta = Read-JsonFile "$Logits.json"
    if ($null -eq $meta) { return $false }
    if ((Get-Prop $meta 'ok') -ne $true) { return $false }
    if ([string](Get-Prop $meta 'chunks') -ne [string]$script:Opt.KldChunks) { return $false }
    if ([string](Get-Prop $meta 'ctx') -ne [string]$script:KLD_CTX) { return $false }
    $mb = Get-Prop $meta 'model.bytes'
    if ($null -ne $mb -and $null -ne $M.Bytes -and [string]$mb -ne [string]$M.Bytes) { return $false }
    $ts = [string](Get-Prop $meta 'text.sha256')
    if ($script:CorpusSha -and $ts -and $ts -ne $script:CorpusSha) { return $false }
    return $true
}

function Test-KldDone([object[]]$Envs, [string]$Key, [string]$Variant, [string]$Logits) {
    $want = Get-PathKey $Logits
    $mine = @($Envs | Where-Object { $_.Experiment -eq 'E2' -and $_.Role -eq 'kld' -and $_.ModelKey -eq $Key -and $_.Variant -eq $Variant })
    $ok = @($mine | Where-Object { -not $_.Failed -and $null -ne $_.KldMean -and (Get-PathKey $_.BaseLogits) -eq $want })
    return [pscustomobject]@{ Done = ($ok.Count -gt 0); Files = @($mine | ForEach-Object { $_.Path }) }
}

function Get-QaArgs([string]$Spec, [string]$Variant, [string]$OutFile, [string]$SetPath, [string]$PlainRef) {
    $a = @('-m', 'pigemma.quality', 'qa', '--model', $Spec, '--threads', [string]$RunThreads, '--server-variant', $Variant,
           '-n', [string]$QaTokens, '-c', [string]$script:QA_CTX, '--port', [string]$Port, '--out', (Get-RelArg $OutFile),
           '--experiment', 'E2', '--request-timeout', [string]$RequestTimeoutS)
    if ($SetPath -ne $script:QA_SET) { $a += @('--qa-set', $SetPath) }
    elseif ($script:Opt.QaIds) { $a += @('--ids', $script:Opt.QaIds) }
    # quality.py falls back to <out dir>/qa_<model>_plain.json itself; name it only when it exists
    if ($PlainRef -and ($DryRun -or (Test-Path -LiteralPath $PlainRef -PathType Leaf))) { $a += @('--plain-ref', (Get-RelArg $PlainRef)) }
    if ($DraftNMax -gt 0 -and $Variant -ne 'plain') { $a += @('--draft-n-max', [string]$DraftNMax) }
    return ($a + (Get-LockArgs))
}

function Invoke-KldStep([object]$M, [string]$Spec, [string]$Key, [string]$Variant, [string]$Logits, [string]$Label) {
    $resArg = Get-DirArg $script:Opt.ResultsDirAbs
    if (-not $DryRun -and -not (Test-Path -LiteralPath $Logits -PathType Leaf)) {
        return (Add-Skip 'quality' $Label 'not-run' "no reference logits at $(Get-RelArg $Logits) (kld-base did not produce them)")
    }
    $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
    $st = Test-KldDone $envs $Key $Variant $Logits
    $sup = @()
    if (-not $st.Done -or $Force) { $sup = @($st.Files) }
    $pyArgs = @('-m', 'pigemma.quality', 'kld', '--model', $Spec, '--base', (Get-RelArg $Logits), '--variant', $Variant,
                '--threads', [string]$RunThreads, '--out', $resArg, '--experiment', 'E2') + (Get-LockArgs) + (Get-TimeoutArgs)
    return (Invoke-Step -StageName 'quality' -Name $Label -PyArgs $pyArgs -Heavy -Done $st.Done `
            -DoneWhy "E2 kld envelope for $Key $Variant against $(Get-RelArg $Logits) present" -Supersede $sup)
}

function Invoke-QaStep([string]$Spec, [string]$Key, [string]$Variant, [string]$OutFile, [string]$SetPath, [string]$PlainRef, [string]$Label) {
    $exp = Get-QaExpected $SetPath
    $done = Test-QaDone $OutFile $exp
    $bak = @()
    if (Test-Path -LiteralPath $OutFile -PathType Leaf) { $bak = @($OutFile) }   # fixed file name: keep a copy before it is rewritten
    return (Invoke-Step -StageName 'quality' -Name $Label -PyArgs (Get-QaArgs $Spec $Variant $OutFile $SetPath $PlainRef) -Heavy `
            -Done $done -DoneWhy "$(Get-RelArg $OutFile) present for the current Q&A set" -BackupFirst $bak)
}

function Get-DownloadHelperText {
    return @'
# pc/desktop_bench.ps1 download helper: written to a temp file at run time and deleted afterwards.
# usage: helper.py REPO FILE REVISION|- DEST_DIR BYTES|- SHA256|-   (run from the repo root)
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(16 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main(argv):
    if len(argv) != 6:
        print("usage: helper.py REPO FILE REVISION DEST_DIR BYTES SHA256", file=sys.stderr)
        return 2
    repo, fname, revision, dest_dir, exp_bytes, exp_sha = argv
    revision = None if revision in ("", "-") else revision
    exp_bytes = None if exp_bytes in ("", "-") else int(exp_bytes)
    exp_sha = None if exp_sha in ("", "-") else exp_sha.lower()
    sys.path.insert(0, os.getcwd())
    token = None
    try:
        from pigemma.gguf_map import hf_token
        token = hf_token()
    except Exception:
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_TOKEN") or None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print("desktop_bench download: huggingface_hub is not importable (%s)" % e, file=sys.stderr)
        return 3
    t0 = time.time()
    try:
        path = hf_hub_download(repo_id=repo, filename=fname, revision=revision, local_dir=dest_dir, token=token)
    except Exception as e:
        msg = str(e)
        if token:
            msg = msg.replace(token, "<token>")
        print("desktop_bench download: FAILED: %s: %s" % (type(e).__name__, msg), file=sys.stderr)
        return 1
    p = Path(path)
    rec = {"path": str(p), "bytes": p.stat().st_size, "download_s": round(time.time() - t0, 1),
           "repo": repo, "file": fname, "revision": revision, "token_used": bool(token)}
    if exp_bytes is not None and rec["bytes"] != exp_bytes:
        rec["error"] = "size %d != catalog %d" % (rec["bytes"], exp_bytes)
        print(json.dumps(rec))
        return 1
    if exp_sha:
        t1 = time.time()
        got = sha256_file(p)
        rec["sha256_s"] = round(time.time() - t1, 1)
        rec["sha256_verified"] = got == exp_sha
        if got != exp_sha:
            rec["error"] = "sha256 %s != catalog %s" % (got, exp_sha)
            print(json.dumps(rec))
            return 1
    print(json.dumps(rec))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
'@
}

function Get-DownloadHelper {
    if (-not $script:HelperPath) {
        $p = Join-Path ([IO.Path]::GetTempPath()) "pigemma_desktop_bench_dl_$PID.py"
        [IO.File]::WriteAllText($p, (Get-DownloadHelperText), $script:Utf8NoBom)
        $script:HelperPath = $p
    }
    return $script:HelperPath
}

function Invoke-LowerQuant([object]$M, [string]$Logits) {
    $cfg = Get-ModelCfg $M
    $lq = $cfg.LowerQuant
    if (-not $lq) { return }
    $label = "lower quant $lq (vs $($M.Key))"
    $entry = Get-CatalogModel $lq
    if ($null -eq $entry) { $null = Add-Skip 'quality' $label 'failed' "'$lq' is not in pigemma/data/catalog.json"; return }
    $bytes = Get-Prop $entry 'bytes'
    $cands = @(Get-CatalogCandidates $entry)
    $path = $null
    foreach ($c in $cands) { if (Test-Path -LiteralPath $c -PathType Leaf) { $path = $c; break } }
    $downloaded = $false
    $repo = [string](Get-Prop $entry 'repo')
    $file = [string](Get-Prop $entry 'file')
    $base = Join-Path $script:RepoRoot 'models'
    if ($env:PIGEMMA_MODELS_DIR) { $base = $env:PIGEMMA_MODELS_DIR }
    $destDir = Join-Path $base ($repo.Replace('/', '__'))
    $dest = Join-Path $destDir ($file.Replace('/', '\'))
    if ($null -ne $path -and $null -ne $bytes -and (Get-Item -LiteralPath $path).Length -ne [int64]$bytes) {
        $null = Add-Skip 'quality' "download $lq" 'failed' "$path exists with $((Get-Item -LiteralPath $path).Length) B, catalog says $bytes B (not touched; delete or fix it)"
        return
    }
    if ($null -eq $path) {
        $free = Get-FreeBytes $destDir
        $need = $null
        if ($null -ne $bytes) {
            $need = [int64][Math]::Max([double]$bytes * $script:DOWNLOAD_DISK_FACTOR, [double]$bytes + $script:DOWNLOAD_KEEP_FREE)
        }
        $okDisk = ($null -ne $free -and $null -ne $need -and $free -ge $need)
        $rev = [string](Get-Prop $entry 'revision'); if (-not $rev) { $rev = '-' }
        $sha = [string](Get-Prop $entry 'sha256'); if (-not $sha) { $sha = '-' }
        $bs = '-'; if ($null -ne $bytes) { $bs = [string]$bytes }
        $diskNote = "needs >= $(Fmt-GB $need) free (max of 1.3 x $(Fmt-GB $bytes) and size + 4 GB) at $destDir; free now $(Fmt-GB $free)"
        if (-not $okDisk) {
            if ($DryRun) { Write-Line ''; Write-Line "== [quality] download $lq"; Write-Line "   NOT ENOUGH DISK: $diskNote" }
            $null = Add-Skip 'quality' "download $lq" 'skipped' "not enough free disk: $diskNote"
            return
        }
        $helperArgs = @('<download helper>', $repo, $file, $rev, (Get-RelArg $destDir), $bs, $sha)
        if (-not $DryRun) { $helperArgs[0] = Get-DownloadHelper }
        $rec = Invoke-Step -StageName 'quality' -Name "download $lq ($(Fmt-GB $bytes)) from $repo" -PyArgs $helperArgs -Heavy -NoInnerDryRun `
            -Note "huggingface_hub, revision $rev, size + sha256 verified against the catalog; $diskNote"
        if ($DryRun) { $path = $dest }
        elseif ($rec.Status -ne 'done') { return }
        else {
            $path = $dest
            foreach ($l in @($rec.Lines)) {
                try { $o = $l | ConvertFrom-Json; if ($o -and (Get-Prop $o 'path')) { $path = [string](Get-Prop $o 'path') } } catch { }
            }
            $downloaded = $true
        }
    }
    $r1 = Invoke-KldStep $M $lq $lq $lq $Logits "kld $lq vs $($M.Key) reference"
    $qaFile = Join-Path $script:Opt.QualityDirAbs "qa_${lq}_plain.json"
    $r2 = Invoke-QaStep $lq $lq 'plain' $qaFile $script:QA_SET $null "qa $lq plain"
    if ($downloaded -and -not $KeepDownloads) {
        if (@('done', 'skipped') -contains $r1.Status -and @('done', 'skipped') -contains $r2.Status) {
            Remove-Item -LiteralPath $path -Force
            Write-Note "deleted the downloaded $lq ($path) to free disk (-KeepDownloads keeps it)"
        } else {
            Write-Warn "kept ${path}: its kld/qa steps did not both finish (rerun resumes without a new download)"
        }
    }
}

function Invoke-StageQuality {
    Write-Line ''
    Write-Line '#### Stage quality (E2): corpus, KLD reference + variants, Q&A'
    # corpus
    $chk = Invoke-Py @('-m', 'pigemma.quality', 'corpus', '--check')
    $upToDate = ($chk.Rc -eq 0)
    $null = Invoke-Step -StageName 'quality' -Name "corpus $($script:CORPUS)" -PyArgs @('-m', 'pigemma.quality', 'corpus') `
        -Done $upToDate -DoneWhy "$($script:CORPUS) is up to date" -NoInnerDryRun
    $script:CorpusSha = Get-FileSha256 (Join-Path $script:RepoRoot ($script:CORPUS.Replace('/', '\')))
    $resArg = Get-DirArg $script:Opt.ResultsDirAbs
    foreach ($m in $script:ModelInfos) {
        if (-not $m.Present) {
            $null = Add-Skip 'quality' "kld/qa $($m.Key)" 'missing' "model file not found: $($m.Path)"
            continue
        }
        $logits = Join-Path $script:Opt.QualityDirAbs "ref_$($m.Key).logits"
        $done = Test-KldBaseDone $logits $m
        $pyArgs = @('-m', 'pigemma.quality', 'kld-base', '--model', $m.Spec, '--text', $script:CORPUS, '-c', [string]$script:KLD_CTX,
                    '--chunks', [string]$script:Opt.KldChunks, '--threads', [string]$RunThreads, '--out', (Get-RelArg $logits),
                    '--results-dir', $resArg, '--experiment', 'E2') + (Get-LockArgs) + (Get-TimeoutArgs)
        if ($Force) { $pyArgs += '--force' }
        $null = Invoke-Step -StageName 'quality' -Name "kld-base $($m.Key) ($($script:Opt.KldChunks) x $($script:KLD_CTX) tokens)" -PyArgs $pyArgs -Heavy `
            -Done $done -DoneWhy "valid reference $(Get-RelArg $logits) for the current corpus" `
            -Note 'disk-gated by quality.py (needs 2 x the 1.61 GB estimate free)'
        foreach ($v in @('norepack', 'hybrid_negctrl')) {
            $null = Invoke-KldStep $m $m.Spec $m.Key $v $logits "kld $($m.Key) $v"
        }
        $plainFile = Join-Path $script:Opt.QualityDirAbs "qa_$($m.Key)_plain.json"
        $null = Invoke-QaStep $m.Spec $m.Key 'plain' $plainFile $script:QA_SET $null "qa $($m.Key) plain"
        $envs = @(Get-Envelopes $script:Opt.ResultsDirAbs)
        $active = @(Get-E1Envs $envs $m.Key | Where-Object { $_.ServerVariant -ne 'plain' -and $_.SpecActive -eq $true } |
                ForEach-Object { $_.ServerVariant } | Select-Object -Unique)
        if ($active.Count -eq 0) {
            Write-Line ''
            Write-Line "   qa $($m.Key): no E1 variant was spec_active in $(Get-DirArg $script:Opt.ResultsDirAbs) (run -Stage spec first, or none drafted): plain only"
        }
        foreach ($v in $active) {
            $f = Join-Path $script:Opt.QualityDirAbs "qa_$($m.Key)_$v.json"
            $null = Invoke-QaStep $m.Spec $m.Key $v $f $script:QA_SET $plainFile "qa $($m.Key) $v (E1: spec_active)"
        }
        if ($QaHard -and -not $Quick) {
            $hf = Join-Path (Join-Path $script:Opt.QualityDirAbs 'hard') "qa_$($m.Key)_plain.json"
            $null = Invoke-QaStep $m.Spec $m.Key 'plain' $hf $script:QA_HARD $null "qa-hard $($m.Key) plain"
        }
        if ($AllowDownload -and -not $Quick) { Invoke-LowerQuant $m $logits }
        elseif (-not $Quick) {
            $lq = (Get-ModelCfg $m).LowerQuant
            if ($lq) { Write-Line "   (lower quant $lq not measured: pass -AllowDownload; it is downloaded only when absent and disk allows)" }
        }
    }
}

# ------------------------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------------------------
function Invoke-Summary {
    if ($NoSummary) { return }
    $inputs = New-Object System.Collections.ArrayList
    if ($DryRun -or (Test-Path -LiteralPath $script:Opt.ResultsDirAbs -PathType Container)) {
        [void]$inputs.Add((Get-RelArg $script:Opt.ResultsDirAbs))
    }
    $resKey = (Get-PathKey $script:Opt.ResultsDirAbs).TrimEnd('\') + '\'
    $qaDirs = @($script:Opt.QualityDirAbs, (Join-Path $script:Opt.QualityDirAbs 'hard'))
    $keys = New-Object System.Collections.ArrayList
    foreach ($m in $script:ModelInfos) {
        [void]$keys.Add($m.Key)
        $lq = (Get-ModelCfg $m).LowerQuant
        if ($lq -and $AllowDownload) { [void]$keys.Add($lq) }
    }
    foreach ($d in $qaDirs) {
        if (-not (Test-Path -LiteralPath $d -PathType Container)) { continue }
        if ((Get-PathKey $d).StartsWith($resKey)) { continue }   # already inside the results dir
        foreach ($k in $keys) {
            foreach ($f in @(Get-ChildItem -LiteralPath $d -File -Filter "qa_$($k)_*.json" -ErrorAction SilentlyContinue)) {
                if ($f.Name -like '*.json') { [void]$inputs.Add((Get-RelArg $f.FullName)) }
            }
        }
    }
    if ($inputs.Count -eq 0) {
        $null = Add-Skip 'summary' 'results summarize' 'skipped' 'no results directory yet'
        return
    }
    $title = "Desktop results $($script:Opt.Date)"
    if ($Quick) { $title += ' (quick)' }
    $pyArgs = @('-m', 'pigemma.results', 'summarize') + $inputs.ToArray() + @('--md', (Get-RelArg $script:Opt.SummaryMdAbs), '--title', $title)
    $null = Invoke-Step -StageName 'summary' -Name "results summarize -> $(Get-RelArg $script:Opt.SummaryMdAbs)" -PyArgs $pyArgs -NoInnerDryRun -IgnoreTimeBox
}

# ------------------------------------------------------------------------------------------------
# Usage, options, main
# ------------------------------------------------------------------------------------------------
function Show-Usage {
    Write-Host @"
$($script:ToolName) $($script:ToolVersion) - desktop driver for E0 (bench), E1 (spec) and E2 (quality)

usage: .\pc\desktop_bench.ps1 -Stage bench,spec,quality [-Models 26b,31b] [-Threads 4,6,12] [-Quick]
                              [-Force] [-AllowDownload] [-DryRun]
       [-Date yyyy-MM-dd] [-OutDir DIR] [-QualityDir DIR] [-SummaryMd FILE] [-MaxMinutes N]
       [-StepTimeoutMin 60] [-RequestTimeoutS 900] [-RunThreads 6] [-CompletionTokens 64] [-Port 18080]
       [-SpecRounds 2] [-SpecTokens 200] [-SpecPrompts N] [-DraftNMax N] [-QaTokens 160] [-KldChunks 6]
       [-MinFreeGB 20] [-IdleCheckS 10] [-IdleMaxPct 5] [-IdleWaitS N] [-RequireIdle]
       [-LowerQuant26 26b-ud-q3_k_m] [-LowerQuant31 31b-ud-iq3_xxs] [-KeepDownloads] [-QaHard]
       [-NoMembw] [-NoSummary] [-Python EXE] [-Help] [-Version]

  -Stage      bench (E0) | spec (E1) | quality (E2) | all; comma list; run order is always bench, spec, quality
  -Models     26b, 31b (default both), a catalog id or a .gguf path
  -Threads    llama-bench thread list (default 26B 4,6,12; 31B 6,12; -Quick 6)
  -Quick      smoke run into results/<date>-quick/ (1 rep, 6 threads, short E1, 2 KLD chunks, 6 QA items)
  -Force      re-run steps whose results exist (earlier envelopes become *.superseded)
  -AllowDownload  E2 also measures 26b-ud-q3_k_m / 31b-ud-iq3_xxs (download only if free disk >= max(1.3 x size,
              size + 4 GB); deleted again after its kld + qa unless -KeepDownloads)
  -DryRun     print every command (plus the llama.cpp command lines) and write nothing
  -MaxMinutes time box: no new step starts after N minutes (exit 5; rerun to resume)

Results: results/<date>/ (E0_*.json, E0_GATE.md, membw_desktop.json, E1_*.json, E1_SUMMARY.md, E2_*.json),
results/quality/ (ref_<model>.logits, qa_<model>_<variant>.json), docs/results/DESKTOP_<date>.md.
Exit codes: 0 ok, 1 step failed, 2 usage, 3 missing input, 4 refused (stops the run),
5 time box reached, 6 E0 repack gate FAIL/INCOMPLETE. Full help: Get-Help .\pc\desktop_bench.ps1 -Full
"@
}

function Initialize-Options {
    # returns an error message (usage error) or $null
    if (@($Rest | Where-Object { $_ }).Count -gt 0) { return "unexpected argument(s): $(@($Rest) -join ' ')" }
    $stages = @(Split-List $Stage | ForEach-Object { $_.ToLowerInvariant() })
    if ($stages.Count -eq 0) { return '-Stage is required (bench, spec, quality)' }
    $set = @{}
    foreach ($s in $stages) {
        switch ($s) {
            'all' { $set['bench'] = $true; $set['spec'] = $true; $set['quality'] = $true }
            { $_ -in @('bench', 'e0') } { $set['bench'] = $true }
            { $_ -in @('spec', 'e1') } { $set['spec'] = $true }
            { $_ -in @('quality', 'e2') } { $set['quality'] = $true }
            default { return "unknown stage '$s' (bench, spec, quality)" }
        }
    }
    $script:Opt.Stages = @(@('bench', 'spec', 'quality') | Where-Object { $set.ContainsKey($_) })
    $script:Opt.ModelSpecs = @(Split-List $Models)
    if ($script:Opt.ModelSpecs.Count -eq 0) { return '-Models is empty' }
    $tl = New-Object System.Collections.ArrayList
    foreach ($t in @(Split-List $Threads)) {
        if ($t -notmatch '^\d+$' -or [int]$t -lt 1 -or [int]$t -gt 1024) { return "bad -Threads value '$t' (positive integers)" }
        [void]$tl.Add([int]$t)
    }
    $script:Opt.Threads = $tl.ToArray()
    if ($Date) {
        if ($Date -notmatch '^\d{4}-\d{2}-\d{2}$') { return "bad -Date '$Date' (yyyy-MM-dd)" }
        $script:Opt.Date = $Date
    } else {
        $script:Opt.Date = (Get-Date).ToString('yyyy-MM-dd', $script:Inv)
    }
    if ($RunThreads -lt 1) { return '-RunThreads must be >= 1' }
    if ($CompletionTokens -lt 1) { return '-CompletionTokens must be >= 1' }
    if ($Port -lt 1 -or $Port -gt 65535) { return '-Port must be 1..65535' }
    if ($SpecRounds -lt 1) { return '-SpecRounds must be >= 1' }
    if ($SpecTokens -lt 0) { return '-SpecTokens must be >= 0 (0 = each prompt''s max_tokens)' }
    if ($SpecPrompts -lt 0) { return '-SpecPrompts must be >= 0 (0 = all)' }
    if ($DraftNMax -lt 0) { return '-DraftNMax must be >= 0' }
    if ($QaTokens -lt 1) { return '-QaTokens must be >= 1' }
    if ($KldChunks -lt 1) { return '-KldChunks must be >= 1' }
    if ($MaxMinutes -lt 0) { return '-MaxMinutes must be >= 0 (0 = no time box)' }
    if ($StepTimeoutMin -le 0) { return '-StepTimeoutMin must be > 0' }
    if ($RequestTimeoutS -lt 1) { return '-RequestTimeoutS must be >= 1' }
    if ($MinFreeGB -lt 0) { return '-MinFreeGB must be >= 0' }
    if ($IdleCheckS -lt 0 -or $IdleWaitS -lt 0 -or $IdleMaxPct -lt 0) { return 'idle options must be >= 0' }
    $q = [bool]$Quick
    $script:Opt.SpecRounds = $SpecRounds
    if ($q -and -not $script:Bound.ContainsKey('SpecRounds')) { $script:Opt.SpecRounds = 1 }
    $script:Opt.SpecPrompts = $SpecPrompts
    if ($q -and -not $script:Bound.ContainsKey('SpecPrompts')) { $script:Opt.SpecPrompts = 2 }
    $script:Opt.SpecTokens = $SpecTokens
    if ($q -and -not $script:Bound.ContainsKey('SpecTokens')) { $script:Opt.SpecTokens = 64 }
    $script:Opt.KldChunks = $KldChunks
    if ($q -and -not $script:Bound.ContainsKey('KldChunks')) { $script:Opt.KldChunks = 2 }
    $script:Opt.QaIds = ''
    if ($q) { $script:Opt.QaIds = $script:QUICK_QA_IDS }
    return $null
}

function Initialize-Paths {
    $suffix = ''
    if ($Quick) { $suffix = '-quick' }
    if ($OutDir) { $script:Opt.ResultsDirAbs = Get-FullPathFrom $script:OrigCwd $OutDir }
    else { $script:Opt.ResultsDirAbs = Join-Path $script:RepoRoot "results\$($script:Opt.Date)$suffix" }
    if ($QualityDir) { $script:Opt.QualityDirAbs = Get-FullPathFrom $script:OrigCwd $QualityDir }
    elseif ($Quick) { $script:Opt.QualityDirAbs = Join-Path $script:Opt.ResultsDirAbs 'quality' }
    else { $script:Opt.QualityDirAbs = Join-Path $script:RepoRoot 'results\quality' }
    if ($SummaryMd) { $script:Opt.SummaryMdAbs = Get-FullPathFrom $script:OrigCwd $SummaryMd }
    else { $script:Opt.SummaryMdAbs = Join-Path $script:RepoRoot "docs\results\DESKTOP_$($script:Opt.Date)$suffix.md" }
    $script:Opt.ModelsDirAbs = Join-Path $script:RepoRoot 'models'
    if ($env:PIGEMMA_MODELS_DIR) { $script:Opt.ModelsDirAbs = $env:PIGEMMA_MODELS_DIR }
    $n = $null
    $gp = Read-JsonFile (Join-Path $script:RepoRoot ($script:GEN_PROMPTS.Replace('/', '\')))
    if ($null -ne $gp) { $n = @($gp).Count }
    if ($script:Opt.SpecPrompts -gt 0 -and ($null -eq $n -or $script:Opt.SpecPrompts -lt $n)) { $n = $script:Opt.SpecPrompts }
    $script:Opt.SpecPromptCount = $n
}

function Get-FinalCode {
    $st = @($script:StepLog | ForEach-Object { $_.Status })
    if ($st -contains 'refused') { return $script:EXIT_REFUSED }
    if ($st -contains 'missing' -or $script:UsageMissing) { return $script:EXIT_MISSING }
    if ($st -contains 'failed') { return $script:EXIT_ERROR }
    if ($script:GateFailed) { return $script:EXIT_GATE }
    if ($script:TimeBoxHit) { return $script:EXIT_TIMEBOX }
    return $script:EXIT_OK
}

function Show-Final([int]$Code) {
    Write-Line ''
    Write-Line '#### desktop_bench summary'
    Write-Line ('  {0,-8} {1,-9} {2,-5} {3,8}  {4}' -f 'stage', 'status', 'rc', 'seconds', 'step')
    foreach ($r in $script:StepLog) {
        $rc = ''; if ($null -ne $r.Rc) { $rc = [string]$r.Rc }
        Write-Line ('  {0,-8} {1,-9} {2,-5} {3,8}  {4}' -f $r.Stage, $r.Status, $rc, (Fmt $r.Seconds 1), $r.Name)
    }
    $meaning = @{
        0 = 'ok'; 1 = 'a step failed'; 2 = 'usage error'; 3 = 'missing input'
        4 = 'refused (stopped)'; 5 = 'time box reached (rerun to resume)'; 6 = 'E0 repack gate not PASS'
    }
    if ($script:Stopped -and $script:StopReason) { Write-Line "  stopped: $($script:StopReason)" }
    Write-Line "  elapsed $(Format-Elapsed); exit $Code ($($meaning[$Code]))"
    Add-Log "exit $Code ($($meaning[$Code]))"
}

function Main {
    if ($Version) { Write-Host "$($script:ToolName) $($script:ToolVersion)"; $script:FinalCode = 0; return }
    if ($Help) { Show-Usage; $script:FinalCode = 0; return }
    $err = Initialize-Options
    if ($err) {
        Show-Usage
        Write-Host "desktop_bench: usage error: $err"
        $script:FinalCode = $script:EXIT_USAGE
        return
    }
    $script:OrigCwd = (Get-Location).ProviderPath
    $script:RepoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
    if (-not (Test-Path -LiteralPath (Join-Path $script:RepoRoot 'pigemma\bench.py'))) {
        Write-Host "desktop_bench: ERROR: $($script:RepoRoot) does not look like the pi-gemma4 repo (no pigemma\bench.py)"
        $script:FinalCode = $script:EXIT_MISSING
        return
    }
    Push-Location -LiteralPath $script:RepoRoot
    try {
        $resolved = New-Object System.Collections.ArrayList
        try {
            foreach ($s in $script:Opt.ModelSpecs) {
                $mi = Resolve-ModelSpec $s
                if (@($resolved | Where-Object { $_.Key -eq $mi.Key }).Count -eq 0) { [void]$resolved.Add($mi) }
            }
        } catch {
            Show-Usage
            Write-Host "desktop_bench: usage error: $($_.Exception.Message)"
            $script:FinalCode = $script:EXIT_USAGE
            return
        }
        # 26B family first: the 31B-dependent steps are scheduled last (manifest run order)
        $script:ModelInfos = @(@($resolved | Where-Object { $_.Family -ne '31b' }) + @($resolved | Where-Object { $_.Family -eq '31b' }))
        Initialize-Paths
        if ($MaxMinutes -gt 0) { $script:Deadline = $script:StartTime.AddMinutes($MaxMinutes) }

        if (-not (Find-Python)) {
            Write-Host 'desktop_bench: MISSING: no Python >= 3.10 found (python, python3, py -3; or -Python / PIGEMMA_PYTHON)'
            $script:FinalCode = $script:EXIT_MISSING
            return
        }
        $env:PYTHONUTF8 = '1'
        $env:PYTHONIOENCODING = 'utf-8'

        if (-not $DryRun) {
            $raw = Join-Path $script:Opt.ResultsDirAbs 'raw'
            if (-not (Test-Path -LiteralPath $raw)) { $null = New-Item -ItemType Directory -Path $raw -Force }
            $script:LogPath = Join-Path $raw ("desktop_bench_" + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ', $script:Inv) + '.log')
        }
        $mode = 'RUN'
        if ($DryRun) { $mode = 'DRY RUN (nothing is run or written)' }
        Write-Line "$($script:ToolName) $($script:ToolVersion) - $mode - $([DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ', $script:Inv))"
        Add-Log "start: $($script:ToolName) $($script:ToolVersion) argv: $($MyInvocation.ScriptName) $(@($script:Bound.Keys | ForEach-Object { "-$_ $($script:Bound[$_])" }) -join ' ')"
        Write-Line "  repo       $($script:RepoRoot)"
        Write-Line "  python     $(Format-Cmd @())"
        Write-Line "  stages     $($script:Opt.Stages -join ', ')"
        foreach ($m in $script:ModelInfos) {
            $state = 'present'; if (-not $m.Present) { $state = 'MISSING' }
            Write-Line ("  model      {0,-16} {1} ({2}, {3})" -f $m.Key, (Get-RelArg $m.Path), (Fmt-GB $m.Bytes), $state)
        }
        Write-Line "  results    $(Get-DirArg $script:Opt.ResultsDirAbs)"
        Write-Line "  quality    $(Get-DirArg $script:Opt.QualityDirAbs)"
        Write-Line "  summary    $(Get-RelArg $script:Opt.SummaryMdAbs)"
        $tb = 'none'; if ($MaxMinutes -gt 0) { $tb = "$(Fmt $MaxMinutes 1) min (no new step after that)" }
        Write-Line "  time box   $tb; per-step timeout $(Fmt $StepTimeoutMin 0) min; request timeout $RequestTimeoutS s"
        Write-Line "  options    quick=$([bool]$Quick) force=$([bool]$Force) allow-download=$([bool]$AllowDownload) threads=$RunThreads E1 rounds=$($script:Opt.SpecRounds) tokens=$($script:Opt.SpecTokens) prompts=$($script:Opt.SpecPromptCount) KLD chunks=$($script:Opt.KldChunks)"
        if ($script:LogPath) { Write-Line "  driver log $(Get-RelArg $script:LogPath)" }
        Show-FreeDisk
        foreach ($m in $script:ModelInfos) {
            if (-not $m.Present) {
                if ($DryRun) { Write-Warn "model $($m.Key) is missing at $($m.Path): its steps are listed but would be skipped" }
                else { Write-Warn "model $($m.Key) is missing at $($m.Path): its steps are skipped (exit 3 at the end)" }
            }
        }
        if (@($script:ModelInfos | Where-Object { -not $_.Present }).Count -gt 0) { $script:UsageMissing = $true }

        # bench lock pre-check: never start while another model job holds results/.bench.lock
        $ls = Invoke-Py @('-m', 'pigemma.paths', 'lock', '--status')
        $lock = $null
        try { $lock = ((@($ls.Lines) -join "`n") | ConvertFrom-Json) } catch { $lock = $null }
        $lstate = [string](Get-Prop $lock 'state')
        Write-Line "  bench lock $lstate $(if ($lstate -eq 'held') { '(holder: ' + [string](Get-Prop $lock 'holder.cmd') + ', pid ' + [string](Get-Prop $lock 'holder.pid') + ')' })"
        if ($lstate -eq 'held' -and -not $DryRun) {
            Write-Host "desktop_bench: REFUSED: $([string](Get-Prop $lock 'path')) is held by another model-loading job; wait for it (python -m pigemma.paths lock --status)"
            $null = Add-StepRecord 'start' 'bench lock' 'refused' 4 0 'lock held'
            $script:FinalCode = $script:EXIT_REFUSED
            Show-Final $script:FinalCode
            return
        }

        if ($script:Opt.Stages -contains 'bench') { Invoke-StageBench }
        if ($script:Opt.Stages -contains 'spec') { Invoke-StageSpec }
        if ($script:Opt.Stages -contains 'quality') { Invoke-StageQuality }
        $wasStopped = $script:Stopped
        # the summary renders whatever exists, even after a refusal or the time box
        $script:Stopped = $false
        Invoke-Summary
        $script:Stopped = $wasStopped
        $script:FinalCode = Get-FinalCode
        Show-Final $script:FinalCode
    } finally {
        Pop-Location
    }
}

# ------------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------------
$savedEnv = @{ PYTHONUTF8 = $env:PYTHONUTF8; PYTHONIOENCODING = $env:PYTHONIOENCODING }
$savedOutEnc = $null
try { $savedOutEnc = [Console]::OutputEncoding; [Console]::OutputEncoding = $script:Utf8NoBom } catch { $savedOutEnc = $null }
try {
    Main
} catch {
    Write-Host "desktop_bench: ERROR: $($_.Exception.Message)"
    Write-Host ($_.ScriptStackTrace)
    Add-Log "ERROR: $($_.Exception.Message)"
    $script:FinalCode = $script:EXIT_ERROR
} finally {
    if ($script:HelperPath -and (Test-Path -LiteralPath $script:HelperPath)) { Remove-Item -LiteralPath $script:HelperPath -Force -ErrorAction SilentlyContinue }
    $env:PYTHONUTF8 = $savedEnv.PYTHONUTF8
    $env:PYTHONIOENCODING = $savedEnv.PYTHONIOENCODING
    if ($null -ne $savedOutEnc) { try { [Console]::OutputEncoding = $savedOutEnc } catch { } }
}
exit $script:FinalCode
