export const meta = {
  name: 'gemma4-pi-experiments',
  description: 'Run the experiments on the desktop CPU and the real Pi 5 16 GB, fix the scripts as they break, write results, then adversarially review and fix',
  phases: [
    { title: 'Experiments', detail: 'E0 alone; E3+E8; E4+E7; real-Pi E9 alongside desktop E1/E2' },
    { title: 'Write-up', detail: 'results docs, README status table with measured/projected/assumed' },
    { title: 'Review', detail: 'adversarial reviewers by lens, then fixers' },
  ],
}

const REPO = 'REPO_ROOT'
const BUILD_REPORT = args && args.build_report ? args.build_report : '(see results of the build workflow)'

const CONTEXT = `# CONTEXT
Read REPO_ROOT\\docs\\design\\FACTS.md completely first (verified facts, measured numbers, emulated-Pi setup, hardware options, process rules). The experiment definitions (procedure, metrics, PASS thresholds, runtime estimates) are the experiments[] array in docs/design/manifest.json; the code is the repo itself (README.md, docs/*.md, pigemma/, tools/, pc/, pi/). Build/test status from the implementation phase:
${BUILD_REPORT}

# RULES
- You run ONE experiment stage. You may edit any repo file (not third_party/, not models/) to fix bugs you hit; keep fixes minimal, re-run the affected unit tests (python -m pytest -q from the repo root) after each fix, and list every fix.
- CPU discipline: only the stage you are in loads a big model. Before loading a model check that no other llama/probe process or model container is running (tasklist, docker ps) and respect results/.bench.lock if the code implements it. Use the thread counts the experiment specifies (desktop default 6; the MoE is fastest at 4).
- Disk discipline: print free space on D: and C: before writing anything large; never exceed the gates in the manifest; never prune Docker caches, images or volumes you did not create; remove temporary large files you created when the stage is done unless the manifest says to keep them.
- Never touch context-friend-1. Never kill processes you did not start (a container named contextfriend-box-stt belongs to the operator - leave it alone).
- Honesty: report only numbers you measured in this stage or read from result files; mark projections/assumptions as such. If a PASS threshold fails, report FAIL with the numbers - do not tune thresholds to pass. If something is genuinely blocked, report BLOCKED with the exact reason and what would unblock it.
- Write machine-readable results where the manifest says (results/...json, results envelope) and a short human summary docs/results/<EXPERIMENT>.md (what was run, exact commands, table of measured numbers, verdict vs threshold, caveats).
- No git commits. If a Bash heredoc fails with "unexpected EOF while looking for matching quote", use the Write tool.`

const STAGE = {
  type: 'object',
  properties: {
    experiment: { type: 'string' },
    status: { type: 'string', enum: ['PASS', 'FAIL', 'PARTIAL', 'BLOCKED'] },
    measured: { type: 'object' },
    verdict_vs_threshold: { type: 'string' },
    result_files: { type: 'array', items: { type: 'string' } },
    fixes: { type: 'array', items: { type: 'string' } },
    problems: { type: 'array', items: { type: 'string' } },
    wall_minutes: { type: 'number' },
    summary: { type: 'string' },
  },
  required: ['experiment', 'status', 'measured', 'verdict_vs_threshold', 'result_files', 'fixes', 'problems', 'summary'],
}

const run = (id, task, extra) => agent(`${CONTEXT}\n\n# YOUR STAGE: ${id}\n${task}\n${extra || ''}\n\nReturn the structured stage report.`,
  { label: `exp:${id}`, phase: 'Experiments', schema: STAGE })

const stages = []
const record = r => { if (r) { stages.push(r); log(`${r.experiment}: ${r.status} - ${(r.summary || '').slice(0, 160)}`) } return r }
const prior = () => stages.map(s => `- ${s.experiment} [${s.status}]: ${s.summary}\n  measured: ${JSON.stringify(s.measured).slice(0, 1500)}\n  files: ${(s.result_files || []).join(', ')}`).join('\n')

phase('Experiments')

// E0 is a timing benchmark: run it alone on an otherwise idle CPU.
const e0 = record(await run('E0-desktop-baseline-and-repack-gate', 'Run experiment E0 exactly as defined in the manifest (desktop, native Windows binaries; both models; default repack vs -nr streaming mode vs the hybrid negative control; buffer-split assertion from the load log; thread sweep only where the manifest says - the idle-machine sweep already exists in results/desktop-baseline-2026-10-04.md, reuse it rather than repeating it). This stage decides whether -nr is a viable Pi mode: report the -nr penalty per model precisely. Nothing else runs on the CPU during this stage.'))

// E3 (trace content is timing-independent) alongside the light E8 dry run.
const [e3, e8] = await parallel([
  () => run('E3-routing-trace', `Run experiment E3: build the probe if needed (bash pc/docker_run.sh build-tools), then collect the 26B-A4B expert-routing trace over the whole calibration corpus data/calibration/* (prefill/teacher-forced; plus the generation prompts if the manifest's E3 includes them) with pc/trace_collect.sh, validate it with python -m pigemma.trace validate, and produce the stats (expert frequency skew, next-token reuse probability vs chance, per-layer entropy, cross-layer predictability, LRU stack-distance curve). Keep the trace under traces/ (gitignored) and the stats JSON under results/h1/. A light E8 dry-run stage runs concurrently (no model loads).
Earlier stages:
${prior()}`),
  () => run('E8-pi-runtime-dry-run', 'Run experiment E8 (pytest for feasibility/pi scripts/catalog, bash -n on pi/*.sh and pc/*.sh, pi/00_feasibility.sh and pi/01_tune_os.sh dry run inside the x86 Docker image, pinner integration on a 64 MiB synthetic file in Docker). Do NOT load any big model (E3 runs concurrently and owns the model). Do not use the arm64 emulated Pi here; that is a later stage.'),
]).then(rs => rs.map(record))

// E4 (pure Python on the trace) can run while E7's heavy act-dump runs.
const [e4, e7] = await parallel([
  () => run('E4-cache-simulation', `Run experiment E4 on the trace from E3: python -m pigemma.cache_sim with all policies (lru, lfu, static, belady, lru_prefetch, lru_perlayer control) across the budget sweep and all Pi device profiles from the catalog, using results/map26.json (or whatever the map path is). Use the E0 norepack calibration for the llama efficiency factor if E0 produced it. Produce results/h1/sim.json (+csv) and docs/results/H1_RESULTS.md with the hit-rate-vs-budget curve table, MB/token, projected tok/s per Pi profile (clearly PROJECTED, with the assumption list), and compare Gemma 4's measured locality against the literature numbers in docs/research/RESEARCH_SUMMARY.md section 3. Do not load any model (E7 runs concurrently).\nEarlier stages:\n${prior()}`),
  () => run('E7-h2-dense-oracle', `Run experiment E7 (hypothesis H2 on the dense 31B) with pc/h2_run.ps1 in the configuration the manifest specifies as the default that fits in about one hour (activation dump in Docker on sampled layers, then decompose + oracle natively). Include the controls the manifest lists (dequant sanity check W_down @ ffn_geglu vs captured ffn_out, iso-byte uniform-quant competitor, bound tightness, teacher-forced caveat). Write results/h2/oracle.json and docs/results/H2_RESULTS.md with: oracle fraction of residual bytes needed at each error threshold, fraction of groups the norm bound certifies as skippable, implied Pi bytes/token and s/token, and a plain verdict on H2.\nEarlier stages:\n${prior()}`),
]).then(rs => rs.map(record))

// E5/E6 (Docker page-cache proxy) and E8b (emulated Pi) were dropped on 2026-10-05: the real Pi 5 16 GB measures the same things on real hardware (stage E9).

// E9 runs on the Pi, E1/E2 on the desktop CPU: different machines, so in parallel.
const [e9, e12] = await parallel([
  () => run('E9-real-pi-context-friend-1', `NEW STAGE (not in the manifest). Work on the REAL target Pi (Raspberry Pi 5 16 GB) - read every UPDATE section of FACTS.md first and follow the safety rules exactly (reach it with ssh -o BatchMode=yes -o ConnectTimeout=20 -o HostKeyAlias=context-friend-1 pi@PI_HOST; nice -n 10; no mlock except the balloon/pinner experiments below which run as root via sudo for a bounded time; keep >= 2 GB free on /; never stop/restart/kill services; never edit /boot/firmware/config.txt; never reboot; the operator authorised testing on this box -> PIGEMMA_I_KNOW=1, record that).
Already measured by the orchestrator (do NOT repeat; read results/pi-real/README.md and results/pi-real/series-26b/*): 31B dense UD-IQ3_XXS 0.53 tok/s gen, 0.63 pp; 26B-A4B UD-Q3_K_M 3.2 tok/s gen, 9.2 pp, zero storage reads after a 148 s load; MTP A/B (code +34 %, EN -2 %, DE -25 %, HE -29 %); balloon memory-pressure baseline 4/6/8 GiB (8 GiB: 0.55-0.79 tok/s, 75-108 MB read per token).
State on the Pi: ~/pigemma/llama.cpp (a7b94df native build), ~/pigemma/models/{gemma-4-26B-A4B-it-UD-Q3_K_M.gguf, mtp-gemma-4-26B-A4B-it-Q4_0.gguf} on the microSD; a 64 GB USB stick (FAT32, 70 MB/s read, 57 MB/s write) mounted by the orchestrator at /mnt/usb (NOT in fstab - remount with: sudo mount -o rw,uid=1000,gid=1000,umask=022,noatime /dev/sda1 /mnt/usb) holding Google's QAT Q4_0 26B as split GGUF parts in /mnt/usb/pigemma/models/26b-qat-q4_0/ (load with -m <first part -00001-of-0000N.gguf>); ad-hoc tools in ~/pigemma (serve_measure.py with --balloon-gib/--extra/--cold, balloon.py, fetch_gguf.sh).
Do, in order:
1. Deploy the repo (pigemma/, pi/, tools/ sources, data/prompts; NOT models/third_party) to ~/pigemma/repo; run pi/00_feasibility.sh (record the verdict), pi/01_tune_os.sh DRY-RUN only, pi/02_build_llama.sh against the existing build (must reuse it; fix the script if not), pi/03_bench_io_mem.sh (fio is not installed - the script must degrade or report; build tools/membw natively), and record the measured device constants as catalog profile 'pi5-16gb-microsd-real' (+ the USB stick as a second storage tier).
2. QAT Q4_0 26B from the USB stick: serve_measure.py cold run (-nr) and, if MemAvailable - 14.44 GB > 1.0 GB, a default-repack run with -c 2048 (watch MemAvailable/swap; abort if MemAvailable < 600 MB), to measure the -nr penalty on ARM and compare QAT Q4_0 vs UD-Q3_K_M speed. Also run the hard QA set data/prompts/qa_hard.json (+ qa_set.json) on BOTH 26B files on the Pi via llama-server (greedy) and score them with the repo's quality code - this is the quality comparison of the two files.
3. Validate the cache simulator against reality: take the E3 trace from the desktop (traces/), run pigemma.cache_sim (LRU, Belady) at the Pi budgets that correspond to the balloon runs (usable page cache = MemAvailable after balloon - llama anon RSS - services) for the UD-Q3_K_M byte map (build its map with gguf_map on the desktop copy if needed - download the GGUF to D: only if D: keeps > 15 GB free, else derive the map from the Pi with gguf_map run there), and compare predicted vs measured read bytes per token. Report the ratio honestly.
4. Hot-set pinning vs plain page cache at the SAME memory budget (the decisive H1 test on real hardware): with balloon 8 GiB (and 6 GiB), run serve_measure on the UD-Q3_K_M model (a) plain and (b) with pi/06_pin_hotset.sh / tools/pinner pinning the non-expert tensors + the hottest experts from the trace into the available budget (leave >= 1.5 GB unpinned for the rest of the page cache), same prompts, interleaved a/b/a/b. Report tok/s and read bytes per token for both. PASS for H1-on-Pi if pinning is >= 1.3x faster than plain page cache at the same budget.
Write results/pi-real/e9.json and docs/results/PI_REAL.md (measured numbers, verdicts, what the scripts got wrong and how you fixed them). Copy raw run directories back to results/pi-real/.\nEarlier stages:\n${prior()}`),
  () => run('E1-E2-speculative-and-quality', `Run E1 (speculative decoding through llama-server via the repo's bench tooling on an IDLE desktop (check total CPU < 5 % first): MTP needs --spec-type draft-mtp explicitly (see FACTS.md); measure baseline vs MTP vs DFlash interleaved A,B,C,A,B,C on the 4 short prompts + code for both models, record acceptance, tokens per verification and tok/s; compare with the Pi A/B in results/pi-real/README.md (on the MoE MTP only pays at very high acceptance)) and E2 (quality: KLD of -nr and the hybrid negative control vs the default Q4_0 run with the disk gate, the 16-item QA set incl. the Hebrew subset via llama-server for both models). Do not download extra quants unless the disk gate passes with margin (>= 1.3x and D: free > 25 GB after the download); if it does not, report the lower-quant KLD as not measured. Write docs/results/E1_E2.md.\nEarlier stages:\n${prior()}`),
]).then(rs => rs.map(record))

phase('Write-up')
const writeup = await agent(`${CONTEXT}\n\n# YOUR STAGE: WRITE-UP\nAll experiment stage reports:\n${prior()}\n\nUpdate README.md (status table MEASURED / PROJECTED / ASSUMED with the real numbers from these stages and results/desktop-baseline-2026-10-04.md; the Hardware options table; quick-start commands verified to match the real CLIs), docs/HYPOTHESES.md (H1 and H2 verdicts against the pre-registered thresholds), docs/EXPERIMENT_PLAN.md (mark which experiments ran, results links), docs/PI_RUNBOOK.md (anything the emulated-Pi stage learned), and write docs/results/SUMMARY.md: one page, what runs today, the measured numbers, the projected Pi numbers per hardware option, what is still unknown until a real Pi is measured, and the honest answer to "can a full Gemma 4 32B run on a Raspberry Pi 5 with good speed and quality". Verify every number you write against a result file; cite the file. Return the structured stage report with experiment='WRITEUP'.`,
  { label: 'writeup', phase: 'Write-up', schema: STAGE })

phase('Review')
const FINDINGS = {
  type: 'object',
  properties: {
    findings: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string' },
          severity: { type: 'string', enum: ['high', 'medium', 'low'] },
          summary: { type: 'string' },
          failure_scenario: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['file', 'severity', 'summary', 'failure_scenario', 'evidence', 'fix'],
      },
    },
    summary: { type: 'string' },
  },
  required: ['findings', 'summary'],
}
const LENSES = [
  'MEASUREMENT CORRECTNESS: trace writer/reader agreement, expert byte-range math (nb[2] per expert, Q4_0 18 B per 32 weights, absolute data_offset), cache simulator semantics and warm-up handling, projection formulas, H2 oracle math (residual groups, norm bounds, error metric, teacher forcing), unit conversions (GB vs GiB, MB/s), and whether each number in docs/results/*.md is actually what the cited result file says.',
  'RASPBERRY PI DEPLOYABILITY: pi/*.sh on real Debian 13 aarch64 with 16 KiB pages (not just the emulated 4 KiB), feasibility gates and exit codes, no destructive or service-affecting commands, deny-listed host, correct llama.cpp flags for Cortex-A76 and the -nr / -lm modes, memlock handling, NVMe gen3 + AI HAT lane conflict, parametric quant selection for 4/8/16 GB, and whether a person with a fresh Pi 5 16 GB + NVMe could follow docs/PI_RUNBOOK.md without guessing.',
  'SCIENTIFIC HONESTY AND CLAIMS: does any doc overstate (speed, quality, "runs on a Pi") beyond what was measured; are emulated numbers labelled; are projections labelled with their assumptions; is the dense-31B reality stated plainly; are FAIL/PARTIAL results reported as such; are the speculative-decoding results described correctly (MTP inactive, DFlash slower); is the Hebrew quality caveat (if any) present.',
]
const reviews = (await parallel(LENSES.map((lens, i) => () =>
  agent(`${CONTEXT}\n\nYou are an adversarial reviewer of the repository at ${REPO}. Read the code, results and docs; run cheap checks (unit tests, python snippets on result files) to confirm or refute; do NOT modify files and do NOT load big models.\nStage reports:\n${prior()}\nWrite-up report: ${JSON.stringify(writeup || {}).slice(0, 3000)}\n\nREVIEW LENS: ${lens}\n\nReport only real, verified defects, each with a concrete failure scenario, the evidence you checked, and a concrete fix. Rank by severity. No padding.`,
    { label: `review:${i + 1}`, phase: 'Review', schema: FINDINGS })
))).filter(Boolean)

const findings = reviews.flatMap(r => r.findings).filter(f => f.severity !== 'low')
log(`${findings.length} medium/high review findings`)
const byFile = {}
for (const f of findings) (byFile[f.file] = byFile[f.file] || []).push(f)
const fixes = (await parallel(Object.entries(byFile).map(([file, fs]) => () =>
  agent(`${CONTEXT}\n\n# FIX THESE VERIFIED REVIEW FINDINGS IN ${file}\n(edit only this file unless a one-line change elsewhere is unavoidable - say so)\n${JSON.stringify(fs, null, 1)}\n\nFirst re-check each finding; if it is wrong, say so and skip it. Otherwise fix, re-run the relevant unit tests, and report. Do not load big models.`,
    { label: `fix:${file}`, phase: 'Review', schema: STAGE })
))).filter(Boolean)

const final = await agent(`${CONTEXT}\n\nFINAL CHECK: run python -m pytest -q from the repo root, bash -n on pi/*.sh and pc/*.sh, and confirm the docs/results/SUMMARY.md numbers still match the result files after the review fixes (${fixes.length} fix agents ran). Report status PASS only if all green.`,
  { label: 'final-check', phase: 'Review', schema: STAGE })

return { stages, writeup, reviews, fixes, final }
