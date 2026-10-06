export const meta = {
  name: 'gemma4-pi-implement',
  description: 'Implement the 36-file pi-gemma4 repo from the design manifest by dependency level, then build the native tools in Docker and make the unit tests pass',
  phases: [
    { title: 'Implement', detail: 'one agent per manifest entry, scheduled by dependency level' },
    { title: 'Build+Test', detail: 'Docker CMake build of probe/pinner/membw, pytest on Windows, fix until green' },
  ],
}

const M = { files: args.files }
const FACTS = `# CONTEXT
BEFORE ANYTHING ELSE read REPO_ROOT\docs\design\FACTS.md completely (verified facts, measured numbers, llama.cpp API details, hardware options, process rules). Then read your own entry in docs/design/manifest.json (files[] where path matches your assignment: purpose, spec, test, depends_on) and the data_formats[] array in the same file (shared contracts: .moetrace, map.json, results envelope, sim.json, catalog.json, facts.json, ...). Those three sources are authoritative; this prompt only points to them.`
const REPO = 'REPO_ROOT'

const FILE_RESULT = {
  type: 'object',
  properties: {
    path: { type: 'string' },
    files_written: { type: 'array', items: { type: 'string' } },
    written: { type: 'boolean' },
    summary: { type: 'string' },
    interface_notes: { type: 'string' },
    self_test: { type: 'string' },
    open_issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['path', 'files_written', 'written', 'summary', 'interface_notes'],
}

const key = p => p.split(' (')[0].trim()
const byPath = {}
for (const f of M.files) byPath[key(f.path)] = f
const resolve = d => {
  const k = key(d)
  if (byPath[k]) return k
  for (const p of Object.keys(byPath)) if (p.endsWith(k) || k.endsWith(p)) return p
  return null
}
const level = {}
const L = (p, stack) => {
  if (level[p] !== undefined) return level[p]
  if (stack.includes(p)) return 0
  const deps = (byPath[p].depends_on || []).map(resolve).filter(d => d && d !== p)
  level[p] = deps.length ? 1 + Math.max(...deps.map(d => L(d, stack.concat([p])))) : 0
  return level[p]
}
for (const p of Object.keys(byPath)) L(p, [])

const manifestIndex = M.files.map(f => `- ${f.path} [${f.language}] (level ${level[key(f.path)]}): ${f.purpose}`).join('\n')
const formats = (M.data_formats || []).map(d => `### ${d.name}\n${d.spec}`).join('\n\n')

const RULES = `
# RULES FOR EVERY IMPLEMENTATION AGENT
- Repository root: ${REPO}. The full design manifest is docs/design/manifest.json (files, data_formats, experiments, risks) and the research numbers are docs/research/RESEARCH_SUMMARY.md - read the parts you need. Write ONLY the file(s) of your assignment (the path plus anything named in its "(+ ...)" suffix, plus its own unit test tests/test_<module>.py when the spec's test field asks for one). Create parent dirs as needed. Never modify third_party/ or models/, never touch other agents' files.
- Read the files you depend on FIRST (they exist already: earlier levels are finished) and match their real interfaces, not just the manifest text. If an earlier file deviates from the manifest, follow the real file and note it in open_issues.
- Python: 3.14 on Windows, 3.13 on the Pi; stdlib + numpy (+ torch lazily, only where the spec says) + gguf (gguf-py, installed) + psutil (installed). Files the spec marks "stdlib only" must import nothing else. Type hints, docstrings, argparse with --help, explicit exit codes, UTF-8 safe output. pathlib everywhere. Run things as "python -m pigemma.<mod>" from the repo root; do NOT pip install anything.
- C/C++: C11/C++17, compiles with gcc/g++ 13-14 on Ubuntu 24.04 (Docker image pi-gemma4-build:latest) and Debian 13 aarch64 (Pi). Check every return code. Build check (Git Bash): MSYS_NO_PATHCONV=1 docker run --rm -v "REPO_ROOT:/work" pi-gemma4-build:latest bash -c 'cmake -S /work/tools -B /tmp/tb -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/tb -j 10 2>&1 | tail -40'. The llama.cpp sources are at /work/third_party/llama.cpp inside the container.
- Bash for the Pi: set -euo pipefail, parametric, prints a verdict and refuses heavy work when RAM/disk are insufficient, never kills/restarts unrelated processes or services, never reboots, apt only behind an explicit flag, works on Debian 13 aarch64 with 16 KiB pages AND inside an arm64 Debian container without systemd/sudo/device-tree/boot config (degrade with a logged warning). Lint with "bash -n" (shellcheck is not installed).
- PowerShell 5.1 for pc/*.ps1 (no &&, no ternary, no null-coalescing).
- If a heredoc in the Bash tool fails with "unexpected EOF while looking for matching quote", write the file with the Write tool instead.
- Heavy runs: you may read GGUF headers / single tensors of the real models (memory-mapped, cheap). Do NOT run inference on the big models EXCEPT where your spec's test field explicitly requires a smoke run (e.g. the probe's --mode names / --smoke): then run it once, with -t 6, after checking that no other llama process is running (tasklist / docker ps), and report timings.
- Honesty: separate MEASURED / PROJECTED / ASSUMED everywhere; never invent measurements. Measured desktop numbers live in results/desktop-baseline-2026-10-04.md.
- No git commits. No questions: make reasonable choices, document them in the file header, list them in open_issues.
- Return structured output: files written, the exact public interface (CLI flags, functions, file formats) dependents can rely on, how you self-tested (commands + results), open issues.

# MANIFEST INDEX (all entries; level = dependency level used for scheduling)
${manifestIndex}

# SHARED DATA FORMATS: docs/design/manifest.json -> data_formats[] (must match exactly across Python, C++ and bash)
`

const maxLevel = Math.max(...Object.values(level))
const done = []

phase('Implement')
for (let lv = 0; lv <= maxLevel; lv++) {
  const files = M.files.filter(f => level[key(f.path)] === lv)
  if (!files.length) continue
  log(`level ${lv}: ${files.length} entries -> ${files.map(f => key(f.path)).join(', ')}`)
  const prior = done.map(r => `- ${r.path} (${(r.files_written || []).join(', ')}): ${r.interface_notes}`).join('\n')
  const out = await parallel(files.map(f => () =>
    agent(`${FACTS}\n${RULES}\n\n# FINISHED ENTRIES AND THEIR REAL INTERFACES\n${prior || '(none yet - you are in level 0)'}\n\n# YOUR ASSIGNMENT\nEntry: ${f.path}\nLanguage: ${f.language}\nDepends on: ${(f.depends_on || []).join(', ') || 'nothing'}\nSpec, purpose and test expectation: the files[] entry with this exact path in docs/design/manifest.json - read it in full.\n\nWrite the file(s) completely (no stubs, no TODO placeholders for required behaviour), then self-test.`,
      { label: `L${lv}:${key(f.path)}`, phase: 'Implement', schema: FILE_RESULT })
  ))
  out.forEach((r, i) => {
    if (r) done.push(r)
    else log(`WARNING: no result for ${key(files[i].path)}`)
  })
}
log(`${done.filter(r => r.written).length}/${M.files.length} entries report written`)

phase('Build+Test')
const BT = {
  type: 'object',
  properties: {
    build_ok: { type: 'boolean' },
    built_binaries: { type: 'array', items: { type: 'string' } },
    pytest_ok: { type: 'boolean' },
    pytest_summary: { type: 'string' },
    bash_n_ok: { type: 'boolean' },
    fixes_applied: { type: 'array', items: { type: 'string' } },
    remaining_problems: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
  required: ['build_ok', 'pytest_ok', 'bash_n_ok', 'fixes_applied', 'remaining_problems', 'summary'],
}
const bt = await agent(`${FACTS}\n${RULES}\n\n# ALL ENTRY REPORTS\n${JSON.stringify(done, null, 1)}\n\n# YOUR JOB: MAKE THE REPO BUILD AND THE UNIT TESTS PASS
You may now edit ANY repo file (not third_party/, not models/). Work until green or until a problem is genuinely blocked:
1. Build all native tools in Docker via the repo's own wrapper if it exists (bash pc/docker_run.sh build-tools), else the raw cmake command from the rules. Fix compile/link errors. Confirm the binaries exist where pigemma/paths.py expects them.
2. Run "python -m pytest -q" from the repo root on Windows (not integration-marked tests). Fix failures in code or tests (fix the code when the test encodes the spec; fix the test only when it contradicts the manifest/data formats).
3. Run "bash -n" on every pi/*.sh and pc/*.sh, and parse every pc/*.ps1 with PowerShell's parser ([System.Management.Automation.Language.Parser]::ParseFile).
4. Cross-check the shared contracts: .moetrace writer (C++ probe) vs reader (pigemma/trace.py) constants and record size; map.json producer vs consumers; results envelope fields; catalog.json loader. Fix drift.
5. Run the GGUF map builder on BOTH real GGUFs and check the verified constants (26B: data_offset 15,823,936, expert unit 3,345,408 B, 802,897,920 B/token worst case, non-expert 1,577,172,088 B; 31B: data_offset 15,833,440, 833 tensors). Save the maps where the manifest says (results/map26.json, results/map31.json or as implemented).
Report precisely: what built, pytest counts, fixes, remaining problems.`,
  { label: 'build-and-test', phase: 'Build+Test', schema: BT })

return { levels: level, entries: done, build_test: bt }
