Fake system root for pi/lib/common.sh (PIGEMMA_FAKE_PROC=tests/fixtures/pi_proc_cf1_16gb).
context-friend-1 as probed read-only on 2026-10-05 (docs/design/FACTS.md UPDATE): Pi 5 Model B Rev 1.1
(e04171), MemTotal 16,607,552 kB, microSD only with 15,501,496,320 B free on /, 16 KiB pages, ulimit -l 8192,
governor ondemand, no HAT, fio not installed, ContextFriend + Hailo services running. Expected: assert_safe_host
exits 4; with PIGEMMA_I_KNOW=1 it passes with a loud warning and facts.override == 'PIGEMMA_I_KNOW'.
MemFree, temperature, service descriptions and the config.txt body are illustrative.
