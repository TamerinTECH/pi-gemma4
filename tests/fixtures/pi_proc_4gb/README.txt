Fake system root for pi/lib/common.sh (PIGEMMA_FAKE_PROC=tests/fixtures/pi_proc_4gb).
Shape of the first probe of context-friend-1 (docs/design/FACTS.md "Raspberry Pi target"): Pi 5 4 GB,
MemTotal 4,039,000 kB, 30 GB microSD (mmcblk0p2 on /), 16 KiB pages, Cortex-A76 x4 (asimddp, no i8mm/sve),
a Hailo AI HAT on the PCIe lane, ContextFriend production services running. Expected: storage microsd,
assert_safe_host exits 4 (protected services; also the catalog's protected_hosts while it lists the name),
feasibility exit 3 with PIGEMMA_I_KNOW=1. Values not stated in FACTS.md (descriptions, MemFree, temperature,
HAT product string, config.txt body) are illustrative, not measurements.
