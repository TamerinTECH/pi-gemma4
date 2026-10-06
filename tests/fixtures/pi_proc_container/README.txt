Fake system root for pi/lib/common.sh (PIGEMMA_FAKE_PROC=tests/fixtures/pi_proc_container).
An arm64 Debian 13 container under qemu-user (docker run --platform linux/arm64 --memory 8g): no systemd (no
systemctl.txt), overlay root, no device-tree, no /boot/firmware/config.txt, cgroup v1 memory limit 8 GiB,
4 KiB pages, 12 vCPUs. Shape captured from pi-gemma4-pi-emu:trixie on 2026-10-05; usage/stat values are
illustrative. Expected: storage unknown, running_services [] with services_source 'unavailable: ...',
cgroup_mem_max 8589934592, ram_available_kb = (8589934592 - (300000000 - 100000000)) // 1024 = 8193295,
assert_safe_host passes with a warning.
