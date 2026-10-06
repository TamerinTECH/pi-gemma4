Fake system root for pi/lib/common.sh (PIGEMMA_FAKE_PROC=tests/fixtures/pi_proc_16gb).
The recommended configuration (docs/design/FACTS.md "Hardware options"): Pi 5 16 GB + NVMe on PCIe Gen3 x1
(dtparam=pciex1,pciex1_gen=3, link 8.0 GT/s x1), model dir /mnt/nvme/models, memlock unlimited, no protected
services (friendly-context.service is a near-miss that must NOT match the protected globs). Expected:
storage nvme, assert_safe_host passes, pick_jobs 4, feasibility exit 0 (26b-q4 FITS_MAPPED, -nr).
Synthetic profile: no such host was probed; numbers are illustrative.
