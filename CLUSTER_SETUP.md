# SwiftLM cluster — setup guide (2-box trace-volume fan-out)

Goal: run the broad domain-flywheel best-of-N generation across **two Macs** so the ~9h one-box 32B
run drops to ~9h ÷ (#boxes). Box A = **coordinator** (your M4 Max 128GB — runs the flywheel, owns the
repo + verifier + training). Box B = **worker laptop** (just generates rollouts over the network).

How it works: the coordinator builds one best-of-N *generation* job per task and fans them across all
workers (round-robin, concurrent). Each worker runs same-prompt best-of-N on its GPU and ships back the
candidate strings. **Only the coordinator verifies** (clonefile → build → test) — so the laptop needs
the model but **NOT** your repos or `~/.swiftlm`.

---

## ⚠️ 0. Pre-check: can the laptop run the model?

The cluster fans the **same** model across both boxes. `DeepSeek-R1-Distill-Qwen-32B-abliterated-4bit`
needs **~20–24 GB free RAM**. So:
- Laptop has **≥32 GB**: ✅ it can be a 32B worker. Proceed.
- Laptop has **16 GB**: ❌ can't hold the 32B. Either (a) do a **7B smoke run** on both (still validates
  the whole pipeline), or (b) keep the 32B job on Box A only and use the laptop for a 7B/14B job. Tell me
  the laptop's RAM and I'll tailor it.

---

## 1. On the laptop (Box B / worker)

**a. Get the code.** Copy or clone the SwiftLM repo onto the laptop (e.g. `~/Documents/SwiftLM`). It
needs **Xcode** installed (the MLX targets build with Metal).

**b. Build the worker binary:**
```sh
cd ~/Documents/SwiftLM
xcodebuild -scheme selfloop -destination 'platform=macOS' -derivedDataPath .xcode \
  -configuration Release -skipMacroValidation -skipPackagePluginValidation build
```

**c. Start the worker** (this auto-downloads the model from HuggingFace on first run — ~18 GB, one time):
```sh
WORKER=1 WORKER_PORT=8787 \
SWIFTLM_MODEL=mlx-community/DeepSeek-R1-Distill-Qwen-32B-abliterated-4bit \
caffeinate -dimsu .xcode/Build/Products/Release/selfloop
```
Wait for: `✅ SwiftLM cluster worker READY — model=… port=8787. Waiting for jobs …`
(`SWIFTLM_MODEL` **must exactly match** Box A's. `caffeinate` keeps it awake during the run.)

**d. Find the laptop's IP** (Box A connects to this):
```sh
ipconfig getifaddr en0      # Wi-Fi;  try en1 if blank (Ethernet/Thunderbolt)
```
Note it, e.g. `192.168.1.42`.

---

## 2. Networking

- Both Macs on the **same LAN** (Wi-Fi or Ethernet) is fine to start. For big reasoning traces, a
  **Thunderbolt bridge** cable (System Settings → Network → Thunderbolt Bridge) is much faster — use that
  box's bridge IP instead.
- **Firewall:** if Box A can't connect, macOS firewall on the laptop is likely blocking. Either allow
  the binary (System Settings → Network → Firewall → Options → add `selfloop` / allow incoming), or for a
  quick test toggle the firewall off on the laptop.
- Quick reachability check from Box A: `nc -vz <laptop-ip> 8787` → should say `succeeded`.

---

## 3. On the M4 Max (Box A / coordinator)

**Smoke test FIRST** (cheap — validates the whole distributed path before the long run). Point a 7B job
at the laptop (start the laptop worker with the **7B** model for this, to match):
```sh
cd ~/Documents/SwiftLM
CLUSTER_WORKERS=<laptop-ip>:8787 \
DOMAIN_FLYWHEEL=1 FLYWHEEL_N=4 FLYWHEEL_KS=1 FLYWHEEL_MAXTOK=1200 DOMAIN_EVAL_SAMPLES=2 FLYWHEEL_TEMP=0 \
.xcode/Build/Products/Release/selfloop
```
Look for `CLUSTER: generation fans across 2 workers (this box + 1 remote)` in the output → the fan-out is
live. (If a worker dies mid-run, its jobs return empty and that task just yields fewer traces — the run
doesn't crash.)

**The real broad run** (32B, overnight — laptop worker must be on the 32B too):
```sh
CLUSTER_WORKERS=<laptop-ip>:8787 \
DOMAIN_FLYWHEEL=1 DOMAIN_DISTILL=1 DOMAIN_DPO=1 THINK=1 \
SWIFTLM_MODEL=mlx-community/DeepSeek-R1-Distill-Qwen-32B-abliterated-4bit \
DOMAIN_HOLDOUT=Linear.backward \
FLYWHEEL_TEMP=0.6 FLYWHEEL_MAXTOK=4096 FLYWHEEL_N=12 FLYWHEEL_KS=1,4 \
DOMAIN_EVAL_SAMPLES=8 DOMAIN_ITERS=80 DOMAIN_LR=1e-5 DOMAIN_BETA=0.1 \
caffeinate -dimsu .xcode/Build/Products/Release/selfloop > /tmp/domain-dpo-cluster.log 2>&1 &
```
Now the 26-task best-of-12 generation splits across both boxes. Watch the per-task band table
(`per-task yield over N train tasks`) — with breadth + 2 boxes you should finally see many in-band tasks
contributing winners.

Add more workers later: `CLUSTER_WORKERS=ip1:8787,ip2:8787,ip3:8787` (comma-separated).

---

## 4. Notes / gotchas

- **Same model on every box** — the coordinator assumes homogeneous workers (it round-robins without
  checking which model a worker holds). Mismatched models = garbage traces.
- The worker is **stateless + generation-only** — no repo, no `~/.swiftlm`, no transcripts leave Box A
  (only prompts go out, candidate code comes back). On-device-only is preserved.
- Worker logs each batch it serves; coordinator logs the fan-out + the per-task yield.
- To stop: Ctrl-C the worker; the coordinator finishes or Ctrl-C it.
- **Bonjour auto-discovery** (skip the manual IP) isn't wired into the CLI yet — coming next; for now use
  explicit `host:port`.
