#!/bin/bash
# Run once after the restart (2026-09-23). Relaunches what a reboot interrupts; everything is restart-safe
# (scoring skips JSONs that already exist).
cd /Users/benjaminmaurel/Documents/PharmaNODE
grep -q DECRES_ALL_DONE logs/chain_decres.log 2>/dev/null || \
  nohup scripts/s4/chain_decres.sh > logs/chain_decres_after_reboot.log 2>&1 &   # FiLM residual scoring, only if unfinished
nohup scripts/s4/sched_dc_decres_ln.sh > logs/sched_dc_decres_ln.log 2>&1 &          # dose-cond residual, low noise, 4 seeds
nohup scripts/s4/chain_dc_decres.sh > logs/chain_dc_decres_after_reboot.log 2>&1 &  # its scoring
nohup scripts/s4/chain_tslow_pre.sh > logs/chain_tslow_pre_after_reboot.log 2>&1 &   # non-Gaussian scenario: stats (no-op if done)
nohup scripts/s4/chain_tslow_net.sh > logs/chain_tslow_net.log 2>&1 &                  # non-Gaussian scenario: 9 network runs + scoring
echo "relaunched at $(date '+%F %T')"
