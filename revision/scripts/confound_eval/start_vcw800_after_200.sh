#!/bin/bash
# Overnight queue (Benjamin, 2026-09-24 23:15: "launch what can be useful"): start the N=800 windowed Vc confirmation
# (chain_vcw_matched.sh, unchanged) only once the N=200 chain has fully finished and scored (VCW200_MATCHED_DONE),
# so it never competes with the N=200 scoring. Log appended to logs/chain_vcw_matched.log.
cd /Users/benjaminmaurel/Documents/PharmaNODE
until grep -q VCW200_MATCHED_DONE logs/chain_vcw200_matched.log 2>/dev/null; do sleep 120; done
echo "[$(date '+%F %T')] VCW200_MATCHED_DONE seen -> starting chain_vcw_matched.sh (N=800)" >> logs/chain_vcw_matched.log
exec bash scripts/confound_eval/chain_vcw_matched.sh >> logs/chain_vcw_matched.log 2>&1
