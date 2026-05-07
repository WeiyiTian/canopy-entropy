#!/usr/bin/env bash
set -euo pipefail

SWEEP="${SWEEP:-all-families-matrix}"

generate() {
  python -m scripts.generate_rollouts.generate_rollouts +sweep="$SWEEP"
}

process() {
  python -m scripts.process_rollouts.embed_rollouts +sweep="$SWEEP"
}

compute() {
  python -m scripts.compute_gen_space.compute_gen_space +sweep="$SWEEP"
}

stats() {
  python -m scripts.stats.bootstrap_base_vs_instruct
  python -m scripts.stats.fit_beta_interaction_regression
}

plots() {
  python -m scripts.plotting.plot_ce_star_decomposition
  python -m scripts.plotting.plot_entropy_rate_vs_length
  python -m scripts.plotting.plot_sequence_length_distribution
  python -m scripts.plotting.plot_variance_distribution
}

case "${1:-all}" in
  all)       generate; process; compute; stats; plots ;;
  generate)  generate ;;
  process)   process ;;
  compute)   compute ;;
  stats)     stats ;;
  plots)     plots ;;
  *) echo "usage: $0 {all|generate|process|compute|stats|plots}  (env: SWEEP=<name>)"; exit 1 ;;
esac
