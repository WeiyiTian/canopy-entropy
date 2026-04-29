# BranchingEntropy

debug
```
python -m scripts.generate_rollouts.generate_rollouts --cfg job +sweep=debug
python -m scripts.generate_rollouts.generate_rollouts +sweep=debug
```

generate rollouts
```
python -m scripts.generate_rollouts.generate_rollouts +sweep=gemma-3-12b-matrix
python -m scripts.generate_rollouts.generate_rollouts +sweep=llama3.1-8b-matrix
python -m scripts.generate_rollouts.generate_rollouts +sweep=qwen3-8b-matrix
python -m scripts.generate_rollouts.generate_rollouts +sweep=all-families-matrix

python -m scripts.generate_rollouts.generate_rollouts +sweep=all-families-matrix resume=false 
```

```
./scripts/tail-sweep.sh generate_rollouts all-families-matrix
./scripts/tail-sweep.sh generate_rollouts gemma
./scripts/tail-sweep.sh generate_rollouts qwen3
./scripts/tail-sweep.sh generate_rollouts llama
```

```
./scripts/tail-sweep.sh embed_rollouts
./scripts/tail-sweep.sh embed_rollouts all-families-matrix
```