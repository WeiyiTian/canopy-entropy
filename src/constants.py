MODEL_VARIANTS = {
    "Qwen2.5-7b": {
        "base": "Qwen2.5-7B",
        "instruct": "Qwen2.5-7B-Instruct",
    },
    "Qwen3-8b": {
        "base": "Qwen3-8B-Base",
        "instruct": "Qwen3-8B",
    },
    "Llama-3.1-8b": {
        "base": "Llama-3.1-8B",
        "instruct": "Llama-3.1-8B-Instruct",
    },
    "Gemma-3-12b": {
        "base": "gemma-3-12b-pt",
        "instruct": "gemma-3-12b-it",
    },
}

GENERATION_SPACE_METRIC_KEYS = (
    "gen_ppl",
    "ce_star_max",
    "branching_factor",
    "entropy_rate_vs_length",
    "truncation_rate",
    "semantic_diversity_vs_length",
    "semantic_diversity",
    "semantic_diversity_bucketed_mean",
)

LENGTH_BUCKET_NAMES = ("short", "medium", "long")

ROLLOUT_SHARDS_ARTIFACT = "rollout_shards"
EMBEDDING_SHARDS_ARTIFACT = "embedding_shards"
REWARD_SHARDS_ARTIFACT = "reward_shards"
