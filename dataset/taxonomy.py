"""
css_bench.dataset.taxonomy
===========================
The semantic projection taxonomy (paper §4.1 / Part C.1): each category
supplies a `positive` word pool (assigned to the correct slot under the
`canon` projection) and a `negative` pool (assigned to the correct slot
under `counter`). Every non-`neutral` category is instantiated as a matched
canonical/counterfactual pair using the *same two literal words* in both,
differing only in which slot each is assigned to — see
`css_bench.dataset.generator.build_dataset`.

`neutral` is a deliberate null baseline: its "positive"/"negative" labeling
is arbitrary (there is no real valence), but it is threaded through the
identical canon/counter machinery as every other category specifically so
it receives the identical statistical treatment — a near-zero effect on
`neutral` alongside a nonzero effect elsewhere is direct evidence the effect
is semantic, not an artifact of relabeling *anything*.
"""
from typing import Dict, List, TypedDict


class CategoryPools(TypedDict):
    positive: List[str]
    negative: List[str]
    description: str


SEMANTIC_TAXONOMY: Dict[str, CategoryPools] = {
    "neutral": {
        "positive": ["A", "X", "One"],
        "negative": ["B", "Y", "Two"],
        "description": "Null baseline: arbitrary identifiers with no semantic content.",
    },
    "arbitrary_nonce": {
        "positive": ["Zorp", "Fenu", "Trask", "Molvi"],
        "negative": ["Blen", "Groth", "Yamex", "Duplo"],
        "description": "Novel tokens with no pretrained association (lexical control). "
                        "Caveat: not guaranteed single-token on real tokenizers -- see "
                        "css_bench.dataset.tokenizer_safety.",
    },
    "valence": {
        "positive": ["Trust", "Share", "Help", "Give", "Ally", "Support", "Unite", "Aid",
                     "Cooperate", "Kindness", "Honor", "Loyal", "Generous", "Fair", "Peace",
                     "Friend", "Gentle", "Honest", "Care", "Protect", "Bond", "Harmony",
                     "Grace", "Mercy", "Comfort", "Respect", "Welcome", "Embrace", "Gift", "Charity"],
        "negative": ["Betray", "Steal", "Harm", "Cheat", "Attack", "Exploit", "Abandon", "Sabotage",
                     "Defect", "Cruelty", "Deceive", "Greed", "Threat", "Punish", "War",
                     "Enemy", "Brutal", "Lie", "Hurt", "Destroy", "Rift", "Chaos",
                     "Malice", "Spite", "Damage", "Insult", "Reject", "Isolate", "Theft", "Ruin"],
        "description": "General positive/negative connotation (the original manipulation).",
    },
    "moral": {
        "positive": ["Good", "Virtuous", "Righteous", "Honorable", "Moral", "Just"],
        "negative": ["Evil", "Wicked", "Sinful", "Corrupt", "Immoral", "Unjust"],
        "description": "Explicit moral framing.",
    },
    "social": {
        "positive": ["Cooperate", "Unite", "Include", "Ally"],
        "negative": ["Exploit", "Divide", "Isolate", "Undermine"],
        "description": "Social/relational framing, closer to the game's own content.",
    },
    "role": {
        "positive": ["Defender", "Guardian", "Protector"],
        "negative": ["Attacker", "Invader", "Aggressor"],
        "description": "Identity/role framing.",
    },
    "prestige": {
        "positive": ["Noble", "Honored", "Esteemed"],
        "negative": ["Shameful", "Disgraced", "Despised"],
        "description": "Status/prestige framing.",
    },
}

ALL_CATEGORIES: List[str] = list(SEMANTIC_TAXONOMY.keys())
