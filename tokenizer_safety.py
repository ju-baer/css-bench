"""
css_bench.dataset.tokenizer_safety
=====================================
Real tokenizers split text unpredictably, so single-token safety needs a
sharper check than "does the word alone tokenize to one token." Prompts end
in `"MY CHOICE: ["` with no trailing space, so what matters is whether
appending the label directly after `[` produces exactly one *additional*
token beyond however many tokens `[` itself takes -- and that the
tokenization doesn't retroactively merge with the bracket into something
else. See tests/test_tokenizer_safety.py for the offline validation against
a simulated BPE merge-boundary case.
"""
from typing import Dict, List, Optional, Tuple

from css_bench.dataset.taxonomy import CategoryPools, SEMANTIC_TAXONOMY


def token_after_bracket(tokenizer, word: str) -> Optional[int]:
    """Returns the token id for `word` as a continuation of "[" (no leading
    space, matching how labels appear in the prompt) IFF it is exactly one
    additional token beyond however many tokens "[" alone takes, and the
    prefix tokenization is unchanged by the merge. Returns None otherwise."""
    bracket_only = tokenizer.encode("[", add_special_tokens=False)
    full = tokenizer.encode("[" + word, add_special_tokens=False)
    if len(full) != len(bracket_only) + 1:
        return None
    if full[:len(bracket_only)] != bracket_only:
        return None
    return full[-1]


def filter_single_token_words(words: List[str], tokenizer) -> List[str]:
    return [w for w in words if token_after_bracket(tokenizer, w) is not None]


def compute_safe_word_intersection_taxonomy(
    model_registry: Dict[str, dict], model_keys: List[str], active_categories: List[str],
) -> Tuple[Dict[str, CategoryPools], Dict[str, dict]]:
    """Loads only tokenizers (not full models -- cheap and fast) for every
    active model, filters every category's word pools down to the
    intersection that's single-token-safe on ALL of them, and warns
    per-category if that intersection is too thin. Using one shared pool per
    category means every model sees IDENTICAL instances, so cross-model
    comparisons in later phases aren't confounded by different models
    effectively running on different datasets.

    `arbitrary_nonce` is the category most likely to warn -- invented words
    have no guarantee of surviving a real BPE vocabulary as a single token;
    if its pool ends up too thin to use, that is itself a reportable fact
    about the limits of this design, not a bug to paper over.
    """
    from transformers import AutoTokenizer

    per_model: Dict[str, dict] = {}
    intersections = {
        cat: {"positive": set(SEMANTIC_TAXONOMY[cat]["positive"]),
              "negative": set(SEMANTIC_TAXONOMY[cat]["negative"])}
        for cat in active_categories
    }
    for key in model_keys:
        hf_id = model_registry[key]["hf_id"]
        print(f"Loading tokenizer for {key} ({hf_id}) ...")
        tok = AutoTokenizer.from_pretrained(hf_id)
        per_model[key] = {}
        for cat in active_categories:
            pools = SEMANTIC_TAXONOMY[cat]
            safe_pos = filter_single_token_words(pools["positive"], tok)
            safe_neg = filter_single_token_words(pools["negative"], tok)
            per_model[key][cat] = {"positive": safe_pos, "negative": safe_neg}
            intersections[cat]["positive"] &= set(safe_pos)
            intersections[cat]["negative"] &= set(safe_neg)
        print(f"  {key}: " + ", ".join(
            f"{cat}={len(per_model[key][cat]['positive'])}+/{len(per_model[key][cat]['negative'])}-"
            for cat in active_categories))

    result: Dict[str, CategoryPools] = {
        cat: {"positive": sorted(intersections[cat]["positive"]),
              "negative": sorted(intersections[cat]["negative"]),
              "description": SEMANTIC_TAXONOMY[cat]["description"]}
        for cat in active_categories
    }
    print(f"\nCross-model intersection ({len(model_keys)} models):")
    for cat in active_categories:
        n_pos, n_neg = len(result[cat]["positive"]), len(result[cat]["negative"])
        print(f"  {cat:18s}: {n_pos} positive / {n_neg} negative words safe on all models")
        if n_pos < 2 or n_neg < 2:
            print(f"    WARNING: '{cat}' has too few surviving words to sample from meaningfully. "
                  f"Consider adding candidates to SEMANTIC_TAXONOMY['{cat}'] or excluding it from "
                  f"ACTIVE_SEMANTIC_CATEGORIES for this run.")
    return result, per_model
