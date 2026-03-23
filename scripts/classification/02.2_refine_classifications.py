import json
import os
import time
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# ========== CONFIGURATION ==========
INPUT_FILENAME = "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_gpt4omini_entity_class_uniq_captions_v16.json"
OUTPUT_FILENAME = "/home/aparcedo/IASEB/interaction_analysis/vg12_vrd_stg_mevis_rvos_gpt4omini_entity_class_uniq_captions_v17.json"
CACHE_FILENAME = "/home/aparcedo/IASEB/interaction_analysis/classification_cache_v17.json"
LOG_FILENAME = "/home/aparcedo/IASEB/interaction_analysis/classification_v17.log"
MODEL = "gpt-4o-mini"
BATCH_SIZE = 1
MAX_RETRIES = 3

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CLASS_CATEGORIES = [
    "Affective",
    "Antagonistic",
    "Body Motion",
    "Communicative",
    "Competitive",
    "Cooperative",
    "Movement",
    "Observation",
    "Passive",
    "Physical Interaction",
    "Provisioning",
    "Proximity",
    "Social",
    "Spatial",
    "Supportive"
]

# ===================== LOGGING SETUP =====================
logging.basicConfig(
    filename=LOG_FILENAME,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)

# -------------------- helper utilities --------------------
def normalize_caption(c: str) -> str:
    """Normalize caption for consistent cache keys (strip only)."""
    if not isinstance(c, str):
        c = str(c)
    return c.strip()

def category_contains_review(cat_field) -> bool:
    """Return True if category field (str or list) contains REVIEW or RECOMMEND_NEW."""
    if isinstance(cat_field, list):
        items = [str(x).strip().upper() for x in cat_field]
    else:
        # treat single string
        items = [str(cat_field).strip().upper()]
    return any(x in ("REVIEW", "RECOMMEND_NEW") for x in items)

# -------------------- robust batch ask --------------------
def ask_gpt_batch(captions, attempt=1):
    """
    Same batching behavior as before. Returns dict mapping normalized caption -> result dict.
    For brevity I keep this minimal; assume it's your previously working implementation.
    """
    # Build prompt
    joined_captions = "\n".join([f"{i+1}. {cap}" for i, cap in enumerate(captions)])
    prompt = f"""
You are labeling visual interaction captions into one or more categories.

Categories:
{', '.join(CLASS_CATEGORIES)}

For each caption, return a JSON list like this:
[
  {{
    "caption": "...",
    "categories": ["Movement", "Body Motion"],
    "needs_review": false,
    "notes": "Brief reason why"
  }},
  ...
]

If no existing category fits, use ["RECOMMEND_NEW"].
If uncertain, use ["REVIEW"].

Captions:
{joined_captions}

Return only valid JSON (no extra text).
    """

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a careful annotator of human-object interactions."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )

        text = response.choices[0].message.content.strip()

        # quick truncation check
        if not text.endswith("]"):
            raise ValueError("Output truncated or incomplete")

        parsed = json.loads(text)
        results = {}
        for item in parsed:
            raw_cap = item.get("caption", "")
            cap = normalize_caption(raw_cap)
            cats = item.get("categories", [])
            if not cats or not isinstance(cats, list) or len(cats) == 0:
                cats = ["REVIEW"]
                needs_review = True
                notes = "No valid categories returned"
            else:
                needs_review = any(str(c).strip().upper() in ("REVIEW", "RECOMMEND_NEW") for c in cats)
                notes = item.get("notes", "")
            # Force normalization and lowercase cache key
            results[cap.lower()] = {
                "categories": cats,
                "needs_review": needs_review,
                "notes": notes
            }

        return results

    except Exception as e:
        logging.error(f"Batch error (attempt {attempt}): {e}")
        # On failure, retry split
        if attempt < MAX_RETRIES and len(captions) > 1:
            half = len(captions) // 2
            logging.info(f"Retrying as two sub-batches (attempt {attempt+1}) sizes {half} and {len(captions)-half}")
            time.sleep(attempt * 2)
            res = {}
            res.update(ask_gpt_batch(captions[:half], attempt+1))
            res.update(ask_gpt_batch(captions[half:], attempt+1))
            return res
        # final fallback mark as REVIEW
        logging.warning("Giving up on batch; marking captions as REVIEW")
        return {normalize_caption(cap): {"categories": ["REVIEW"], "needs_review": True, "notes": f"Batch failed: {e}"} for cap in captions}


# -------------------- main re-review pass --------------------
def main():
    logging.info(f"Loading input file: {INPUT_FILENAME}")
    with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_entries = len(data)
    logging.info(f"Loaded {total_entries} total entries")

    # Build set of captions that currently need re-review (normalize)
    to_fix_set = set()
    for entry in data:
        cat_field = entry.get("category", [])
        if category_contains_review(cat_field):
            to_fix_set.add(normalize_caption(entry.get("caption", "")))

    to_fix_list = sorted(list(to_fix_set))
    logging.info(f"Found {len(to_fix_list)} captions needing re-review")

    # Load or create cache
    initial_cache_len = 0
    if os.path.exists(CACHE_FILENAME):
        with open(CACHE_FILENAME, "r", encoding="utf-8") as f:
            raw_cache = json.load(f)
        # Normalize cache keys (strip) to avoid mismatch due to whitespace
        cache = {normalize_caption(k): v for k, v in raw_cache.items()}
        initial_cache_len = len(cache)
        logging.info(f"Loaded existing cache with {initial_cache_len} entries (normalized)")
    else:
        cache = {}

    # Build list of captions we still need to call the API for
    new_captions = [c for c in to_fix_list if c not in cache]
    logging.info(f"Processing {len(new_captions)} new re-review captions (not in cache)")

    # Batch process
    added_count = 0
    for i in tqdm(range(0, len(new_captions), BATCH_SIZE), desc="Reclassifying"):
        batch_raw = new_captions[i:i + BATCH_SIZE]
        # For safety, send original (non-normalized) captions in prompt — but our keys are normalized.
        # Here we'll use the normalized strings (they're the actual captions stripped).
        batch_results = ask_gpt_batch(batch_raw)
        # merge results into cache (normalized keys already)
        before = len(cache)
        cache.update(batch_results)
        after = len(cache)
        added_now = after - before
        added_count += added_now
        logging.info(f"Batch {i//BATCH_SIZE + 1}: added {added_now} new cached captions (cache size now {after})")

        # Save incremental progress
        with open(CACHE_FILENAME, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        time.sleep(0.5)

    logging.info(f"Done batching. Added {added_count} captions this run to cache (initial cache {initial_cache_len} -> now {len(cache)})")

    # Apply cached reclassifications to dataset entries
    applied_count = 0
    still_needing = []
    for entry in data:
        norm_cap = normalize_caption(entry.get("caption", "")).lower()
        result = cache.get(norm_cap)
        if not result and norm_cap.replace("’", "'") in cache:
            # handle curly vs straight apostrophes
            result = cache[norm_cap.replace("’", "'")]
        if result:
            entry["old_category"] = entry.get("category")
            entry["category"] = result.get("categories", entry.get("category"))
            entry["needs_review"] = result.get("needs_review", False)
            entry["notes"] = result.get("notes", entry.get("notes", ""))
            applied_count += 1
        else:
            if category_contains_review(entry.get("category", [])):
                still_needing.append(norm_cap)


    logging.info(f"Applied reclassifications to {applied_count} entries (may include duplicates across dataset entries).")
    if still_needing:
        logging.warning(f"{len(still_needing)} captions still missing from cache after run (examples): {still_needing[:10]}")

    # Save final output
    with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    logging.info("=== SUMMARY ===")
    logging.info(f"Initial cache size: {initial_cache_len}")
    logging.info(f"To-fix captions found: {len(to_fix_list)}")
    logging.info(f"New captions processed: {len(new_captions)}")
    logging.info(f"Added to cache this run: {added_count}")
    logging.info(f"Cache size after run: {len(cache)}")
    logging.info(f"Applied reclassifications to dataset entries: {applied_count}")
    if still_needing:
        logging.info(f"Captions still requiring re-review: {len(still_needing)}")
    logging.info(f"Saved updated file to: {OUTPUT_FILENAME}")
    logging.info("Re-review pass complete.")


if __name__ == "__main__":
    main()
