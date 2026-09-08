#!/usr/bin/env python3
"""Create leakage-controlled Hindi and Nepali comparative-reasoning data.

The generator is deterministic: re-running it with the same seed replaces no
files and produces byte-for-byte equivalent JSONL records.  Test examples use
only held-out entity names and the held-out ``price`` relation; neither occurs
in train or validation.
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260908
SIZES = {"train": 3600, "val": 400, "test": 400}

NAMES = {
    "hindi": ["आरव", "विवान", "अर्जुन", "कबीर", "रोहन", "अमन", "आदित्य", "राहुल", "मोहन", "करण", "नीरज", "सूरज", "विकास", "निखिल", "पंकज", "संजय", "दीपक", "अनिल", "रवि", "मनोज", "पूजा", "रिया", "काव्या", "अनन्या", "सिया", "नेहा", "मीरा", "सोनम", "निशा", "प्रिया", "श्रुति", "आरती", "सविता", "किरण", "वंदना", "रेखा", "मनीषा", "स्वाति", "राधा", "तन्वी", "ईशा", "दिव्या", "साक्षी", "नम्रता", "हेमा", "लता", "गीता", "सीमा", "उमा", "ज्योति"],
    "nepali": ["आरव", "सुमन", "बिमल", "रमेश", "सुरेश", "किरण", "दीपक", "राजन", "अनिल", "निश्चल", "प्रकाश", "विवेक", "मनोज", "अमित", "रोशन", "बिनोद", "कमल", "नवराज", "सञ्जय", "सुजन", "सिता", "राधा", "माया", "गीता", "सरिता", "निशा", "कविता", "पूनम", "अनिता", "सविता", "रेखा", "मिना", "सुजाता", "प्रतिमा", "संगीता", "आरती", "विनिता", "स्मृति", "कल्पना", "दिप्ती", "रञ्जना", "इशा", "श्रुति", "शोभा", "उमा", "लक्ष्मी", "वन्दना", "सुनिता", "रिता", "मञ्जु"],
}

ATTRS = {
    "height": (140, 200), "age": (5, 90), "price": (10, 10000), "quantity": (1, 1000)
}

HINDI = {
    "height": {"value": "की ऊँचाई {v} सेंटीमीटर है", "high": "लंबा", "low": "छोटा"},
    "age": {"value": "की उम्र {v} वर्ष है", "high": "बड़ी उम्र का", "low": "कम उम्र का"},
    "price": {"value": "की वस्तु की कीमत {v} रुपये है", "high": "महंगा", "low": "सस्ता"},
    "quantity": {"value": "के पास {v} किताबें हैं", "high": "ज़्यादा किताबों वाला", "low": "कम किताबों वाला"},
}
NEPALI = {
    "height": {"value": "को उचाइ {v} सेन्टिमिटर छ", "high": "अग्लो", "low": "होचो"},
    "age": {"value": "को उमेर {v} वर्ष छ", "high": "धेरै उमेरको", "low": "कम उमेरको"},
    "price": {"value": "को सामानको मूल्य {v} रुपैयाँ छ", "high": "महँगो", "low": "सस्तो"},
    "quantity": {"value": "सँग {v} वटा किताब छन्", "high": "धेरै किताब भएको", "low": "कम किताब भएको"},
}

def distinct_values(rng, attr, count):
    lo, hi = ATTRS[attr]
    return rng.sample(range(lo, hi + 1), count)

def direct(language, names, attr, template, rng):
    a, b = rng.sample(names, 2)
    va, vb = distinct_values(rng, attr, 2)
    lex = (HINDI if language == "hindi" else NEPALI)[attr]
    larger = a if va > vb else b
    smaller = b if va > vb else a
    ask_high = rng.choice([True, False])
    answer = larger if ask_high else smaller
    adjective = lex["high"] if ask_high else lex["low"]
    av, bv = lex["value"].format(v=va), lex["value"].format(v=vb)
    if language == "hindi":
        forms = [
            f"{a} {av} और {b} {bv}। इनमें कौन {adjective} है?",
            f"{a} {av}; {b} {bv}। बताइए, कौन {adjective} है?",
            f"तुलना कीजिए: {a} {av} जबकि {b} {bv}। कौन {adjective} है?",
        ]
    else:
        forms = [
            f"{a} {av} र {b} {bv}। यीमध्ये को {adjective} छ?",
            f"{a} {av}; {b} {bv}। भन्नुहोस्, को {adjective} छ?",
            f"तुलना गर्नुहोस्: {a} {av} तर {b} {bv}। को {adjective} छ?",
        ]
    return forms[template % len(forms)], answer, [a, b], {a: va, b: vb}

def transitive(language, names, attr, template, rng):
    a, b, c = rng.sample(names, 3)
    # Values are retained as auditable ground truth but deliberately omitted
    # from the prompt: the answer requires composing two stated relations.
    high, mid, low = sorted(distinct_values(rng, attr, 3), reverse=True)
    lex = (HINDI if language == "hindi" else NEPALI)[attr]
    ask_high = rng.choice([True, False])
    answer = a if ask_high else c
    adjective = lex["high"] if ask_high else lex["low"]
    relation = lex["high"]
    if language == "hindi":
        forms = [
            f"{a} {b} से अधिक {relation} है, और {b} {c} से अधिक {relation} है। इनमें सबसे {adjective} कौन है?",
            f"{a}, {b} से अधिक {relation} है। {b}, {c} से अधिक {relation} है। बताइए, सबसे {adjective} कौन है?",
            f"क्रम यह है: {a} > {b} और {b} > {c}, जहाँ > का अर्थ अधिक {relation} है। सबसे {adjective} कौन है?",
        ]
    else:
        forms = [
            f"{a}, {b} भन्दा बढी {relation} छ र {b}, {c} भन्दा बढी {relation} छ। यीमध्ये सबैभन्दा {adjective} को छ?",
            f"{a} {b} भन्दा बढी {relation} छ। {b} {c} भन्दा बढी {relation} छ। सबैभन्दा {adjective} को हो?",
            f"क्रम यस्तो छ: {a} > {b} र {b} > {c}, जहाँ > को अर्थ बढी {relation} हो। सबैभन्दा {adjective} को हो?",
        ]
    return forms[template % len(forms)], answer, [a, b, c], {a: high, b: mid, c: low}

def make_split(language, split, rng, names, attributes):
    records, seen = [], set()
    while len(records) < SIZES[split]:
        task_type = "direct" if len(records) % 2 == 0 else "transitive"
        attr = rng.choice(attributes)
        template = rng.randrange(3)
        factory = direct if task_type == "direct" else transitive
        prompt, answer, entities, values = factory(language, names, attr, template, rng)
        if prompt in seen:
            continue
        seen.add(prompt)
        answer_prefix = "उत्तर: " if language == "hindi" else "जवाफ: "
        records.append({
            "id": f"{language}-{split}-{len(records)+1:05d}", "language": language,
            "split": split, "task_type": task_type, "attribute": attr,
            "template_id": template, "entities": entities, "values": values,
            "prompt": prompt, "answer_prefix": answer_prefix, "answer": answer,
            "training_text": f"{prompt}\n{answer_prefix}{answer}",
        })
    return records

def verify(language, records, train_names, test_names):
    train_val = records["train"] + records["val"]
    assert all(r["attribute"] != "price" for r in train_val)
    assert all(r["attribute"] == "price" for r in records["test"])
    assert all(set(r["entities"]).issubset(train_names) for r in train_val)
    assert all(set(r["entities"]).issubset(test_names) for r in records["test"])
    assert not {n for r in train_val for n in r["entities"]} & test_names
    assert {r["task_type"] for r in records["test"]} == {"direct", "transitive"}

def generate(language, output_dir, seed):
    rng = random.Random(seed + (0 if language == "hindi" else 1000))
    held_out_count = 10  # 20% of the 50-name pool
    train_names, test_names = NAMES[language][:-held_out_count], NAMES[language][-held_out_count:]
    records = {
        "train": make_split(language, "train", rng, train_names, ["height", "age", "quantity"]),
        "val": make_split(language, "val", rng, train_names, ["height", "age", "quantity"]),
        "test": make_split(language, "test", rng, test_names, ["price"]),
    }
    verify(language, records, set(train_names), set(test_names))
    language_dir = output_dir / language
    language_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        with (language_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metadata = {
        "seed": seed, "sizes": SIZES, "name_pool_size": len(NAMES[language]),
        "train_val_names": train_names, "held_out_test_names": test_names,
        "held_out_attribute_for_test": "price",
        "train_val_attributes": ["height", "age", "quantity"],
        "leakage_checks": "passed: disjoint entity pools; price occurs only in test",
    }
    (language_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{language}: wrote {sum(SIZES.values())} examples to {language_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["hindi", "nepali", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "phase3" / "data")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    for language in (["hindi", "nepali"] if args.language == "all" else [args.language]):
        generate(language, args.output_dir, args.seed)

if __name__ == "__main__":
    main()
