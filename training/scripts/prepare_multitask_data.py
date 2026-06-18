"""Prepare joint intent + DST training data for TherapistTransformer.

Produces training/data/multitask_train.jsonl and multitask_val.jsonl.
Each record:
{
  "text":     "My boss keeps piling on work",
  "intent":   "work_stress",
  "tokens":   ["my", "boss", "keeps", "piling", "on", "work"],
  "dst_tags": ["O", "B-PERSON", "O", "O", "O", "B-TOPIC"]
}

Usage:
    python training/scripts/prepare_multitask_data.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

SEED_DATA: list[dict] = [
    # crisis
    {"text": "I've been thinking about hurting myself",    "intent": "crisis"},
    {"text": "I don't want to be here anymore",            "intent": "crisis"},
    {"text": "I've been having thoughts of suicide",       "intent": "crisis"},
    {"text": "I've been cutting myself again",             "intent": "crisis"},
    {"text": "I want to end it all",                       "intent": "crisis"},
    # anxiety
    {"text": "I can't stop worrying about everything",     "intent": "anxiety"},
    {"text": "I had a panic attack on the tube this morning", "intent": "anxiety"},
    {"text": "My heart races every time I think about it", "intent": "anxiety"},
    {"text": "I feel overwhelmed all the time",            "intent": "anxiety"},
    {"text": "I keep catastrophising about the future",    "intent": "anxiety"},
    {"text": "I feel so anxious and I can't breathe",      "intent": "anxiety"},
    # depression
    {"text": "I just feel empty like nothing matters",     "intent": "depression"},
    {"text": "I can't get out of bed most mornings",       "intent": "depression"},
    {"text": "Everything feels pointless",                 "intent": "depression"},
    {"text": "I've lost interest in things I used to love","intent": "depression"},
    {"text": "I feel hopeless and numb",                   "intent": "depression"},
    {"text": "I'm exhausted all the time for no reason",   "intent": "depression"},
    # venting
    {"text": "I'm so sick of how my flatmate treats me",   "intent": "venting"},
    {"text": "My boss is an absolute nightmare and I'm furious", "intent": "venting"},
    {"text": "I just need to vent today was terrible",     "intent": "venting"},
    {"text": "I'm fed up with everything",                 "intent": "venting"},
    {"text": "I'm so pissed off right now",                "intent": "venting"},
    # seeking_advice
    {"text": "I don't know what to do about my relationship", "intent": "seeking_advice"},
    {"text": "Should I quit my job",                       "intent": "seeking_advice"},
    {"text": "What do you think I should do",              "intent": "seeking_advice"},
    {"text": "How do I deal with this",                    "intent": "seeking_advice"},
    {"text": "Any advice on how to handle this",           "intent": "seeking_advice"},
    # relationship
    {"text": "My partner and I keep having the same argument", "intent": "relationship"},
    {"text": "I feel so lonely even when I'm with people", "intent": "relationship"},
    {"text": "My mum doesn't understand me at all",        "intent": "relationship"},
    {"text": "I had a huge fight with my boyfriend",       "intent": "relationship"},
    {"text": "My friend betrayed my trust",                "intent": "relationship"},
    {"text": "I feel disconnected from my family",         "intent": "relationship"},
    # work_stress
    {"text": "I have three deadlines this week and I can't cope", "intent": "work_stress"},
    {"text": "I think I'm burning out from work",          "intent": "work_stress"},
    {"text": "My exams are in two weeks and I'm not ready","intent": "work_stress"},
    {"text": "My boss keeps piling on work",               "intent": "work_stress"},
    {"text": "I'm exhausted from overworking",             "intent": "work_stress"},
    {"text": "I stayed at the office until midnight again","intent": "work_stress"},
    # self_esteem
    {"text": "I hate how I look",                          "intent": "self_esteem"},
    {"text": "I'm not smart enough for this",              "intent": "self_esteem"},
    {"text": "Everyone else seems to have their life together", "intent": "self_esteem"},
    {"text": "I feel like I'm not good enough",            "intent": "self_esteem"},
    {"text": "I keep comparing myself to others",          "intent": "self_esteem"},
    # trauma
    {"text": "Something happened to me as a child that I've never talked about", "intent": "trauma"},
    {"text": "I keep having flashbacks from what happened", "intent": "trauma"},
    {"text": "I was abused and I still can't get past it", "intent": "trauma"},
    # gratitude
    {"text": "Talking to you really helped me last time",  "intent": "gratitude"},
    {"text": "Thank you I feel so much better",            "intent": "gratitude"},
    {"text": "I really appreciate you listening",          "intent": "gratitude"},
    # progress
    {"text": "I actually had a good week for once",        "intent": "progress"},
    {"text": "I stood up for myself today and it felt amazing", "intent": "progress"},
    {"text": "I've been sleeping better lately",           "intent": "progress"},
    # checking_in
    {"text": "Hey just checking in",                       "intent": "checking_in"},
    {"text": "Hi how are you",                             "intent": "checking_in"},
    {"text": "Good morning",                               "intent": "checking_in"},
    {"text": "I just wanted to say hello",                 "intent": "checking_in"},
    # general
    {"text": "I had a strange day",                        "intent": "general"},
    {"text": "Not sure where to start",                    "intent": "general"},
    {"text": "I've been thinking a lot lately",            "intent": "general"},
]


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"([^\w\s'])", r" \1 ", text)
    return text.split()


# rule-based DST tags (training labels for the token NER head)
_PERSON_WORDS  = {"mum", "mom", "dad", "father", "mother", "brother", "sister",
                  "friend", "partner", "boyfriend", "girlfriend", "husband", "wife",
                  "boss", "colleague", "teacher", "flatmate"}
_TOPIC_WORDS   = {"work", "job", "career", "office", "exam", "deadline", "school",
                  "university", "uni", "sleep", "health", "relationship", "money"}
_ISSUE_WORDS   = {"anxiety", "depression", "trauma", "abuse", "panic", "hopeless",
                  "suicide", "cutting", "self-harm", "ptsd", "burnout"}
_MOOD_WORDS    = {"anxious", "sad", "angry", "happy", "hopeless", "numb", "empty",
                  "overwhelmed", "exhausted", "furious", "pissed", "grateful", "better"}


def _tag(tokens: list[str]) -> list[str]:
    tags = []
    prev = "O"
    for tok in tokens:
        t = tok.strip("'.,!?")
        if t in _PERSON_WORDS:
            tags.append("B-PERSON" if prev != "I-PERSON" else "I-PERSON")
            prev = "I-PERSON"
        elif t in _TOPIC_WORDS:
            tags.append("B-TOPIC" if prev != "I-TOPIC" else "I-TOPIC")
            prev = "I-TOPIC"
        elif t in _ISSUE_WORDS:
            tags.append("B-ISSUE" if prev != "I-ISSUE" else "I-ISSUE")
            prev = "I-ISSUE"
        elif t in _MOOD_WORDS:
            tags.append("B-MOOD" if prev != "I-MOOD" else "I-MOOD")
            prev = "I-MOOD"
        else:
            tags.append("O")
            prev = "O"
    return tags


def main():
    random.seed(42)
    out = Path("training/data")
    out.mkdir(parents=True, exist_ok=True)

    records = []
    for item in SEED_DATA:
        tokens   = _tokenize(item["text"])
        dst_tags = _tag(tokens)
        records.append({
            "text":     item["text"],
            "intent":   item["intent"],
            "tokens":   tokens,
            "dst_tags": dst_tags,
        })

    random.shuffle(records)
    split     = int(len(records) * 0.85)
    train_rec = records[:split]
    val_rec   = records[split:]

    def write(path, data):
        Path(path).write_text("\n".join(json.dumps(r) for r in data))

    write(out / "multitask_train.jsonl", train_rec)
    write(out / "multitask_val.jsonl",   val_rec)
    print(f"Wrote {len(train_rec)} train / {len(val_rec)} val records.")
    print(f"Intents: {sorted({r['intent'] for r in records})}")


if __name__ == "__main__":
    main()
