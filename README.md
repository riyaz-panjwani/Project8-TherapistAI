# Project 8 — Intent Detection & Dialogue State Tracking
### Personal Therapist AI · Single Eternal Session

A conversational AI therapist that **remembers everything** across all your visits — one continuous session, no new-chat button, ever.

---

## Architecture

```
User browser (WebSocket)
        │
        ▼
FastAPI  ──►  RoBERTa Intent Classifier  ──►  13 therapy intents
        │
        ├──►  ConvBERT DST               ──►  mood / topics / people slots
        │
        └──►  SQLite (aiosqlite)         ──►  full message + profile history
```

### Intent labels
`venting · seeking_advice · anxiety · depression · crisis · gratitude · relationship · work_stress · self_esteem · checking_in · trauma · progress · general`

### Dialogue state slots
| Slot | Description |
|------|-------------|
| `current_mood` | positive / neutral / negative / crisis |
| `mood_score` | 0.0 – 1.0 running average |
| `active_topics` | relationship / work / academic / health / self-image / trauma |
| `mentioned_people` | "my mum", "my boss", etc. |
| `disclosed_issues` | permanent record of everything discussed |
| `needs_followup` | flagged when crisis / advice-seeking detected |

---

## Quickstart

```bash
# 1. Install dependencies
cd backend
pip install -r requirements.txt

# 2. Run the server (heuristic mode — no training needed)
uvicorn main:app --reload --port 8000

# 3. Open browser
open http://localhost:8000
```

## Training (optional — improves accuracy)

```bash
# Prepare data from MultiWOZ + seed therapy utterances
python training/scripts/prepare_data.py

# Fine-tune RoBERTa intent classifier (~30 min on GPU)
python training/scripts/train_intent.py

# Fine-tune ConvBERT DST (needs labelled NER data)
python training/scripts/train_dst.py
```

Once checkpoints exist at `training/checkpoints/intent/` and `training/checkpoints/dst/`, the server loads them automatically on next start.

## Tests

```bash
cd tests
pytest -v
```

---

## Key design decisions

- **No new-chat button** — user ID is minted once in `localStorage` and never reset. All history is permanent.
- **Crisis safety** — crisis intent always overrides other labels and appends real helpline numbers to the response.
- **Heuristic fallback** — the app is fully functional before training; RoBERTa/ConvBERT drop in as upgrades.
- **Socratic method** — responses never give unsolicited advice; they reflect and ask questions.
- **Memory sidebar** — live view of what the model has extracted: mood, topics, people mentioned.
