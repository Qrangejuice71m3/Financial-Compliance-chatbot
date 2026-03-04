# Financial Compliance Chatbot (MiFID II + SFDR)

## Features

- Financial preference state machine (3 questions):
  - Risk tolerance (Low/Medium/High)
  - Investment horizon (Short/Medium/Long)
  - Loss tolerance (Low/Medium/High)
- ESG preference state machine (3 questions):
  - Environmental, Social, and Governance scoring (1-5)
- Natural-language profile extraction:
  - Uses LLM to extract financial and ESG profile slots from a single message and write them into `investor_profile`
  - If profile slots are incomplete, the bot asks follow-up questions automatically
  - Once complete, it returns recognized financial and ESG scores explicitly
- Natural-language Q&A:
  - Calls LLM API for user questions outside slot collection
- Trade-off interceptor:
  - `check_compliance_tension(user_risk_level, selected_product_risk)`
  - If product risk exceeds client risk capacity, it outputs MiFID II Art. 24/25 compliant tension wording
- Vector retrieval + EET-based answering:
  - For fund "green level" questions, the system retrieves from default EET data first, then answers based on retrieved disclosure
  - If EET does not mention "zero emission/net zero guarantee", it appends a mandatory disclaimer:
    - `This information is based on current disclosures and does not constitute a final carbon-neutral guarantee.`

## Run

```bash
cd /Users/juiceq/Desktop/金融合规
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 app.py
```

Open in browser: `http://127.0.0.1:8000/`

## LLM API Configuration (Optional)

Without LLM API config, the app still runs, but LLM-based answering falls back.

```bash
export OPENAI_API_KEY="your_key"
export OPENAI_MODEL="gpt-4o-mini"
# Optional custom gateway
# export OPENAI_BASE_URL="https://api.openai.com/v1/chat/completions"
```

You can also set `API Key / Base URL / Model` directly in the web UI (`LLM API Settings`). The values are stored in browser local storage and sent with `/chat` requests.

## Quick Test Flow

1. Input: `I want an investment plan` (starts financial profile flow)
2. Answer: `medium`, `long`, `medium`
3. Input: `I care about ESG` (starts ESG flow)
4. Answer: `5`, `4`, `3`
5. Input: `Recommend a high-risk green fund for me` (triggers Trade-off check)
6. Input: `How green is this fund?` (triggers EET retrieval)

## Mock Default EET Data

- File: `/Users/juiceq/Desktop/金融合规/eet_default.json`
- Contains mock official-style EET fields for 3 funds (SFDR class, taxonomy alignment, PAI, governance policy, zero-emission statement, etc.)
