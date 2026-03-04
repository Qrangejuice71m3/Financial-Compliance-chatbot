from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request as urllib_request
from uuid import uuid4

from flask import Flask, jsonify, request as flask_request, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "frontend"
EET_DATA_PATH = BASE_DIR / "eet_default.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")

SYSTEM_PROMPT = """Role: You are a professional financial compliance investment assistant operating under MiFID II and SFDR.

Core Mission:
1. Identify and record the client's sustainability preference.
2. Ensure recommendations are aligned with the client's best financial interest.
3. Provide fair, clear, and non-misleading information; avoid greenwashing.

Strict Operational Rules:
1. Financial suitability first: confirm risk capacity before sustainability preference (MiFID II Art 25).
2. Tension handling: if sustainability preference implies excessive risk, explicitly state the tension and propose alternatives.
3. Non-misleading language: use legal terms such as "promotes environmental characteristics (SFDR Art 8)".
4. Grounding: all product statements must be based on available disclosures/knowledge base.
"""

FINANCE_QUESTIONS = [
    (
        "risk_level",
        "Q1/3 (Financial Risk): What is your risk tolerance level: Low / Medium / High?",
    ),
    (
        "investment_horizon",
        "Q2/3 (Investment Horizon): What is your investment horizon: Short (1-3y) / Medium (3-7y) / Long (7+y)?",
    ),
    (
        "loss_tolerance",
        "Q3/3 (Loss Tolerance): In stress conditions, what drawdown can you accept: Low (<=10%) / Medium (10%-25%) / High (>25%)?",
    ),
]

ESG_QUESTIONS = [
    (
        "environment",
        "Q1/3 (Environmental Goal): How important are environmental objectives (e.g. decarbonization, clean energy)? Please answer 1-5.",
    ),
    (
        "social",
        "Q2/3 (Social Goal): How important are social objectives (e.g. labor rights, inclusion)? Please answer 1-5.",
    ),
    (
        "governance",
        "Q3/3 (Governance Goal): How important are governance objectives (e.g. board independence, anti-corruption)? Please answer 1-5.",
    ),
]

FINANCE_TRIGGER_KEYWORDS = ["投资", "理财", "资产配置", "推荐", "financial", "risk", "风险"]
ESG_TRIGGER_KEYWORDS = ["esg", "环保", "可持续", "绿色投资", "sustainability"]

RISK_LEVEL_MAP = {"low": 1, "medium": 2, "high": 3}


@dataclass
class SessionState:
    session_id: str
    dialogue_state: str = "idle"
    question_index: int = -1
    investor_profile: dict[str, Any] = field(
        default_factory=lambda: {
            "financial_preference": {
                "risk_level": None,
                "investment_horizon": None,
                "loss_tolerance": None,
                "completed": False,
            },
            "sustainability_preference": {
                "environment": None,
                "social": None,
                "governance": None,
                "total_score": None,
                "level": None,
                "completed": False,
            },
        }
    )
    history: list[dict[str, Any]] = field(default_factory=list)
    audit_log: list[dict[str, Any]] = field(default_factory=list)


app = Flask(__name__, static_folder=None)
SESSIONS: dict[str, SessionState] = {}
EET_RECORDS: list[dict[str, Any]] = []
EET_INDEX: list[dict[str, Any]] = []


def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def add_audit(session: SessionState, event: str, detail: str) -> None:
    session.audit_log.append({"timestamp": utc_ts(), "event": event, "detail": detail})


def get_or_create_session(session_id: str | None) -> SessionState:
    if session_id and session_id in SESSIONS:
        return SESSIONS[session_id]
    sid = session_id or str(uuid4())
    s = SessionState(session_id=sid)
    SESSIONS[sid] = s
    return s


def load_default_eet_data() -> list[dict[str, Any]]:
    if EET_DATA_PATH.exists():
        try:
            return json.loads(EET_DATA_PATH.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return []


def tokenize_for_vector(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{1,4}", text.lower())
    return [t for t in tokens if t.strip()]


def to_term_vector(tokens: list[str]) -> dict[str, float]:
    vec: dict[str, float] = {}
    for tok in tokens:
        vec[tok] = vec.get(tok, 0.0) + 1.0
    return vec


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(a[k] * b.get(k, 0.0) for k in a)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build_eet_text_blob(rec: dict[str, Any]) -> str:
    fields = [
        str(rec.get("fund_name", "")),
        str(rec.get("sfdr_classification", "")),
        str(rec.get("environmental_objective", "")),
        str(rec.get("social_objective", "")),
        str(rec.get("governance_policy", "")),
        str(rec.get("taxonomy_alignment_pct", "")),
        str(rec.get("pai_considered", "")),
        str(rec.get("zero_emission_statement", "")),
        str(rec.get("official_source", "")),
    ]
    return " ".join(fields)


def init_eet_vector_index() -> None:
    global EET_RECORDS, EET_INDEX
    EET_RECORDS = load_default_eet_data()
    EET_INDEX = []
    for rec in EET_RECORDS:
        text_blob = build_eet_text_blob(rec)
        EET_INDEX.append({"record": rec, "vector": to_term_vector(tokenize_for_vector(text_blob))})


def contains_keyword(text: str, keywords: list[str]) -> bool:
    lower_txt = text.lower()
    return any(k in lower_txt for k in keywords)


def is_green_degree_query(text: str) -> bool:
    kws = ["绿色程度", "绿色", "esg程度", "环保程度", "碳排放", "碳中和", "零排放", "sfdr", "eet", "基金"]
    return contains_keyword(text, kws) and any(k in text.lower() for k in ["基金", "fund", "绿色", "esg", "环保"])


def retrieve_eet_record(query: str) -> tuple[dict[str, Any] | None, float]:
    if not EET_INDEX:
        return None, 0.0
    q_vec = to_term_vector(tokenize_for_vector(query))
    best_score = -1.0
    best_rec: dict[str, Any] | None = None
    for item in EET_INDEX:
        score = cosine_similarity(q_vec, item["vector"])
        if score > best_score:
            best_score = score
            best_rec = item["record"]
    return best_rec, max(best_score, 0.0)


def record_mentions_zero_emission(rec: dict[str, Any]) -> bool:
    txt = build_eet_text_blob(rec).lower()
    tags = ["零排放", "zero emission", "net zero", "净零", "碳中和"]
    return any(t in txt for t in tags)


def parse_risk_level(text: str) -> str | None:
    txt = text.lower()
    if any(k in txt for k in ["low", "低", "保守", "谨慎", "风险小", "低波动", "中低"]):
        return "Low"
    if any(k in txt for k in ["medium", "中", "稳健", "平衡", "中等风险"]):
        return "Medium"
    if any(k in txt for k in ["high", "高", "激进", "进取", "高波动", "中高"]):
        return "High"
    return None


def parse_zh_number(text: str) -> int | None:
    mapping = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
    if text == "十":
        return 10
    if text.startswith("十") and len(text) == 2 and text[1] in mapping:
        return 10 + mapping[text[1]]
    if text.endswith("十") and len(text) == 2 and text[0] in mapping:
        return mapping[text[0]] * 10
    if len(text) == 3 and text[1] == "十" and text[0] in mapping and text[2] in mapping:
        return mapping[text[0]] * 10 + mapping[text[2]]
    if len(text) == 1 and text in mapping:
        return mapping[text]
    return None


def parse_investment_horizon(text: str) -> str | None:
    txt = text.lower()
    m = re.search(r"(\d{1,2})\s*年", txt)
    if m:
        years = int(m.group(1))
        if years <= 3:
            return "Short"
        if years <= 7:
            return "Medium"
        return "Long"
    m_zh = re.search(r"([一二两三四五六七八九十]{1,3})\s*年", text)
    if m_zh:
        years = parse_zh_number(m_zh.group(1))
        if years is not None:
            if years <= 3:
                return "Short"
            if years <= 7:
                return "Medium"
            return "Long"
    if any(k in txt for k in ["短期", "1-3年", "short"]):
        return "Short"
    if any(k in txt for k in ["中期", "3-7年", "medium term", "mid"]):
        return "Medium"
    if any(k in txt for k in ["长期", "7年以上", "long"]):
        return "Long"
    return None


def parse_loss_tolerance(text: str) -> str | None:
    txt = text.lower()
    perc = re.findall(r"(\d{1,2})\s*%", txt)
    if not perc:
        perc = re.findall(r"(\d{1,2})\s*(?:个点|点)", text)
    if perc:
        val = max(int(p) for p in perc)
        if val <= 10:
            return "Low"
        if val <= 25:
            return "Medium"
        return "High"
    m_zh = re.search(r"([一二两三四五六七八九十]{1,3})\s*(?:个点|点|百分比?)", text)
    if m_zh:
        val = parse_zh_number(m_zh.group(1))
        if val is not None:
            if val <= 10:
                return "Low"
            if val <= 25:
                return "Medium"
            return "High"
    if any(k in txt for k in ["亏10", "回撤10", "低", "low", "小幅"]):
        return "Low"
    if any(k in txt for k in ["10%-25", "中", "medium", "适中"]):
        return "Medium"
    if any(k in txt for k in [">25", "高", "high", "可以承受大波动"]):
        return "High"
    return None


def parse_score_1_to_5(text: str) -> int | None:
    txt = text.lower()
    for num in [1, 2, 3, 4, 5]:
        if str(num) in txt:
            return num
    zh_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}
    for token, value in zh_map.items():
        if token in txt:
            return value
    return None


def infer_esg_score_from_phrase(text: str) -> int | None:
    txt = text.lower()
    if any(k in txt for k in ["非常重视", "特别重视", "最重视", "最高", "很高", "强烈"]):
        return 5
    if any(k in txt for k in ["重视", "较高", "偏高", "关注"]):
        return 4
    if any(k in txt for k in ["一般", "中等", "还行", "普通"]):
        return 3
    if any(k in txt for k in ["较低", "不太", "偏低", "一般般"]):
        return 2
    if any(k in txt for k in ["不关注", "不考虑", "不在意", "完全不"]):
        return 1
    return None


def extract_dimension_score(text: str, aliases: list[str]) -> int | None:
    for alias in aliases:
        p1 = re.search(rf"{alias}\D{{0,6}}([1-5])\s*分", text, flags=re.IGNORECASE)
        if p1:
            return int(p1.group(1))
        p2 = re.search(rf"([1-5])\s*分\D{{0,6}}{alias}", text, flags=re.IGNORECASE)
        if p2:
            return int(p2.group(1))
        p3 = re.search(rf"{alias}\D{{0,8}}([一二三四五])\s*分?", text, flags=re.IGNORECASE)
        if p3:
            return {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5}.get(p3.group(1))
        idx = text.lower().find(alias.lower())
        if idx >= 0:
            win = text[max(0, idx - 8) : idx + len(alias) + 10]
            score = infer_esg_score_from_phrase(win)
            if score:
                return score
    return None


def score_level(total: int) -> str:
    if total <= 6:
        return "Low ESG Preference"
    if total <= 10:
        return "Medium ESG Preference"
    return "High ESG Preference"


def build_state(session: SessionState) -> dict[str, Any]:
    return {
        "dialogue_state": session.dialogue_state,
        "question_index": session.question_index,
        "investor_profile": session.investor_profile,
    }


def get_source_trace() -> list[dict[str, str]]:
    return [{"file": "04_regulatory_mapping.csv", "row_ref": "Article 24 / 25 / 25a"}]


def start_financial_flow(session: SessionState) -> str:
    session.dialogue_state = "finance"
    session.question_index = 0
    add_audit(session, "financial_flow_started", "Started financial preference questionnaire.")
    return "Financial preference capture started (MiFID II Art 25).\n" + FINANCE_QUESTIONS[0][1]


def start_esg_flow(session: SessionState) -> str:
    session.dialogue_state = "esg"
    session.question_index = 0
    add_audit(session, "esg_flow_started", "Started ESG preference questionnaire.")
    return "Sustainability preference capture started (ESMA / SFDR).\n" + ESG_QUESTIONS[0][1]


def check_compliance_tension(user_risk_level: str, selected_product_risk: str) -> str:
    """
    If product risk is higher than user risk capacity, return compliance tension wording.
    """
    user_v = RISK_LEVEL_MAP.get(user_risk_level.lower(), 0)
    product_v = RISK_LEVEL_MAP.get(selected_product_risk.lower(), 0)
    if product_v > user_v:
        return (
            "Tension detected: although this product aligns with your sustainability preference, "
            "its risk level is above your financial risk assessment result. "
            "To balance MiFID II Article 24 (fair, clear, non-misleading information) and Article 25 (suitability), "
            "we recommend prioritizing alternatives with risk aligned to your profile. "
            "To protect your best interest, we suggest limiting this type of asset to 10% or less of allocation."
        )
    return ""


def parse_selected_product_risk(text: str) -> str | None:
    txt = text.lower()
    if any(k in txt for k in ["低风险", "low risk", "conservative", "保守型", "低波动"]):
        return "Low"
    if any(k in txt for k in ["中风险", "medium risk", "balanced", "稳健型", "中等波动"]):
        return "Medium"
    if any(
        k in txt
        for k in [
            "高风险",
            "high risk",
            "aggressive",
            "进取型",
            "高波动",
            "高收益",
            "高回报",
            "高beta",
            "高杠杆",
        ]
    ):
        return "High"
    return None


def llm_detect_product_risk(user_message: str, api_key: str, model: str, base_url: str) -> str | None:
    """
    Let LLM infer the discussed product risk level from natural language.
    Return one of: Low / Medium / High / None.
    """
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a risk classifier for financial product descriptions. "
                    "From the user text, infer the product risk level and output JSON only."
                ),
            },
            {
                "role": "system",
                "content": 'Schema: {"selected_product_risk":"Low|Medium|High|None","confidence":"0-1"}',
            },
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
    }
    req = urllib_request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            if not isinstance(parsed, dict):
                return None
            val = str(parsed.get("selected_product_risk", "")).strip().lower()
            if val in {"low", "medium", "high"}:
                return val.capitalize()
            return None
    except (error.URLError, KeyError, ValueError):
        return None


def extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except ValueError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except ValueError:
        return None


def call_llm(
    session: SessionState,
    user_message: str,
    tension_note: str,
    api_key: str,
    model: str,
    base_url: str,
) -> str | None:
    if not api_key:
        return None

    context = {
        "investor_profile": session.investor_profile,
        "tension_note": tension_note,
    }
    tension_policy = (
        "If tension_note is not empty, you MUST explicitly state the conflict: "
        "the product matches sustainability preference but exceeds financial risk capacity, "
        "and you MUST include allocation guidance: limit this type of asset to 10% or less."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "system",
                "content": (
                    "Answer using investor_profile context. "
                    "If tension_note exists, state the compliance warning first and then alternatives. "
                    f"{tension_policy}"
                    f"Context: {json.dumps(context, ensure_ascii=False)}"
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.2,
    }

    req = urllib_request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except (error.URLError, KeyError, ValueError):
        return None


def call_llm_for_eet(
    user_message: str,
    eet_record: dict[str, Any],
    api_key: str,
    model: str,
    base_url: str,
) -> str | None:
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a financial compliance assistant. "
                    "Answer strictly based on the provided EET data and do not add undisclosed facts. "
                    "Use SFDR/ESG terminology and keep language fair, clear, and non-misleading."
                ),
            },
            {
                "role": "system",
                "content": f"EET data: {json.dumps(eet_record, ensure_ascii=False)}",
            },
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.1,
    }
    req = urllib_request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return body["choices"][0]["message"]["content"]
    except (error.URLError, KeyError, ValueError):
        return None


def llm_extract_profiles(
    session: SessionState,
    user_message: str,
    api_key: str,
    model: str,
    base_url: str,
) -> dict[str, Any] | None:
    if not api_key:
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a profile extraction engine. "
                    "Extract financial_preference and sustainability_preference from user text. "
                    "Return JSON only."
                ),
            },
            {
                "role": "system",
                "content": (
                    'JSON schema: {"financial_preference":{"risk_level":"Low|Medium|High|null",'
                    '"investment_horizon":"Short|Medium|Long|null","loss_tolerance":"Low|Medium|High|null"},'
                    '"sustainability_preference":{"environment":"1-5|null","social":"1-5|null","governance":"1-5|null"}}'
                ),
            },
            {
                "role": "system",
                "content": f"Current known profile: {json.dumps(session.investor_profile, ensure_ascii=False)}",
            },
            {"role": "user", "content": user_message},
        ],
        "temperature": 0,
    }
    req = urllib_request.Request(
        base_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib_request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            parsed = extract_json_object(content)
            return parsed if isinstance(parsed, dict) else None
    except (error.URLError, KeyError, ValueError):
        return None


def normalize_risk_value(value: Any) -> str | None:
    if value is None:
        return None
    txt = str(value).strip().lower()
    if txt in {"low", "l"}:
        return "Low"
    if txt in {"medium", "mid", "m"}:
        return "Medium"
    if txt in {"high", "h"}:
        return "High"
    return None


def normalize_horizon_value(value: Any) -> str | None:
    if value is None:
        return None
    txt = str(value).strip().lower()
    if txt in {"short", "short-term", "short term"}:
        return "Short"
    if txt in {"medium", "mid", "medium-term", "medium term"}:
        return "Medium"
    if txt in {"long", "long-term", "long term"}:
        return "Long"
    return None


def normalize_score_value(value: Any) -> int | None:
    if value is None:
        return None
    try:
        score = int(str(value).strip())
    except ValueError:
        return None
    return score if 1 <= score <= 5 else None


def apply_extracted_profiles(session: SessionState, extracted: dict[str, Any] | None) -> list[str]:
    if not extracted:
        return []

    updated_fields: list[str] = []
    fp = session.investor_profile["financial_preference"]
    sp = session.investor_profile["sustainability_preference"]
    extracted_fp = extracted.get("financial_preference") if isinstance(extracted.get("financial_preference"), dict) else {}
    extracted_sp = (
        extracted.get("sustainability_preference") if isinstance(extracted.get("sustainability_preference"), dict) else {}
    )

    risk_level = normalize_risk_value(extracted_fp.get("risk_level"))
    horizon = normalize_horizon_value(extracted_fp.get("investment_horizon"))
    loss = normalize_risk_value(extracted_fp.get("loss_tolerance"))

    if risk_level and fp["risk_level"] != risk_level:
        fp["risk_level"] = risk_level
        updated_fields.append("financial_preference.risk_level")
    if horizon and fp["investment_horizon"] != horizon:
        fp["investment_horizon"] = horizon
        updated_fields.append("financial_preference.investment_horizon")
    if loss and fp["loss_tolerance"] != loss:
        fp["loss_tolerance"] = loss
        updated_fields.append("financial_preference.loss_tolerance")

    env_score = normalize_score_value(extracted_sp.get("environment"))
    soc_score = normalize_score_value(extracted_sp.get("social"))
    gov_score = normalize_score_value(extracted_sp.get("governance"))
    if env_score and sp["environment"] != env_score:
        sp["environment"] = env_score
        updated_fields.append("sustainability_preference.environment")
    if soc_score and sp["social"] != soc_score:
        sp["social"] = soc_score
        updated_fields.append("sustainability_preference.social")
    if gov_score and sp["governance"] != gov_score:
        sp["governance"] = gov_score
        updated_fields.append("sustainability_preference.governance")

    fp_completed = bool(fp["risk_level"] and fp["investment_horizon"] and fp["loss_tolerance"])
    if fp_completed and not fp["completed"]:
        updated_fields.append("financial_preference.completed")
    fp["completed"] = fp_completed

    if sp["environment"] and sp["social"] and sp["governance"]:
        total = int(sp["environment"]) + int(sp["social"]) + int(sp["governance"])
        level = score_level(total)
        if sp["total_score"] != total:
            sp["total_score"] = total
            updated_fields.append("sustainability_preference.total_score")
        if sp["level"] != level:
            sp["level"] = level
            updated_fields.append("sustainability_preference.level")
        if not sp["completed"]:
            sp["completed"] = True
            updated_fields.append("sustainability_preference.completed")

    if updated_fields:
        add_audit(session, "profile_extracted_by_llm", ", ".join(updated_fields))
    return updated_fields


def next_finance_question(session: SessionState) -> str:
    fp = session.investor_profile["financial_preference"]
    if not fp["risk_level"]:
        session.dialogue_state = "finance"
        session.question_index = 0
        return FINANCE_QUESTIONS[0][1]
    if not fp["investment_horizon"]:
        session.dialogue_state = "finance"
        session.question_index = 1
        return FINANCE_QUESTIONS[1][1]
    if not fp["loss_tolerance"]:
        session.dialogue_state = "finance"
        session.question_index = 2
        return FINANCE_QUESTIONS[2][1]
    return ""


def next_esg_question(session: SessionState) -> str:
    sp = session.investor_profile["sustainability_preference"]
    if not sp["environment"]:
        session.dialogue_state = "esg"
        session.question_index = 0
        return ESG_QUESTIONS[0][1]
    if not sp["social"]:
        session.dialogue_state = "esg"
        session.question_index = 1
        return ESG_QUESTIONS[1][1]
    if not sp["governance"]:
        session.dialogue_state = "esg"
        session.question_index = 2
        return ESG_QUESTIONS[2][1]
    return ""


def recompute_profile_flags(session: SessionState) -> None:
    fp = session.investor_profile["financial_preference"]
    sp = session.investor_profile["sustainability_preference"]
    fp["completed"] = bool(fp["risk_level"] and fp["investment_horizon"] and fp["loss_tolerance"])

    if sp["environment"] and sp["social"] and sp["governance"]:
        total = int(sp["environment"]) + int(sp["social"]) + int(sp["governance"])
        sp["total_score"] = total
        sp["level"] = score_level(total)
        sp["completed"] = True
    else:
        sp["completed"] = False


def apply_natural_language_profile_extraction(session: SessionState, user_message: str) -> list[str]:
    updated: list[str] = []
    fp = session.investor_profile["financial_preference"]
    sp = session.investor_profile["sustainability_preference"]
    txt = user_message.strip()

    risk = parse_risk_level(txt)
    horizon = parse_investment_horizon(txt)
    loss = parse_loss_tolerance(txt)
    env = extract_dimension_score(txt, ["环境", "环保", "environment", "e维度"])
    soc = extract_dimension_score(txt, ["社会", "social", "s维度"])
    gov = extract_dimension_score(txt, ["治理", "governance", "g维度"])

    if risk and fp["risk_level"] != risk:
        fp["risk_level"] = risk
        updated.append("financial_preference.risk_level")
    if horizon and fp["investment_horizon"] != horizon:
        fp["investment_horizon"] = horizon
        updated.append("financial_preference.investment_horizon")
    if loss and fp["loss_tolerance"] != loss:
        fp["loss_tolerance"] = loss
        updated.append("financial_preference.loss_tolerance")
    if env and sp["environment"] != env:
        sp["environment"] = env
        updated.append("sustainability_preference.environment")
    if soc and sp["social"] != soc:
        sp["social"] = soc
        updated.append("sustainability_preference.social")
    if gov and sp["governance"] != gov:
        sp["governance"] = gov
        updated.append("sustainability_preference.governance")

    recompute_profile_flags(session)
    if updated:
        add_audit(session, "profile_extracted_by_rules", ", ".join(updated))
    return updated


def get_next_missing_prompt(session: SessionState) -> str:
    fp = session.investor_profile["financial_preference"]
    sp = session.investor_profile["sustainability_preference"]

    if not fp["risk_level"]:
        session.dialogue_state = "finance"
        session.question_index = 0
        return FINANCE_QUESTIONS[0][1]
    if not fp["investment_horizon"]:
        session.dialogue_state = "finance"
        session.question_index = 1
        return FINANCE_QUESTIONS[1][1]
    if not fp["loss_tolerance"]:
        session.dialogue_state = "finance"
        session.question_index = 2
        return FINANCE_QUESTIONS[2][1]
    if not sp["environment"]:
        session.dialogue_state = "esg"
        session.question_index = 0
        return ESG_QUESTIONS[0][1]
    if not sp["social"]:
        session.dialogue_state = "esg"
        session.question_index = 1
        return ESG_QUESTIONS[1][1]
    if not sp["governance"]:
        session.dialogue_state = "esg"
        session.question_index = 2
        return ESG_QUESTIONS[2][1]
    session.dialogue_state = "idle"
    session.question_index = -1
    return ""


def build_profile_summary_text(session: SessionState) -> str:
    fp = session.investor_profile["financial_preference"]
    sp = session.investor_profile["sustainability_preference"]
    return (
        "Recognized profile summary:\n"
        f"- Financial profile: risk={fp['risk_level']}, horizon={fp['investment_horizon']}, loss={fp['loss_tolerance']}\n"
        f"- ESG profile: E={sp['environment']}, S={sp['social']}, G={sp['governance']}, total={sp['total_score']}, level={sp['level']}"
    )


def apply_contextual_answer_extraction(session: SessionState, user_message: str) -> list[str]:
    """Capture short answers in questionnaire state while still allowing global parsing."""
    updated: list[str] = []
    fp = session.investor_profile["financial_preference"]
    sp = session.investor_profile["sustainability_preference"]

    if session.dialogue_state == "finance" and 0 <= session.question_index < len(FINANCE_QUESTIONS):
        field_name, _ = FINANCE_QUESTIONS[session.question_index]
        parsed_value = None
        if field_name == "risk_level":
            parsed_value = parse_risk_level(user_message)
        elif field_name == "investment_horizon":
            parsed_value = parse_investment_horizon(user_message)
        elif field_name == "loss_tolerance":
            parsed_value = parse_loss_tolerance(user_message)
        if parsed_value and fp.get(field_name) != parsed_value:
            fp[field_name] = parsed_value
            updated.append(f"financial_preference.{field_name}")

    if session.dialogue_state == "esg" and 0 <= session.question_index < len(ESG_QUESTIONS):
        field_name, _ = ESG_QUESTIONS[session.question_index]
        score = parse_score_1_to_5(user_message) or infer_esg_score_from_phrase(user_message)
        if score and sp.get(field_name) != score:
            sp[field_name] = score
            updated.append(f"sustainability_preference.{field_name}")

    if updated:
        recompute_profile_flags(session)
        add_audit(session, "profile_extracted_by_context", ", ".join(updated))
    return updated


def resolve_llm_config(payload: dict[str, Any]) -> tuple[str, str, str]:
    llm_cfg = payload.get("llm_config") if isinstance(payload.get("llm_config"), dict) else {}
    api_key = str(llm_cfg.get("api_key") or payload.get("api_key") or OPENAI_API_KEY).strip()
    model = str(llm_cfg.get("model") or payload.get("api_model") or OPENAI_MODEL).strip() or OPENAI_MODEL
    base_url = str(llm_cfg.get("base_url") or payload.get("api_base_url") or OPENAI_BASE_URL).strip() or OPENAI_BASE_URL
    return api_key, model, base_url


def handle_finance_question(session: SessionState, user_message: str) -> str:
    field_name, _ = FINANCE_QUESTIONS[session.question_index]
    fp = session.investor_profile["financial_preference"]

    parsed_value = None
    if field_name == "risk_level":
        parsed_value = parse_risk_level(user_message)
    elif field_name == "investment_horizon":
        parsed_value = parse_investment_horizon(user_message)
    elif field_name == "loss_tolerance":
        parsed_value = parse_loss_tolerance(user_message)

    if not parsed_value:
        return f"Could not parse a valid answer. Please try again.\n{FINANCE_QUESTIONS[session.question_index][1]}"

    fp[field_name] = parsed_value
    add_audit(session, "financial_answer_recorded", f"{field_name}={parsed_value}")

    if session.question_index < len(FINANCE_QUESTIONS) - 1:
        session.question_index += 1
        return FINANCE_QUESTIONS[session.question_index][1]

    fp["completed"] = True
    session.question_index = -1
    add_audit(session, "financial_profile_completed", "Financial preference questionnaire completed.")
    return (
        "Financial profile capture completed and stored in investor_profile.\n"
        f"Risk tolerance: {fp['risk_level']}, horizon: {fp['investment_horizon']}, loss tolerance: {fp['loss_tolerance']}.\n"
        "If you want to continue with sustainability profiling, mention ESG or sustainability."
    )


def handle_esg_question(session: SessionState, user_message: str) -> str:
    field_name, _ = ESG_QUESTIONS[session.question_index]
    sp = session.investor_profile["sustainability_preference"]
    score = parse_score_1_to_5(user_message)
    if score is None:
        return f"Could not parse a valid score. Please answer 1-5.\n{ESG_QUESTIONS[session.question_index][1]}"

    sp[field_name] = score
    add_audit(session, "esg_score_recorded", f"{field_name}={score}")

    if session.question_index < len(ESG_QUESTIONS) - 1:
        session.question_index += 1
        return ESG_QUESTIONS[session.question_index][1]

    total = (sp["environment"] or 0) + (sp["social"] or 0) + (sp["governance"] or 0)
    sp["total_score"] = total
    sp["level"] = score_level(total)
    sp["completed"] = True
    session.question_index = -1
    add_audit(session, "esg_profile_completed", f"total_score={total}, level={sp['level']}")
    return (
        "Sustainability profile capture completed and stored in investor_profile.\n"
        f"E={sp['environment']}, S={sp['social']}, G={sp['governance']}, total={total}, level={sp['level']}."
    )


@app.get("/")
def index() -> Any:
    return send_from_directory(str(STATIC_DIR), "index.html")


@app.get("/static/<path:filename>")
def serve_static(filename: str) -> Any:
    return send_from_directory(str(STATIC_DIR), filename)


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/chat")
def chat() -> Any:
    payload = flask_request.get_json(silent=True) or {}
    user_message = str(payload.get("user_message", "")).strip()
    session = get_or_create_session(payload.get("session_id") if isinstance(payload.get("session_id"), str) else None)
    session.history.append({"role": "user", "content": user_message, "timestamp": utc_ts()})
    api_key, api_model, api_base_url = resolve_llm_config(payload)

    reply = ""
    source_trace = get_source_trace()

    # Green-degree query path: MUST retrieve EET first, then answer.
    if is_green_degree_query(user_message):
        rec, score = retrieve_eet_record(user_message)
        if rec is None:
            reply = "No EET disclosure data was retrieved, so the fund's green level cannot be assessed at this time."
            add_audit(session, "eet_retrieval_failed", "No EET record matched.")
        else:
            llm_answer = call_llm_for_eet(
                user_message=user_message,
                eet_record=rec,
                api_key=api_key,
                model=api_model,
                base_url=api_base_url,
            )
            if llm_answer:
                reply = llm_answer
            else:
                reply = (
                    f"Based on retrieved official EET disclosure, {rec.get('fund_name')} is classified as {rec.get('sfdr_classification')} under SFDR, "
                    f"with taxonomy alignment around {rec.get('taxonomy_alignment_pct')}%, "
                    f"and PAI consideration status: {rec.get('pai_considered')}."
                )
            if not record_mentions_zero_emission(rec):
                reply += "\n\nThis information is based on current disclosures and does not constitute a final carbon-neutral guarantee."
            add_audit(
                session,
                "eet_retrieved",
                f"fund={rec.get('fund_name')}, similarity={score:.3f}",
            )
            source_trace = [
                {"file": "eet_default.json", "row_ref": str(rec.get("fund_name", "unknown"))},
                {"file": "04_regulatory_mapping.csv", "row_ref": "Article 24 (non-misleading)"},
            ]
        session.history.append({"role": "assistant", "content": reply, "timestamp": utc_ts()})
        response = {
            "session_id": session.session_id,
            "reply": reply,
            "state": build_state(session),
            "audit_log": session.audit_log,
            "source_trace": source_trace,
            "system_prompt_used": SYSTEM_PROMPT,
        }
        return jsonify(response)

    llm_extracted = llm_extract_profiles(
        session=session,
        user_message=user_message,
        api_key=api_key,
        model=api_model,
        base_url=api_base_url,
    )
    llm_updated = apply_extracted_profiles(session, llm_extracted)
    updated_fields = list(llm_updated)

    # Fallback: if LLM extraction is unavailable/empty, use local rule parser.
    if not updated_fields:
        context_updated = apply_contextual_answer_extraction(session, user_message)
        rule_updated = apply_natural_language_profile_extraction(session, user_message)
        updated_fields = context_updated + [f for f in rule_updated if f not in context_updated]

    # Detect product risk request early so tension can be surfaced even before profile is fully complete.
    selected_risk_for_tension = llm_detect_product_risk(
        user_message=user_message,
        api_key=api_key,
        model=api_model,
        base_url=api_base_url,
    ) or parse_selected_product_risk(user_message)

    missing_prompt = get_next_missing_prompt(session)
    fp = session.investor_profile["financial_preference"]
    sp = session.investor_profile["sustainability_preference"]

    if missing_prompt:
        prefix = "Part of your profile has been recognized. Please complete the missing slots.\n"
        if updated_fields:
            prefix = "Recognized and updated fields:\n- " + "\n- ".join(updated_fields) + "\n"
        reply = (
            f"{prefix}To complete your profile, please provide:\n{missing_prompt}\n"
            'You can answer in natural language, e.g. "I am moderate, 5+ years, max drawdown 20%."'
        )
        if selected_risk_for_tension:
            user_risk_for_tension = fp["risk_level"] or "Low"
            provisional_tension = check_compliance_tension(user_risk_for_tension, selected_risk_for_tension)
            if provisional_tension:
                reply = (
                    provisional_tension
                    + "\n\n(Preliminary check based on current profile data.)\n\n"
                    + reply
                )
    else:
        if updated_fields:
            add_audit(session, "profile_fully_completed", "Financial and ESG profiles completed.")
            reply = build_profile_summary_text(session)
            # 用户在同一轮是提问句时，补充回答，但先确认识别分数。
            if "?" in user_message or "？" in user_message:
                user_risk = fp["risk_level"] or "Low"
                selected_risk = selected_risk_for_tension
                tension_text = check_compliance_tension(user_risk, selected_risk) if selected_risk else ""
                llm_answer = call_llm(
                    session=session,
                    user_message=user_message,
                    tension_note=tension_text,
                    api_key=api_key,
                    model=api_model,
                    base_url=api_base_url,
                )
                if llm_answer:
                    reply = reply + "\n\n" + (tension_text + "\n\n" if tension_text else "") + llm_answer
        else:
            user_risk = fp["risk_level"] or "Low"
            selected_risk = selected_risk_for_tension
            if selected_risk:
                add_audit(
                    session,
                    "product_risk_detected",
                    f"user_risk={user_risk}, selected_product_risk={selected_risk}",
                )
            tension_text = check_compliance_tension(user_risk, selected_risk) if selected_risk else ""
            llm_answer = call_llm(
                session=session,
                user_message=user_message,
                tension_note=tension_text,
                api_key=api_key,
                model=api_model,
                base_url=api_base_url,
            )
            if llm_answer:
                reply = (tension_text + "\n\n" if tension_text else "") + llm_answer
                add_audit(session, "llm_answered", f"tension={bool(tension_text)}")
            else:
                reply = "Profile is complete, but the LLM call failed. Please check API settings."
                add_audit(session, "fallback_reply", "LLM unavailable or not configured.")

    session.history.append({"role": "assistant", "content": reply, "timestamp": utc_ts()})
    response = {
        "session_id": session.session_id,
        "reply": reply,
        "state": build_state(session),
        "audit_log": session.audit_log,
        "source_trace": source_trace,
        "system_prompt_used": SYSTEM_PROMPT,
    }
    return jsonify(response)


init_eet_vector_index()


if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
