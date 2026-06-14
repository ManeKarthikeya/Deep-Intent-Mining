
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None


st.set_page_config(
    page_title="Deep Intent Mining from Student Doubt Messages",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Deep Intent Mining from Student Doubt Messages"
APP_SUBTITLE = "Classify student doubt text into topic + hidden intent + emotion + urgency signals."

TOPIC_LABELS = [
    "Mathematics",
    "Programming",
    "Data Structures",
    "Machine Learning",
    "NLP",
    "Database / SQL",
    "Computer Networks",
    "Operating Systems",
    "Electronics",
    "Other",
]

INTENT_LABELS = [
    "Concept misunderstanding",
    "Confusion",
    "Frustration",
    "Exam anxiety",
    "Assignment/Submission stress",
    "Debugging / implementation issue",
    "Need for explanation",
    "Resource request",
    "Deadline pressure",
    "Clarification request",
]

EMOTION_LABELS = [
    "confusion",
    "frustration",
    "anxiety",
    "curiosity",
    "confidence",
    "neutral",
    "sadness",
    "stress",
]

TOPIC_PROTOTYPES = {
    "Mathematics": [
        "algebra calculus probability statistics integration differentiation theorem proof equation formula matrix vector derivative limit",
        "math problem solving numerical derivation graph geometry trigonometry",
    ],
    "Programming": [
        "python c++ java code function class loop syntax runtime compile error bug debugging variable object library",
        "program code not working compile error runtime exception",
    ],
    "Data Structures": [
        "array linked list stack queue tree graph heap recursion traversal sorting searching complexity data structure",
        "insert delete pointer node algorithm",
    ],
    "Machine Learning": [
        "model training accuracy overfitting underfitting classification regression dataset feature loss validation neural network",
        "machine learning prediction preprocessing pipeline evaluation",
    ],
    "NLP": [
        "tokenization embedding transformer bert roberta text classification sentiment intent entity language model prompt",
        "natural language processing sentence embeddings topic modeling",
    ],
    "Database / SQL": [
        "sql query select join group by aggregation table database normalization primary key foreign key index transaction",
        "mysql postgresql query error schema relation",
    ],
    "Computer Networks": [
        "tcp udp ip subnet routing switching protocol dns http packet bandwidth latency client server",
        "network topology socket firewall",
    ],
    "Operating Systems": [
        "process thread scheduling deadlock semaphore mutex memory paging virtual memory filesystem kernel",
        "os synchronization context switch",
    ],
    "Electronics": [
        "circuit voltage current resistance capacitor transistor diode op amp logic gate electronics signal",
        "digital analog circuit design",
    ],
    "Other": [
        "student doubt question clarification topic unsure please help explain",
    ],
}

INTENT_PROTOTYPES = {
    "Concept misunderstanding": [
        "I don't understand the concept",
        "please explain the theory again",
        "the idea is unclear",
    ],
    "Confusion": [
        "I am confused about this",
        "what does this mean",
        "can you simplify",
    ],
    "Frustration": [
        "this is not working",
        "I tried many times and still failed",
        "I'm stuck and annoyed",
    ],
    "Exam anxiety": [
        "exam is near and I am nervous",
        "I am scared about marks",
        "I may forget in the exam",
    ],
    "Assignment/Submission stress": [
        "submission deadline is close",
        "need to submit assignment today",
        "time is running out",
    ],
    "Debugging / implementation issue": [
        "code error bug exception output mismatch",
        "implementation is failing",
        "debug this program",
    ],
    "Need for explanation": [
        "explain step by step",
        "give intuition and example",
        "why does this happen",
    ],
    "Resource request": [
        "share notes book video reference pdf",
        "can you provide materials",
        "any sample or tutorial",
    ],
    "Deadline pressure": [
        "urgent before deadline",
        "need help quickly",
        "little time left",
    ],
    "Clarification request": [
        "please clarify the meaning",
        "is it correct",
        "what exactly should I do",
    ],
}

NEGATIVE_CUES = {
    "frustration": ["stuck", "annoyed", "frustrated", "fed up", "failed", "not working", "error", "bug", "broken", "confusing"],
    "anxiety": ["anxious", "worried", "scared", "panic", "nervous", "exam", "deadline", "marks", "fail", "stress"],
    "confusion": ["confused", "don't understand", "cannot understand", "unclear", "what is", "how does", "meaning"],
    "curiosity": ["why", "how", "what if", "can you explain", "intuition", "example"],
}

THEME = {
    "bg": "radial-gradient(circle at top, #0f172a 0%, #111827 40%, #0b1220 100%)",
    "card": "rgba(15,23,42,0.86)",
    "card_border": "rgba(148,163,184,0.12)",
    "muted": "#94a3b8",
}


def normalize_text(text: str) -> str:
    text = str(text or "").strip()
    return re.sub(r"\s+", " ", text)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _proto_texts(label_map: Dict[str, List[str]]) -> Tuple[List[str], List[str]]:
    labels = list(label_map.keys())
    docs = [" ".join(label_map[label]) for label in labels]
    return labels, docs


class FastSemanticEngine:
    def __init__(self):
        self.topic_labels, self.topic_docs = _proto_texts(TOPIC_PROTOTYPES)
        self.intent_labels, self.intent_docs = _proto_texts(INTENT_PROTOTYPES)

        corpus = self.topic_docs + self.intent_docs
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.vectorizer.fit(corpus)

        self.topic_matrix = self.vectorizer.transform(self.topic_docs)
        self.intent_matrix = self.vectorizer.transform(self.intent_docs)

    def score(self, text: str, labels: List[str], matrix) -> Tuple[str, float]:
        vec = self.vectorizer.transform([text])
        sims = linear_kernel(vec, matrix).ravel()
        idx = int(np.argmax(sims))
        return labels[idx], float(sims[idx])


@st.cache_resource(show_spinner=False)
def get_engine():
    return FastSemanticEngine()


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    if SentenceTransformer is None:
        return None
    for model_name in ["sentence-transformers/all-MiniLM-L6-v2", "all-MiniLM-L6-v2"]:
        try:
            return SentenceTransformer(model_name)
        except Exception:
            continue
    return None


@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    if pipeline is None:
        return None
    for model_name in [
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "distilbert-base-uncased-finetuned-sst-2-english",
    ]:
        try:
            return pipeline("sentiment-analysis", model=model_name, truncation=True)
        except Exception:
            continue
    return None


@st.cache_resource(show_spinner=False)
def load_emotion_model():
    if pipeline is None:
        return None
    for model_name in [
        "j-hartmann/emotion-english-distilroberta-base",
        "bhadresh-savani/distilbert-base-uncased-emotion",
    ]:
        try:
            return pipeline("text-classification", model=model_name, truncation=True, top_k=None)
        except Exception:
            continue
    return None


def model_emotion(text: str, emotion_pipe) -> Tuple[str, float]:
    if emotion_pipe is None:
        return rule_based_emotion(text)

    try:
        result = emotion_pipe(text)
        flat = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
        if not flat:
            return rule_based_emotion(text)

        best = max(flat, key=lambda x: x.get("score", 0.0))
        label = str(best.get("label", "neutral")).lower()
        score = float(best.get("score", 0.0))
        mapping = {
            "joy": "confidence",
            "anger": "frustration",
            "fear": "anxiety",
            "sadness": "sadness",
            "neutral": "neutral",
            "surprise": "curiosity",
            "disgust": "frustration",
        }
        label = mapping.get(label, label)
        return (label if label in EMOTION_LABELS else "neutral"), score
    except Exception:
        return rule_based_emotion(text)


def sentiment_label(text: str, sentiment_pipe) -> Tuple[str, float]:
    t = text.strip()
    if not t:
        return "neutral", 0.0

    if sentiment_pipe is None:
        positive = len(re.findall(r"\b(thanks|great|good|understand|clear|solved)\b", t.lower()))
        negative = len(re.findall(r"\b(confused|stuck|error|wrong|bad|sad|angry|frustrated|scared)\b", t.lower()))
        if positive > negative:
            return "positive", round(min(0.95, 0.55 + 0.1 * positive), 3)
        if negative > positive:
            return "negative", round(min(0.95, 0.55 + 0.1 * negative), 3)
        return "neutral", 0.5

    try:
        result = sentiment_pipe(t)
        item = result[0] if isinstance(result, list) else result
        label = str(item.get("label", "neutral")).lower()
        score = float(item.get("score", 0.0))
        if "pos" in label:
            label = "positive"
        elif "neg" in label:
            label = "negative"
        else:
            label = "neutral"
        return label, score
    except Exception:
        return "neutral", 0.5


def rule_based_emotion(text: str) -> Tuple[str, float]:
    t = text.lower()
    scores = {k: 0.0 for k in EMOTION_LABELS}

    for label, cues in NEGATIVE_CUES.items():
        for cue in cues:
            if cue in t:
                scores[label] += 1.0

    exclamations = t.count("!")
    questions = t.count("?")
    if exclamations > 0:
        scores["frustration"] += 0.5
        scores["stress"] += 0.25
    if questions >= 2:
        scores["confusion"] += 0.5
        scores["curiosity"] += 0.25

    if re.search(r"\b(exam|test|quiz|midterm|final)\b", t):
        scores["anxiety"] += 1.0
        scores["stress"] += 0.5
    if re.search(r"\b(deadline|submission|submit|urgent)\b", t):
        scores["stress"] += 1.0
        scores["anxiety"] += 0.5

    if all(v == 0 for v in scores.values()):
        return "neutral", 0.35

    label = max(scores, key=scores.get)
    total = sum(scores.values()) or 1.0
    return label, float(round(scores[label] / total, 3))


def intensity_score(text: str) -> float:
    t = text.lower()
    ex = t.count("!")
    qs = t.count("?")
    caps = len(re.findall(r"\b[A-Z]{3,}\b", text))
    urgent = len(re.findall(r"\b(urgent|immediately|asap|today|now|deadline)\b", t))
    neg = len(re.findall(r"\b(not working|can't|cannot|failed|stuck|error|wrong)\b", t))
    score = 0.15 * ex + 0.1 * qs + 0.15 * caps + 0.25 * urgent + 0.2 * neg
    return float(min(1.0, round(score, 3)))


def aspect_analysis(text: str) -> Dict[str, List[str]]:
    aspects = {
        "concept": [],
        "implementation": [],
        "exam": [],
        "deadline": [],
        "resources": [],
        "math": [],
    }
    t = text.lower()

    patterns = {
        "concept": ["concept", "meaning", "understand", "explain", "intuition", "theory"],
        "implementation": ["code", "bug", "error", "debug", "implementation", "run", "compile"],
        "exam": ["exam", "test", "quiz", "midterm", "final", "marks", "score"],
        "deadline": ["deadline", "submission", "submit", "urgent", "late", "today"],
        "resources": ["notes", "video", "pdf", "book", "reference", "material"],
        "math": ["equation", "formula", "theorem", "proof", "calculate", "derivative", "integral"],
    }

    for aspect, cues in patterns.items():
        for cue in cues:
            if cue in t:
                aspects[aspect].append(cue)

    return aspects


def _keyword_boost(text: str, label: str, keyword_bank: Dict[str, List[str]]) -> float:
    t = text.lower()
    hits = 0
    for kw in keyword_bank.get(label, []):
        if kw in t:
            hits += 1
    return min(1.0, hits / 4.0)


TOPIC_KEYWORDS = {
    "Mathematics": ["equation", "formula", "derivative", "integral", "theorem", "proof", "matrix", "geometry", "probability"],
    "Programming": ["code", "python", "java", "c++", "bug", "error", "compile", "runtime", "function", "class"],
    "Data Structures": ["array", "linked list", "stack", "queue", "tree", "graph", "heap", "recursion", "complexity"],
    "Machine Learning": ["model", "training", "accuracy", "overfitting", "classification", "regression", "loss", "neural network"],
    "NLP": ["tokenization", "embedding", "transformer", "bert", "roberta", "intent", "sentiment", "language model"],
    "Database / SQL": ["sql", "select", "join", "group by", "database", "primary key", "foreign key", "transaction"],
    "Computer Networks": ["tcp", "udp", "ip", "routing", "dns", "http", "packet", "latency", "protocol"],
    "Operating Systems": ["process", "thread", "deadlock", "semaphore", "mutex", "paging", "memory", "kernel"],
    "Electronics": ["circuit", "voltage", "current", "resistance", "transistor", "diode", "op amp", "logic gate"],
    "Other": [],
}

INTENT_KEYWORDS = {
    "Concept misunderstanding": ["don't understand", "understand", "concept", "theory", "meaning", "explain"],
    "Confusion": ["confused", "unclear", "simplify", "what does this mean", "how does"],
    "Frustration": ["stuck", "annoyed", "frustrated", "not working", "failed", "bug", "error"],
    "Exam anxiety": ["exam", "test", "quiz", "scared", "marks", "nervous", "forget"],
    "Assignment/Submission stress": ["assignment", "submission", "submit", "deadline", "today", "urgent"],
    "Debugging / implementation issue": ["bug", "error", "debug", "compile", "runtime", "implementation", "code"],
    "Need for explanation": ["explain", "step by step", "intuition", "example", "why does this happen"],
    "Resource request": ["notes", "video", "pdf", "book", "reference", "materials", "tutorial"],
    "Deadline pressure": ["deadline", "urgent", "asap", "little time", "quickly", "now"],
    "Clarification request": ["clarify", "correct", "what exactly", "should I do", "is it right"],
}


def infer_topic(text: str, engine: FastSemanticEngine) -> Tuple[str, float]:
    label, score = engine.score(text, engine.topic_labels, engine.topic_matrix)
    boost = _keyword_boost(text, label, TOPIC_KEYWORDS)
    final = 0.82 * score + 0.18 * boost
    return label, float(round(final, 3))


def infer_intent(text: str, engine: FastSemanticEngine) -> Tuple[str, float]:
    label, score = engine.score(text, engine.intent_labels, engine.intent_matrix)
    boost = _keyword_boost(text, label, INTENT_KEYWORDS)
    final = 0.82 * score + 0.18 * boost
    return label, float(round(final, 3))


def generate_summary(topic, topic_score, intent, intent_score, emotion, emotion_score, sentiment, sentiment_score, intensity, aspects):
    aspect_bits = []
    for k, v in aspects.items():
        if v:
            aspect_bits.append(f"{k}: {', '.join(sorted(set(v))[:3])}")
    aspect_text = "; ".join(aspect_bits) if aspect_bits else "No strong aspect keywords detected."

    risk = "low"
    if emotion in {"anxiety", "stress"} or intensity > 0.65:
        risk = "high"
    elif emotion in {"frustration", "sadness"} or intensity > 0.35:
        risk = "medium"

    return {
        "topic_summary": f"Topic: {topic} ({topic_score:.2f})",
        "intent_summary": f"Hidden intent: {intent} ({intent_score:.2f})",
        "emotion_summary": f"Emotion: {emotion} ({emotion_score:.2f})",
        "sentiment_summary": f"Sentiment: {sentiment} ({sentiment_score:.2f})",
        "risk_summary": f"Learning-risk signal: {risk}",
        "aspect_summary": aspect_text,
    }


def analyze_text(text: str, engine: FastSemanticEngine, emotion_pipe, sentiment_pipe):
    text = normalize_text(text)
    topic, topic_score = infer_topic(text, engine)
    intent, intent_score = infer_intent(text, engine)
    emotion, emotion_score = model_emotion(text, emotion_pipe)
    sentiment, sentiment_score = sentiment_label(text, sentiment_pipe)
    intensity = intensity_score(text)
    aspects = aspect_analysis(text)
    summary = generate_summary(topic, topic_score, intent, intent_score, emotion, emotion_score, sentiment, sentiment_score, intensity, aspects)
    confidence = round(float(np.mean([topic_score, intent_score, emotion_score, sentiment_score])), 3)
    return {
        "text": text,
        "topic": topic,
        "topic_score": round(topic_score, 3),
        "intent": intent,
        "intent_score": round(intent_score, 3),
        "emotion": emotion,
        "emotion_score": round(emotion_score, 3),
        "sentiment": sentiment,
        "sentiment_score": round(sentiment_score, 3),
        "intensity": intensity,
        "confidence": confidence,
        "aspects": aspects,
        "summary": summary,
    }


def app_header():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-badge">NLP • Streamlit • Real-time doubt mining</div>
            <h1>Deep Intent Mining from Student Doubt Messages</h1>
            <p>Detect the topic, hidden intent, emotion, sentiment, and learning-risk signal behind student messages from chats, emails, or LMS notes.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_controls():
    st.sidebar.markdown("## Controls")
    use_models = st.sidebar.toggle(
        "Use Hugging Face models",
        value=False,
        help="Turn on only if you want heavier semantic models for emotion and sentiment. Keeping this off makes the app much faster.",
    )
    show_debug = st.sidebar.toggle("Show analysis details", value=True)
    top_k = st.sidebar.slider("Top-k labels shown in explanation", min_value=3, max_value=8, value=5)
    return use_models, show_debug, top_k


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at top, #0f172a 0%, #111827 40%, #0b1220 100%);
            color: #e5e7eb;
        }
        .hero {
            padding: 1.3rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, rgba(15,23,42,0.95), rgba(30,41,59,0.82));
            border: 1px solid rgba(148,163,184,0.15);
            box-shadow: 0 20px 60px rgba(0,0,0,0.28);
            margin-bottom: 1rem;
        }
        .hero h1 {
            margin: 0.35rem 0 0.5rem 0;
            font-size: 2.2rem;
            line-height: 1.1;
            color: #f8fafc;
        }
        .hero p {
            margin: 0;
            color: #cbd5e1;
            font-size: 1.02rem;
        }
        .hero-badge {
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 999px;
            background: rgba(59,130,246,0.16);
            color: #bfdbfe;
            font-size: 0.82rem;
            border: 1px solid rgba(96,165,250,0.26);
        }
        .card {
            background: rgba(15,23,42,0.86);
            border: 1px solid rgba(148,163,184,0.12);
            padding: 1rem 1rem 0.9rem 1rem;
            border-radius: 20px;
            box-shadow: 0 14px 42px rgba(0,0,0,0.20);
        }
        .small-note {
            color: #94a3b8;
            font-size: 0.88rem;
        }
        .pill {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            margin: 0.18rem 0.25rem 0.18rem 0;
            border-radius: 999px;
            background: rgba(51,65,85,0.8);
            color: #e2e8f0;
            font-size: 0.82rem;
            border: 1px solid rgba(148,163,184,0.12);
        }
        .metric-wrap {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }
        .metric-box {
            flex: 1 1 180px;
            background: rgba(30,41,59,0.8);
            border: 1px solid rgba(148,163,184,0.12);
            border-radius: 18px;
            padding: 0.85rem 0.9rem;
        }
        .metric-title { color: #94a3b8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.07em; }
        .metric-value { color: #f8fafc; font-size: 1.18rem; font-weight: 700; margin-top: 0.15rem; }
        .metric-sub { color: #cbd5e1; font-size: 0.84rem; margin-top: 0.1rem; }
        .stTextArea textarea {
            background: rgba(15,23,42,0.85) !important;
            color: #f8fafc !important;
            border-radius: 18px !important;
        }
        .stButton button {
            border-radius: 14px !important;
            font-weight: 700 !important;
            padding: 0.55rem 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def pretty_pills(items: List[str], color_class: str = "pill"):
    html = ""
    for item in items:
        html += f'<span class="pill {color_class}">{item}</span>'
    return html


def flatten_row(result: Dict) -> Dict:
    return {
        "text": result["text"],
        "topic": result["topic"],
        "topic_score": result["topic_score"],
        "intent": result["intent"],
        "intent_score": result["intent_score"],
        "emotion": result["emotion"],
        "emotion_score": result["emotion_score"],
        "sentiment": result["sentiment"],
        "sentiment_score": result["sentiment_score"],
        "intensity": result["intensity"],
        "confidence": result["confidence"],
        "risk_summary": result["summary"]["risk_summary"],
        "aspect_summary": result["summary"]["aspect_summary"],
    }


def render_result(result: Dict, show_debug: bool, top_k: int):
    st.markdown(
        f"""
        <div class="card">
            <h3 style="margin-top:0;color:#f8fafc;">Prediction Overview</h3>
            <div class="metric-wrap">
                <div class="metric-box">
                    <div class="metric-title">Topic</div>
                    <div class="metric-value">{result['topic']}</div>
                    <div class="metric-sub">score: {result['topic_score']:.3f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Hidden intent</div>
                    <div class="metric-value">{result['intent']}</div>
                    <div class="metric-sub">score: {result['intent_score']:.3f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Emotion</div>
                    <div class="metric-value">{result['emotion']}</div>
                    <div class="metric-sub">score: {result['emotion_score']:.3f}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-title">Risk signal</div>
                    <div class="metric-value">{result['summary']['risk_summary'].split(': ')[1]}</div>
                    <div class="metric-sub">intensity: {result['intensity']:.3f}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1.2, 0.8])
    with c1:
        st.markdown("#### Interpretable summary")
        st.info(result["summary"]["topic_summary"])
        st.info(result["summary"]["intent_summary"])
        st.info(result["summary"]["emotion_summary"])
        st.info(result["summary"]["sentiment_summary"])
        st.warning(result["summary"]["risk_summary"])
        st.caption(result["summary"]["aspect_summary"])

    with c2:
        st.markdown("#### Detected signals")
        st.markdown(
            pretty_pills([f"Confidence {result['confidence']:.2f}", f"Intensity {result['intensity']:.2f}"]),
            unsafe_allow_html=True,
        )
        st.markdown("**Sentence view**")
        sents = split_sentences(result["text"])
        if sents:
            for s in sents[:6]:
                st.write("•", s)
        else:
            st.write("No sentence segments detected.")

    if show_debug:
        st.markdown("#### Analysis details")
        detail_cols = st.columns(3)
        with detail_cols[0]:
            st.write("Aspects")
            st.json(result["aspects"])
        with detail_cols[1]:
            st.write("Model strategy")
            st.write(
                "Topic and intent use cached TF-IDF similarity for speed. Emotion and sentiment use Hugging Face models only when enabled; otherwise fast heuristics are used."
            )
        with detail_cols[2]:
            st.write("Why this matters")
            st.write("High anxiety or frustration can be routed to teacher support, while confusion can trigger concept re-teaching.")


def _counts_df(series: pd.Series, col_name: str) -> pd.DataFrame:
    counts = series.value_counts(dropna=False).reset_index()
    counts.columns = [col_name, "count"]
    return counts


def render_batch_charts(df: pd.DataFrame):
    st.markdown("### Batch analytics")

    topic_counts = _counts_df(df["topic"], "topic")
    intent_counts = _counts_df(df["intent"], "intent")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(topic_counts, x="topic", y="count", title="Topic distribution")
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig2 = px.bar(intent_counts, x="intent", y="count", title="Intent distribution")
        fig2.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=360)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.histogram(df, x="confidence", nbins=10, title="Confidence distribution")
        fig3.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
        st.plotly_chart(fig3, use_container_width=True)
    with c4:
        fig4 = px.histogram(df, x="intensity", nbins=10, title="Intensity distribution")
        fig4.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=320)
        st.plotly_chart(fig4, use_container_width=True)


def main():
    inject_css()
    app_header()
    use_models, show_debug, top_k = sidebar_controls()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Input modes")
    mode = st.sidebar.radio("Choose input type", ["Single message", "Paste multiple messages", "Upload CSV"], index=0)

    engine = get_engine()
    emotion_pipe = load_emotion_model() if use_models else None
    sentiment_pipe = load_sentiment_model() if use_models else None

    st.markdown(
        """
        <div class="card">
            <div class="small-note">
            Pipeline: sentence embeddings → topic & intent similarity → emotion detection → sentiment analysis → aspect cues → learning-risk summary
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    if mode == "Single message":
        with st.form("single_form", clear_on_submit=False):
            text = st.text_area(
                "Enter a student doubt message",
                placeholder="Example: I am confused about backpropagation. The formulas look easy, but I keep getting wrong answers in the exam and I am scared I will fail.",
                height=190,
            )
            col1, col2 = st.columns([1, 2])
            with col1:
                analyze_btn = st.form_submit_button("Analyze message", type="primary")
            with col2:
                st.caption("Works for chat messages, emails, LMS notes, and mixed student queries.")
        if analyze_btn and text.strip():
            result = analyze_text(text, engine, emotion_pipe, sentiment_pipe)
            render_result(result, show_debug, top_k)
        elif analyze_btn:
            st.warning("Please enter a message before analyzing.")

    elif mode == "Paste multiple messages":
        with st.form("batch_form", clear_on_submit=False):
            text = st.text_area(
                "Paste one message per line",
                placeholder="Message 1...\nMessage 2...\nMessage 3...",
                height=220,
            )
            analyze_batch = st.form_submit_button("Analyze batch", type="primary")
        if analyze_batch:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines:
                st.warning("Please paste at least one message.")
            else:
                rows = [analyze_text(line, engine, emotion_pipe, sentiment_pipe) for line in lines]
                df = pd.DataFrame([flatten_row(r) for r in rows])
                st.success(f"Analyzed {len(df)} messages.")
                st.dataframe(df, use_container_width=True)
                render_batch_charts(df)
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results as CSV", csv, "student_doubt_results.csv", "text/csv", use_container_width=True)

    else:
        uploaded = st.file_uploader("Upload CSV with a 'text' column", type=["csv"])
        if uploaded is not None:
            try:
                df_in = pd.read_csv(uploaded)
                if "text" not in df_in.columns:
                    st.error("The CSV must contain a column named 'text'.")
                else:
                    rows = [analyze_text(str(x), engine, emotion_pipe, sentiment_pipe) for x in df_in["text"].fillna("")]
                    df = pd.DataFrame([flatten_row(r) for r in rows])
                    st.success(f"Processed {len(df)} rows from the uploaded file.")
                    st.dataframe(df, use_container_width=True)
                    render_batch_charts(df)
                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button("Download enriched CSV", csv, "annotated_student_doubts.csv", "text/csv", use_container_width=True)
            except Exception as e:
                st.error(f"Could not read the file: {e}")

    with st.expander("What this project does"):
        st.markdown(
            """
            This project detects:
            - **Topic**: the academic subject area
            - **Hidden intent**: confusion, frustration, exam anxiety, concept misunderstanding, and more
            - **Emotion**: a deeper emotional state behind the text
            - **Sentiment**: overall tone
            - **Aspect cues**: signals like exam pressure, coding bugs, resources, or deadlines
            - **Learning-risk signal**: a practical indicator for teacher intervention
            """
        )


if __name__ == "__main__":
    main()
