"""AI Resume Bullet Optimizer - Streamlit app.

Run with:  streamlit run app.py
Pages (sidebar): Upload -> Optimizer -> Job Match -> Skills Gap -> Score -> Dashboard -> Download
"""
from __future__ import annotations

import html
import inspect
import os
from statistics import mean
from typing import Dict, List

import streamlit as st

try:  # optional: load OPENAI_API_KEY from a local .env file
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # pragma: no cover
    pass

from modules import charts, export, llm, nlp_matcher, samples, scorer
from modules import skills as skills_mod
from modules.bullet_rules import analyze_bullet, offline_optimize
from modules.resume_parser import (Bullet, ParsedResume, ResumeParseError, extract_text_from_pdf,
                                   parse_resume)

st.set_page_config(page_title="AI Resume Bullet Optimizer", page_icon="📄", layout="wide")

MAX_UPLOAD_MB = 10
MAX_BULLETS_PER_RUN = 30
PAGES = ["1 · Upload Resume", "2 · Bullet Optimizer", "3 · Job Match", "4 · Skills Gap",
         "5 · Score & Suggestions", "6 · Analytics Dashboard", "7 · Download"]

# Streamlit renamed use_container_width -> width="stretch"; support both.
_STRETCH = ({"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters
            else {"use_container_width": True})

CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.hero {padding: 1.2rem 1.5rem; border-radius: 16px; margin-bottom: 1.2rem;
       background: linear-gradient(135deg, rgba(79,70,229,.16), rgba(16,185,129,.12));
       border: 1px solid rgba(128,128,128,.2);}
.hero h2 {margin: 0 0 .25rem 0;} .hero p {margin: 0; opacity: .8;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: .9rem 1.1rem;
       margin-bottom: .8rem; background: rgba(128,128,128,.06);}
.card h4 {margin: 0 0 .4rem 0; font-size: 1rem;}
.chip {display: inline-block; padding: .15rem .65rem; margin: .15rem .2rem .15rem 0; border-radius: 999px;
       font-size: .82rem; border: 1px solid rgba(128,128,128,.3);}
.chip-good {background: rgba(16,185,129,.15); border-color: rgba(16,185,129,.5);}
.chip-bad {background: rgba(239,68,68,.13); border-color: rgba(239,68,68,.5);}
.chip-info {background: rgba(79,70,229,.13); border-color: rgba(79,70,229,.5);}
.badge {padding: .1rem .55rem; border-radius: 8px; font-size: .78rem; font-weight: 600;}
.b-High {background: rgba(239,68,68,.18);} .b-Medium {background: rgba(245,158,11,.2);}
.b-Low {background: rgba(16,185,129,.18);}
.bigscore {font-size: 3rem; font-weight: 700; line-height: 1;}
@media (max-width: 768px) {.block-container {padding: 1rem .6rem;} .bigscore {font-size: 2.2rem;}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# State & helpers
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "parsed": None, "manual_bullets": [], "match": None, "jd_info": None, "gap": None, "recs": None,
    "opt": {}, "ai_suggestions": None, "last_sig": None, "page": PAGES[0], "w_jd": "",
    "w_method": "TF-IDF (fast, no downloads)", "w_model": llm.DEFAULT_MODEL, "w_manual": "", "w_tailor": False,
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

# Keep widget values alive when their page isn't rendered (Streamlit drops unrendered widget state).
for _k in [k for k in st.session_state if str(k).startswith("w_")]:
    st.session_state[_k] = st.session_state[_k]


def esc(s) -> str:
    return html.escape(str(s))


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h2>{esc(title)}</h2><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def chips(items, kind: str = "info") -> str:
    return "".join(f'<span class="chip chip-{kind}">{esc(i)}</span>' for i in items) or "<em>none</em>"


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="card"><h4>{esc(title)}</h4>{body_html}</div>', unsafe_allow_html=True)


def show_chart(fig) -> None:
    st.plotly_chart(fig, **_STRETCH)


def go_to(page: str) -> None:
    st.session_state.page = page


def get_api_key() -> str:
    key = st.session_state.get("w_api_key", "").strip()
    if key:
        return key
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:  # noqa: BLE001 - no secrets file configured
            key = ""
    return key


def all_bullets() -> List[Bullet]:
    p: ParsedResume = st.session_state.parsed
    if p is None:
        return []
    manual = [Bullet(1000 + i, "manual", "Added manually", t) for i, t in enumerate(st.session_state.manual_bullets)]
    return p.bullets + manual


def manual_bullet_objects() -> List[Bullet]:
    return [b for b in all_bullets() if b.section == "manual"]


def reset_analysis() -> None:
    for k in ("match", "jd_info", "gap", "recs", "ai_suggestions"):
        st.session_state[k] = None
    st.session_state.opt = {}
    for k in [k for k in st.session_state if str(k).startswith(("w_edit_", "w_acc_", "w_sel", "w_manual", "w_raw"))]:
        del st.session_state[k]


def load_resume(text: str, pages: int = 1) -> None:
    st.session_state.parsed = parse_resume(text, pages)
    st.session_state.manual_bullets = []
    reset_analysis()


def load_sample_resume() -> None:
    load_resume(samples.RESUME, 1)
    f = st.session_state.get("upload_pdf")   # don't let a previously uploaded file overwrite the sample
    st.session_state.last_sig = (f.name, f.size) if f else "sample"


def load_sample_jd() -> None:
    st.session_state.w_jd = samples.JOB_DESCRIPTION


def require_resume() -> bool:
    if st.session_state.parsed is None:
        st.info("Upload a resume first (or try the sample resume).")
        st.button("Go to Upload", on_click=go_to, args=(PAGES[0],), key="goto_upload")
        return False
    return True


def get_replacements() -> Dict[int, str]:
    reps = {}
    for bid, res in st.session_state.opt.items():
        if st.session_state.get(f"w_acc_{bid}", False):
            txt = st.session_state.get(f"w_edit_{bid}", res["optimized"]).strip()
            if txt:
                reps[bid] = txt
    return reps


def compute_score() -> scorer.ResumeScore:
    return scorer.score_resume(st.session_state.parsed, all_bullets(), st.session_state.match)


def compute_suggestions(score: scorer.ResumeScore) -> List[dict]:
    return scorer.rule_based_suggestions(st.session_state.parsed, score, st.session_state.gap,
                                         st.session_state.match, st.session_state.recs)


def score_color(v: float) -> str:
    return "#10B981" if v >= 75 else "#F59E0B" if v >= 50 else "#EF4444"


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> None:
    with st.sidebar:
        st.markdown("## 📄 Resume Optimizer")
        st.caption("NLP + OpenAI, built to stay honest: no invented achievements.")
        st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")

        s = st.session_state
        st.divider()
        st.markdown("**Progress**")
        steps = [("Resume loaded", s.parsed is not None), ("Bullets optimized", bool(s.opt)),
                 ("Job description matched", s.match is not None)]
        for label, done in steps:
            st.markdown(f"{'✅' if done else '⬜'} {label}")

        st.divider()
        with st.expander("⚙️ OpenAI settings", expanded=not get_api_key()):
            st.text_input("API key", type="password", key="w_api_key", placeholder="sk-…  (or set OPENAI_API_KEY)")
            st.text_input("Model", key="w_model",
                          help="Any chat model your key can use, e.g. gpt-4o-mini.")
            if get_api_key():
                st.success("AI mode available")
            else:
                st.info("No key found: the app still works with local rule-based rewriting.")
        st.caption("Your resume is processed in memory and only sent to OpenAI when you click an AI button.")


# --------------------------------------------------------------------------- #
# Page 1 - Upload
# --------------------------------------------------------------------------- #
def page_upload() -> None:
    hero("Upload your resume", "We extract your skills, education, experience and projects automatically.")
    left, right = st.columns([3, 2])
    with left:
        up = st.file_uploader("Resume (PDF, text-based, up to 10 MB)", type=["pdf"], key="upload_pdf")
        if up is not None:
            sig = (up.name, up.size)
            if st.session_state.last_sig != sig:
                try:
                    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
                        raise ResumeParseError(f"File is larger than {MAX_UPLOAD_MB} MB.")
                    with st.spinner("Reading your resume…"):
                        text, pages = extract_text_from_pdf(up.getvalue())
                        load_resume(text, pages)
                    st.session_state.last_sig = sig
                except ResumeParseError as exc:
                    st.error(str(exc))
    with right:
        card("No PDF handy?", "<p>Try the app instantly with a fictional sample resume and job description.</p>")
        st.button("Use sample resume", on_click=load_sample_resume, type="primary")

    p: ParsedResume = st.session_state.parsed
    if p is None:
        return

    st.success(f"Resume loaded: {p.word_count} words, {p.num_pages} page(s), {len(p.bullets)} bullet points found.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skills detected", len(p.skills))
    c2.metric("Bullets", len(p.bullets))
    c3.metric("Experience entries", len(p.experience))
    c4.metric("Projects", len(p.projects))
    if not p.bullets:
        st.warning("No bullet points were detected. Add some below, or fix the extracted text.")

    a, b = st.columns(2)
    with a:
        contact = [x for x in [p.name, p.email, p.phone, *p.links] if x]
        card("Contact", "<br>".join(esc(x) for x in contact) or "<em>not found</em>")
        card("Education", "<br>".join(esc(x) for x in p.education) or "<em>No Education section detected</em>")
    with b:
        body = "".join(f"<p style='margin:.2rem 0'><b>{esc(cat)}</b><br>{chips(sk)}</p>"
                       for cat, sk in p.skill_categories.items()) or "<em>No known skills detected</em>"
        if p.extra_skills:
            body += f"<p style='margin:.2rem 0'><b>Other listed</b><br>{chips(p.extra_skills, 'info')}</p>"
        card("Skills", body)

    for title, entries in (("Experience", p.experience), ("Projects", p.projects)):
        if entries:
            st.markdown(f"#### {title}")
            for e in entries:
                with st.expander(e.header or "(untitled)"):
                    for bl in e.bullets:
                        st.markdown(f"- {bl.text}")

    st.markdown("#### Fix or add bullets")
    st.caption("The parser is heuristic. Add any missed achievements here (one per line) to include them in optimization.")
    st.session_state.setdefault("w_manual", "")
    manual = st.text_area("Extra bullets (one per line)", height=110, key="w_manual")
    st.session_state.manual_bullets = [ln.strip("•-* \t") for ln in manual.splitlines() if ln.strip("•-* \t")]

    with st.expander("Edit extracted text and re-parse"):
        st.session_state.setdefault("w_raw", p.raw_text)
        edited = st.text_area("Raw extracted text", height=300, key="w_raw")
        if st.button("Re-parse edited text"):
            load_resume(edited, p.num_pages)
            st.rerun()
    st.button("Next: optimize bullets →", on_click=go_to, args=(PAGES[1],), type="primary")


# --------------------------------------------------------------------------- #
# Page 2 - Bullet optimizer
# --------------------------------------------------------------------------- #
def run_optimization(selected: List[int], use_ai: bool, style: str, jd: str) -> None:
    bullets = {b.id: b for b in all_bullets()}
    pairs = [(i, bullets[i].text, bullets[i].entry) for i in selected]
    bar = st.progress(0.0, text="Starting…")
    if use_ai:
        results = llm.optimize_bullets(
            pairs, api_key=get_api_key(), model=st.session_state.get("w_model") or llm.DEFAULT_MODEL,
            style=style, job_description=jd,
            progress=lambda d, t: bar.progress(d / t, text=f"Rewrote {d} of {t} bullets"))
    else:
        results = offline_optimize([(i, t) for i, t, _ in pairs])
    bar.empty()
    st.session_state.opt = {r["id"]: r for r in results}
    for r in results:  # (re)seed the editable widgets
        st.session_state[f"w_edit_{r['id']}"] = r["optimized"]
        st.session_state[f"w_acc_{r['id']}"] = not r["warnings"] and r["optimized"].strip() != r["original"].strip()


def page_optimizer() -> None:
    hero("Bullet point optimizer", "Stronger verbs, tighter wording, measurable impact - without inventing anything.")
    if not require_resume():
        return
    bullets = all_bullets()
    if not bullets:
        st.warning("No bullets to optimize. Add some on the Upload page.")
        return
    by_id = {b.id: b for b in bullets}

    with st.container(border=True):
        has_key = bool(get_api_key())
        c1, c2 = st.columns([2, 1])
        with c1:
            engines = ["AI rewrite (OpenAI)", "Local rules (offline, conservative)"]
            st.session_state.setdefault("w_engine", engines[0 if has_key else 1])
            mode = st.radio("Engine", engines, horizontal=True, key="w_engine")
        with c2:
            style = st.selectbox("Style", list(llm.STYLE_GUIDES), key="w_style")
        use_ai = mode.startswith("AI")
        if use_ai and not has_key:
            st.warning("Add an OpenAI API key in the sidebar to use AI mode.")
        jd = st.session_state.w_jd.strip()
        tailor = st.checkbox("Tailor wording to my job description (only where truthful)",
                             disabled=not jd, key="w_tailor", help="Paste a job description on the Job Match page.")
        ids = [b.id for b in bullets]
        valid_default = [i for i in st.session_state.get("w_sel", ids[:MAX_BULLETS_PER_RUN]) if i in by_id]
        st.session_state["w_sel"] = valid_default
        selected = st.multiselect("Bullets to optimize", ids, key="w_sel",
                                  format_func=lambda i: f"[{by_id[i].section}] {by_id[i].text[:85]}")
        if len(selected) > MAX_BULLETS_PER_RUN:
            st.warning(f"Select at most {MAX_BULLETS_PER_RUN} bullets per run.")
        go = st.button("✨ Optimize selected bullets", type="primary",
                       disabled=not selected or len(selected) > MAX_BULLETS_PER_RUN or (use_ai and not has_key))

    if go:
        try:
            run_optimization(selected, use_ai, style, jd if (tailor and jd) else "")
        except llm.LLMError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - never crash the UI
            st.error(f"Something went wrong while optimizing: {exc}")

    opt = st.session_state.opt
    if not opt:
        return
    before = [analyze_bullet(r["original"]).score for r in opt.values()]
    after = [analyze_bullet(st.session_state.get(f"w_edit_{i}", r["optimized"])).score for i, r in opt.items()]
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg quality before", f"{mean(before):.0f}/100")
    m2.metric("Avg quality after", f"{mean(after):.0f}/100", f"{mean(after) - mean(before):+.0f}")
    m3.metric("Accepted rewrites", sum(bool(st.session_state.get(f"w_acc_{i}")) for i in opt))
    st.caption("Review each suggestion. Edit the text freely, and tick **Use** to include it in your download. "
               "Rewrites that introduce new numbers or skills are flagged and unticked by default.")

    for bid, r in opt.items():
        with st.container(border=True):
            left, right = st.columns(2)
            with left:
                st.markdown(f"**Original** &nbsp; <span class='badge b-Low'>{analyze_bullet(r['original']).score:.0f}/100</span>",
                            unsafe_allow_html=True)
                st.write(r["original"])
                st.caption(f"{by_id[bid].entry[:70] if bid in by_id else ''}")
            with right:
                st.text_area("Optimized (editable)", key=f"w_edit_{bid}", height=90)
                st.checkbox("Use this version", key=f"w_acc_{bid}")
            if r["changes"]:
                st.caption("Changes: " + " · ".join(r["changes"]))
            for w in r["warnings"]:
                st.warning(w, icon="⚠️")
            if r.get("metric_prompt"):
                st.info(f"💡 {r['metric_prompt']} If you have a real number, add it to the text above.")
    st.button("Next: match a job description →", on_click=go_to, args=(PAGES[2],), type="primary")


# --------------------------------------------------------------------------- #
# Page 3 - Job match
# --------------------------------------------------------------------------- #
def page_match() -> None:
    hero("Job description matching", "Paste a job posting to see how well your resume fits - and why.")
    if not require_resume():
        return
    p: ParsedResume = st.session_state.parsed
    st.text_area("Job description", key="w_jd", height=240, placeholder="Paste the full job posting here…")
    c1, c2 = st.columns([3, 2])
    with c1:
        emb_ok = nlp_matcher.embeddings_available()
        options = ["TF-IDF (fast, no downloads)", "Sentence embeddings (semantic)"]
        st.radio("Similarity method", options, key="w_method", horizontal=True)
        if not emb_ok:
            st.caption("Embeddings need `pip install sentence-transformers`; TF-IDF is used if it's missing.")
    with c2:
        st.write("")
        st.button("Load sample job description", on_click=load_sample_jd)

    if st.button("🔍 Analyze match", type="primary"):
        jd = st.session_state.w_jd.strip()
        try:
            if not jd:
                raise ValueError("Paste a job description first.")
            jd_info = skills_mod.analyze_job_description(jd)
            method = "embeddings" if st.session_state.w_method.startswith("Sentence") else "tfidf"
            with st.spinner("Analyzing…" if method == "tfidf" else "Loading embedding model (first run downloads ~90 MB)…"):
                match = nlp_matcher.match_resume(
                    p.raw_text, jd, list(p.skills), list(jd_info),
                    [(b.id, b.text) for b in all_bullets()], method=method)
            gap = skills_mod.skill_gap(p.skills, jd_info)
            st.session_state.jd_info, st.session_state.match, st.session_state.gap = jd_info, match, gap
            st.session_state.recs = skills_mod.recommend_learning(gap["missing"], jd_info, p.skills)
            st.session_state.ai_suggestions = None
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not analyze the match: {exc}")

    m = st.session_state.match
    if m is None:
        return
    if m.note:
        st.warning(m.note)
    g1, g2 = st.columns([1, 2])
    with g1:
        show_chart(charts.gauge(m.overall, "Overall Match"))
    with g2:
        a, b, c = st.columns(3)
        a.metric("Text similarity", f"{m.text_score:.0f}%", help=f"{m.method}; raw cosine = {m.raw_similarity:.2f}")
        b.metric("Skill coverage", "n/a" if m.skill_coverage is None else f"{m.skill_coverage:.0f}%",
                 help="Share of the job's known skills found on your resume")
        c.metric("Method", m.method.split(" (")[0])
        st.progress(min(1.0, m.overall / 100), text=f"Overall match {m.overall:.0f}%")
        st.caption("Overall = 50% text similarity + 50% skill coverage.")

    k1, k2 = st.columns(2)
    found = [k["term"] for k in m.keywords if k["in_resume"]]
    missing = [k["term"] for k in m.keywords if not k["in_resume"]]
    with k1:
        card("Job keywords found in your resume", chips(found, "good"))
    with k2:
        card("Job keywords missing", chips(missing, "bad"))
    if m.keywords:
        show_chart(charts.keyword_bar(m.keywords))
    if m.bullet_relevance:
        show_chart(charts.bullet_relevance_bar(m.bullet_relevance))
        low = sorted(m.bullet_relevance, key=lambda x: x["relevance"])[:3]
        st.caption("Least relevant bullets for this job: " + " | ".join(f"“{x['text'][:60]}…”" for x in low))
    st.button("Next: skills gap →", on_click=go_to, args=(PAGES[3],), type="primary")


# --------------------------------------------------------------------------- #
# Page 4 - Skills gap
# --------------------------------------------------------------------------- #
def page_gap() -> None:
    hero("Skills gap analysis", "What the job asks for, what you already have, and what to learn next.")
    if not require_resume():
        return
    gap, info, recs = st.session_state.gap, st.session_state.jd_info, st.session_state.recs
    if gap is None:
        st.info("Run the match on the **Job Match** page first.")
        st.button("Go to Job Match", on_click=go_to, args=(PAGES[2],))
        return
    if not info:
        st.warning("No recognisable skills were found in that job description. Try pasting the full posting.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Skills the job asks for", len(info))
    c2.metric("You have", len(gap["matched"]))
    c3.metric("Missing", len(gap["missing"]))
    st.progress(min(1.0, (gap["coverage"] or 0) / 100), text=f"Skill coverage {gap['coverage']:.0f}%")

    a, b = st.columns(2)
    with a:
        card("✅ Matched skills", chips(gap["matched"], "good"))
    with b:
        card("❌ Missing skills", chips(gap["missing"], "bad"))
    card("Your other skills (not requested in this job)", chips(gap["extra"], "info"))
    show_chart(charts.skills_gap_bar(info, gap["matched"], gap["missing"]))

    st.markdown("#### What to learn next")
    if not recs:
        st.success("You cover every skill the job asks for. 🎉")
    for r in recs:
        with st.container(border=True):
            st.markdown(f"**{r['skill']}** &nbsp; <span class='badge b-{r['priority']}'>{r['priority']} priority</span>"
                        f" &nbsp; <span class='chip'>{esc(r['category'])}</span>", unsafe_allow_html=True)
            st.write(f"{r['reason']} ({r['learning_step']})")
            st.caption(f"Next step: {r['action']} Only list it on your resume once you can discuss it.")
    st.button("Next: score & suggestions →", on_click=go_to, args=(PAGES[4],), type="primary")


# --------------------------------------------------------------------------- #
# Page 5 - Score & suggestions
# --------------------------------------------------------------------------- #
def page_score() -> None:
    hero("Resume score & improvement suggestions", "Relevance, clarity, skills and ATS-friendliness.")
    if not require_resume():
        return
    sc = compute_score()
    top, comps = st.columns([1, 3])
    with top:
        st.markdown(f"<div class='card'><div>Overall score</div><div class='bigscore' "
                    f"style='color:{score_color(sc.overall)}'>{sc.overall:.0f}</div><div>out of 100</div></div>",
                    unsafe_allow_html=True)
    with comps:
        cols = st.columns(len(sc.components()))
        hints = {"Relevance": "Match to the job description", "Clarity": "Action verbs, metrics, length",
                 "Skills": "Breadth and variety of skills", "ATS": "Standard sections & parseability"}
        for col, (name, val) in zip(cols, sc.components().items()):
            col.metric(name, f"{val:.0f}/100", help=hints[name])
            col.progress(min(1.0, val / 100))
        if sc.relevance is None:
            st.caption("Add a job description on the Job Match page to include a Relevance score.")

    t1, t2, t3 = st.tabs(["Suggestions", "ATS checklist", "Bullet-by-bullet"])
    with t1:
        sugg = compute_suggestions(sc)
        for s in sugg:
            st.markdown(f"<span class='badge b-{s['priority']}'>{s['priority']}</span> **{esc(s['area'])}** - {esc(s['action'])}",
                        unsafe_allow_html=True)
        st.divider()
        st.markdown("##### AI coaching (optional)")
        if st.button("Generate AI suggestions", disabled=not get_api_key(),
                     help="Uses your OpenAI key. Sends scores and weak bullets, not your contact details."):
            ctx = {
                "scores": {k: round(v) for k, v in sc.components().items()},
                "missing_skills": (st.session_state.gap or {}).get("missing", [])[:10],
                "matched_skills": (st.session_state.gap or {}).get("matched", [])[:10],
                "failed_ats_checks": [c["label"] for c in sc.ats_checks if not c["passed"]],
                "weakest_bullets": [{"text": b["text"], "issues": b["issues"]}
                                    for b in sorted(sc.bullet_scores, key=lambda x: x["score"])[:5]],
                "missing_job_keywords": [k["term"] for k in (st.session_state.match.keywords if st.session_state.match else [])
                                         if not k["in_resume"]][:10],
            }
            try:
                with st.spinner("Thinking…"):
                    st.session_state.ai_suggestions = llm.generate_suggestions(
                        ctx, get_api_key(), st.session_state.get("w_model") or llm.DEFAULT_MODEL)
            except llm.LLMError as exc:
                st.error(str(exc))
        if not get_api_key():
            st.caption("Add an API key in the sidebar to enable AI coaching.")
        for s in st.session_state.ai_suggestions or []:
            st.markdown(f"<span class='badge b-{s['priority']}'>{s['priority']}</span> **{esc(s['area'])}** - {esc(s['action'])}",
                        unsafe_allow_html=True)
    with t2:
        for c in sc.ats_checks:
            st.markdown(f"{'✅' if c['passed'] else '❌'} **{c['label']}** ({c['points']}/{c['max_points']})"
                        + ("" if c["passed"] else f" - {c['tip']}"))
    with t3:
        rows = [{"Score": round(b["score"]), "Bullet": b["text"], "Issues": "; ".join(b["issues"]) or "-"}
                for b in sorted(sc.bullet_scores, key=lambda x: x["score"])]
        st.dataframe(rows, hide_index=True, **_STRETCH)
    st.button("Next: analytics dashboard →", on_click=go_to, args=(PAGES[5],), type="primary")


# --------------------------------------------------------------------------- #
# Page 6 - Dashboard
# --------------------------------------------------------------------------- #
def page_dashboard() -> None:
    hero("Analytics dashboard", "All your scores and comparisons in one place.")
    if not require_resume():
        return
    p: ParsedResume = st.session_state.parsed
    sc = compute_score()
    m, gap, info = st.session_state.match, st.session_state.gap, st.session_state.jd_info

    g1, g2, g3 = st.columns(3)
    with g1:
        show_chart(charts.gauge(sc.overall, "Resume Score"))
    with g2:
        if m:
            show_chart(charts.gauge(m.overall, "Job Match"))
        else:
            st.info("Add a job description to see your match score.")
    with g3:
        if gap and gap["coverage"] is not None:
            show_chart(charts.gauge(gap["coverage"], "Skill Coverage"))
        else:
            st.info("Skill coverage appears after a job match.")

    r1, r2 = st.columns(2)
    with r1:
        show_chart(charts.radar(sc.components()))
    with r2:
        show_chart(charts.score_bars(sc.components()))

    s1, s2 = st.columns(2)
    with s1:
        if p.skill_categories:
            show_chart(charts.category_donut(p.skill_categories))
    with s2:
        if gap and info:
            show_chart(charts.skills_gap_bar(info, gap["matched"], gap["missing"]))
        else:
            st.info("The skill-gap chart appears after a job match.")

    if m and m.bullet_relevance:
        show_chart(charts.bullet_relevance_bar(m.bullet_relevance))
    opt = st.session_state.opt
    if opt:
        reps = get_replacements()
        before = [analyze_bullet(r["original"]).score for r in opt.values()]
        after = [analyze_bullet(reps.get(i, r["original"])).score for i, r in opt.items()]
        show_chart(charts.before_after(before, after))
        st.caption("'Optimized' reflects only the rewrites you ticked **Use** on.")
    else:
        st.info("Optimize some bullets to see the before/after comparison.")


# --------------------------------------------------------------------------- #
# Page 7 - Download
# --------------------------------------------------------------------------- #
def page_download() -> None:
    hero("Download your optimized resume content", "Accepted rewrites replace the originals; everything else is kept as-is.")
    if not require_resume():
        return
    p: ParsedResume = st.session_state.parsed
    reps, manual = get_replacements(), manual_bullet_objects()
    st.metric("Rewrites included", len(reps))
    if not st.session_state.opt:
        st.info("You haven't accepted any rewrites yet, so this export equals your original content.")

    fmt = st.radio("Format", ["Word (.docx)", "Markdown (.md)", "Plain text (.txt)", "Analysis report (.json)"],
                   horizontal=True, key="w_fmt")
    base = (p.name or "resume").lower().replace(" ", "_")
    try:
        if fmt.startswith("Word"):
            data, name, mime = (export.to_docx(p, reps, manual), f"{base}_optimized.docx",
                                "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            preview = export.to_markdown(p, reps, manual)
        elif fmt.startswith("Markdown"):
            preview = export.to_markdown(p, reps, manual)
            data, name, mime = preview, f"{base}_optimized.md", "text/markdown"
        elif fmt.startswith("Plain"):
            preview = export.to_text(p, reps, manual)
            data, name, mime = preview, f"{base}_optimized.txt", "text/plain"
        else:
            sc = compute_score()
            m = st.session_state.match
            payload = {
                "resume": p.to_dict(),
                "scores": {k: round(v, 1) for k, v in {**sc.components(), "Overall": sc.overall}.items()},
                "job_match": None if not m else {"overall": m.overall, "text_score": m.text_score,
                                                 "skill_coverage": m.skill_coverage, "method": m.method,
                                                 "keywords": m.keywords},
                "skill_gap": st.session_state.gap,
                "learning_recommendations": st.session_state.recs,
                "suggestions": st.session_state.ai_suggestions or compute_suggestions(sc),
                "accepted_rewrites": [{"original": st.session_state.opt[i]["original"], "optimized": t}
                                      for i, t in reps.items()],
            }
            preview = export.to_json(payload)
            data, name, mime = preview, f"{base}_analysis.json", "application/json"
        st.download_button("⬇️ Download", data=data, file_name=name, mime=mime, type="primary")
        with st.expander("Preview", expanded=True):
            st.code(preview, language="markdown" if "Markdown" in fmt or "Word" in fmt else None)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not build the download: {exc}")
    st.caption("Review all content before sending it to employers: you are responsible for its accuracy.")


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
sidebar()
{
    PAGES[0]: page_upload, PAGES[1]: page_optimizer, PAGES[2]: page_match, PAGES[3]: page_gap,
    PAGES[4]: page_score, PAGES[5]: page_dashboard, PAGES[6]: page_download,
}[st.session_state.page]()
