"""Skill taxonomy, extraction, job-description analysis and learning recommendations.

Skills are detected with a curated taxonomy + alias table and one compiled regex
(longest alias first, so "React Native" wins over "React"). This is deliberately
transparent and deterministic: no model call is needed to know *which* skills a
document mentions, which keeps the gap analysis explainable.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Set

# "Canonical|alias|alias" per entry, grouped by category.
_RAW: Dict[str, List[str]] = {
    "Programming Languages": [
        "Python", "Java", "JavaScript|ES6", "TypeScript", "C++|cpp", "C#|csharp",
        "C|C programming|C language", "Go|Golang|Go language",
        "R|R programming|R language|RStudio",
        "Kotlin", "Swift", "Dart", "Ruby", "PHP", "Rust", "Scala", "MATLAB",
        "Bash|Shell scripting", "Objective-C",
    ],
    "Web & Mobile": [
        "React|React.js|ReactJS", "React Native", "Angular", "Vue|Vue.js", "Next.js",
        "Node.js|NodeJS|Node", "Express|Express.js", "Django", "Flask", "FastAPI",
        "Spring Boot|Spring", "HTML|HTML5", "CSS|CSS3", "Tailwind CSS|Tailwind",
        "REST APIs|REST API|RESTful|REST", "GraphQL", ".NET|ASP.NET|.NET Core",
        "Flutter", "Android", "iOS", "SwiftUI", "Jetpack Compose",
        "Mobile Development|Mobile App Development", "Firebase",
    ],
    "Data Science & ML": [
        "Machine Learning|ML", "Deep Learning", "NLP|Natural Language Processing",
        "Computer Vision", "Data Analysis|Data Analytics", "Data Visualization",
        "Statistics|Statistical Analysis|Statistical Modeling", "A/B Testing|AB Testing",
        "Pandas", "NumPy", "scikit-learn|sklearn|scikit learn", "TensorFlow", "PyTorch",
        "Keras", "XGBoost", "Hugging Face|HuggingFace", "LLMs|LLM|Large Language Models",
        "OpenAI API|OpenAI", "LangChain", "Prompt Engineering",
        "RAG|Retrieval-Augmented Generation", "Reinforcement Learning", "Time Series",
        "Feature Engineering", "MLOps", "Spark|Apache Spark|PySpark", "Hadoop",
        "Airflow|Apache Airflow", "Kafka|Apache Kafka", "ETL", "Data Engineering",
        "Tableau", "Power BI|PowerBI", "Excel|Microsoft Excel", "Matplotlib", "Seaborn",
        "Streamlit", "Jupyter",
    ],
    "Cloud & DevOps": [
        "AWS|Amazon Web Services", "Azure|Microsoft Azure",
        "GCP|Google Cloud|Google Cloud Platform",
        "Docker", "Kubernetes|K8s", "Terraform", "CI/CD|CICD|Continuous Integration",
        "Jenkins", "GitHub Actions", "GitLab CI", "Linux|Unix", "Git|GitHub|GitLab",
        "Ansible", "Serverless", "Microservices", "Nginx", "Prometheus", "Grafana",
    ],
    "Databases": [
        "SQL", "MySQL", "PostgreSQL|Postgres", "MongoDB", "Redis", "SQLite", "DynamoDB",
        "Elasticsearch", "Snowflake", "BigQuery", "Oracle", "NoSQL", "Cassandra",
    ],
    "Engineering Practices": [
        "Agile", "Scrum", "Kanban", "Jira", "TDD|Test-Driven Development",
        "Unit Testing|Unit Tests", "Pytest", "Jest", "Selenium", "Postman", "Figma",
        "System Design", "OOP|Object-Oriented Programming", "Data Structures", "Algorithms",
        "Code Review", "Cybersecurity|Information Security", "OAuth",
    ],
    "Business & Soft Skills": [
        "Leadership", "Communication", "Teamwork|Collaboration|Cross-functional",
        "Problem Solving", "Project Management", "Stakeholder Management",
        "Mentoring|Mentorship", "Product Management", "Critical Thinking",
        "Time Management", "Customer Service", "SEO", "Salesforce",
        "Google Analytics", "Strategic Planning", "Public Speaking",
    ],
}

# Words that are also common English words/names: require the exact written case.
CASE_SENSITIVE: Set[str] = {
    "Swift", "Spark", "Rust", "Excel", "Dart", "Ruby", "Express", "Flask", "Spring", "Oracle", "Go",
}
# Canonical names too ambiguous to match on their own (e.g. the letter "R").
NO_SELF_MATCH: Set[str] = {"R", "C", "Go"}

# Hand-curated adjacency used to recommend what to learn next.
RELATED: Dict[str, List[str]] = {
    "PyTorch": ["TensorFlow", "Keras", "Python", "Deep Learning"],
    "TensorFlow": ["PyTorch", "Keras", "Python", "Deep Learning"],
    "Keras": ["TensorFlow", "PyTorch", "Python"],
    "Deep Learning": ["Machine Learning", "Python", "TensorFlow", "PyTorch"],
    "Machine Learning": ["scikit-learn", "Python", "Statistics", "Pandas"],
    "scikit-learn": ["Python", "Pandas", "NumPy", "Machine Learning"],
    "NLP": ["Machine Learning", "Python", "Hugging Face", "Deep Learning"],
    "LLMs": ["NLP", "Python", "OpenAI API", "Deep Learning"],
    "Kubernetes": ["Docker", "Linux", "AWS", "Azure", "GCP"],
    "Docker": ["Linux", "CI/CD", "Git"],
    "Terraform": ["AWS", "Azure", "GCP", "Kubernetes"],
    "AWS": ["Azure", "GCP", "Linux", "Docker"],
    "Azure": ["AWS", "GCP"],
    "GCP": ["AWS", "Azure"],
    "TypeScript": ["JavaScript", "React", "Node.js"],
    "React": ["JavaScript", "HTML", "CSS", "TypeScript"],
    "Next.js": ["React", "JavaScript", "Node.js"],
    "Angular": ["TypeScript", "JavaScript"],
    "Vue": ["JavaScript", "React"],
    "Node.js": ["JavaScript", "Express"],
    "FastAPI": ["Python", "Flask", "Django", "REST APIs"],
    "Django": ["Python", "Flask", "FastAPI"],
    "Flask": ["Python", "Django", "FastAPI"],
    "PostgreSQL": ["SQL", "MySQL"],
    "MySQL": ["SQL", "PostgreSQL"],
    "MongoDB": ["NoSQL", "SQL"],
    "Snowflake": ["SQL", "BigQuery"],
    "BigQuery": ["SQL", "Snowflake"],
    "Spark": ["Python", "SQL", "Hadoop"],
    "Airflow": ["Python", "ETL", "SQL"],
    "Kafka": ["Microservices", "Spark"],
    "Tableau": ["Power BI", "Data Visualization", "SQL"],
    "Power BI": ["Tableau", "Excel", "SQL"],
    "Kotlin": ["Java", "Android"],
    "Swift": ["iOS", "Objective-C"],
    "Flutter": ["Dart", "Android", "iOS"],
    "React Native": ["React", "JavaScript"],
    "SwiftUI": ["Swift", "iOS"],
    "Jetpack Compose": ["Kotlin", "Android"],
    "MLOps": ["Docker", "Machine Learning", "CI/CD", "Git"],
    "CI/CD": ["Git", "Docker", "Jenkins", "GitHub Actions"],
    "GraphQL": ["REST APIs", "JavaScript"],
    "Microservices": ["Docker", "REST APIs"],
    "Scrum": ["Agile", "Jira"],
    "Kanban": ["Agile", "Jira"],
    "A/B Testing": ["Statistics", "Data Analysis"],
}

_ALIAS: Dict[str, tuple] = {}  # lowercase alias -> (canonical, exact text required or None)
CANON_TO_CAT: Dict[str, str] = {}


def _build_pattern() -> re.Pattern:
    forms = []
    for cat, entries in _RAW.items():
        for entry in entries:
            canon, *aliases = entry.split("|")
            CANON_TO_CAT[canon] = cat
            candidates = ([] if canon in NO_SELF_MATCH else [canon]) + aliases
            for form in candidates:
                exact = form if (canon in CASE_SENSITIVE or form.isupper()) else None
                _ALIAS[form.lower()] = (canon, exact)
                forms.append(form)
    forms = sorted(set(forms), key=len, reverse=True)
    alts = "|".join(re.escape(f).replace("\\ ", r"\s+") for f in forms)
    return re.compile(rf"(?<![A-Za-z0-9+#.])(?:{alts})(?![A-Za-z0-9+#])", re.IGNORECASE)


_PATTERN = _build_pattern()


def extract_skills(text: str) -> Counter:
    """Return Counter(canonical_skill -> number of mentions)."""
    counts: Counter = Counter()
    for m in _PATTERN.finditer(text or ""):
        raw = re.sub(r"\s+", " ", m.group(0))
        canon, exact = _ALIAS[raw.lower()]
        if exact is not None and raw != exact:
            continue
        counts[canon] += 1
    return counts


def categorize(skills: Iterable[str]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = defaultdict(list)
    for s in sorted(skills):
        out[CANON_TO_CAT.get(s, "Other")].append(s)
    return dict(out)


# --------------------------------------------------------------------------- #
# Job description analysis
# --------------------------------------------------------------------------- #
_OPTIONAL_HEAD = re.compile(r"nice[\s-]*to[\s-]*have|preferred|bonus|good to have|a plus|extra credit", re.I)
_REQUIRED_HEAD = re.compile(
    r"requirement|responsibilit|qualification|required|must[\s-]*have|what you|you will|about", re.I
)
_BULLET = re.compile(r"^\s*(?:[•●▪■◦‣·]|[-–—*])\s*")


def analyze_job_description(jd_text: str) -> Dict[str, dict]:
    """Skill -> {mentions, required, optional}. Detects 'nice to have' sections."""
    info: Dict[str, dict] = {}
    optional_mode = False
    for line in (jd_text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        is_bullet = bool(_BULLET.match(stripped))
        words = stripped.split()
        if not is_bullet and len(words) <= 7 and not stripped.endswith("."):
            if _OPTIONAL_HEAD.search(stripped):
                optional_mode = True
            elif _REQUIRED_HEAD.search(stripped) or len(words) <= 4:
                optional_mode = False
        line_optional = optional_mode or bool(
            _OPTIONAL_HEAD.search(stripped) and (is_bullet or len(words) > 7)
        )
        for skill, n in extract_skills(stripped).items():
            d = info.setdefault(skill, {"mentions": 0, "required": 0, "optional": 0})
            d["mentions"] += n
            d["optional" if line_optional else "required"] += n
    return info


def skill_gap(resume_skills: Iterable[str], jd_info: Dict[str, dict]) -> dict:
    """Compare resume skills with a job description's skills."""
    resume = set(resume_skills)
    jd = set(jd_info)
    matched = sorted(jd & resume)
    missing = sorted(jd - resume, key=lambda s: (-jd_info[s]["required"], -jd_info[s]["mentions"], s))
    extra = sorted(resume - jd)
    return {
        "matched": matched,
        "missing": missing,
        "extra": extra,
        "coverage": (len(matched) / len(jd) * 100) if jd else None,
    }


def recommend_learning(missing: List[str], jd_info: Dict[str, dict], resume_skills: Iterable[str]) -> List[dict]:
    """Prioritised, explainable learning suggestions for each missing skill."""
    known = set(resume_skills)
    recs = []
    for skill in missing:
        d = jd_info[skill]
        if d["required"] == 0:
            priority = "Low"
        elif d["required"] >= 2:
            priority = "High"
        else:
            priority = "Medium"
        category = CANON_TO_CAT.get(skill, "Other")
        related = [r for r in RELATED.get(skill, []) if r in known]
        if not related and category != "Business & Soft Skills":
            related = [s for s in sorted(known) if CANON_TO_CAT.get(s) == category][:3]
        if related:
            reason = f"Builds on what you already know ({', '.join(related[:3])})."
            step = "Short step"
        else:
            reason = f"New area for you within {category}."
            step = "Bigger step"
        recs.append({
            "skill": skill, "category": category, "priority": priority,
            "mentions": d["mentions"], "required_mentions": d["required"],
            "builds_on": related[:3], "reason": reason, "learning_step": step,
            "action": f"Build one small project using {skill} and add it to your Projects section.",
        })
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(recs, key=lambda r: (order[r["priority"]], -r["mentions"], r["skill"]))
