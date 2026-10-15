"""PDF text extraction and rule-based resume structuring.

Pipeline:  PDF bytes -> raw text -> section split -> blocks (bullets / text)
           -> entries (jobs, projects) -> bullets with stable ids + skills.

Everything here is heuristic by nature (resumes have no standard layout), so the
parser exposes its raw text and the UI lets users correct it.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .skills import categorize, extract_skills


class ResumeParseError(Exception):
    """Raised when a resume cannot be read or contains no usable text."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Block:
    kind: str                      # "bullet" | "text"
    text: str
    bullet_id: Optional[int] = None


@dataclass
class Bullet:
    id: int
    section: str                   # experience | projects | other | manual
    entry: str                     # job / project header this bullet belongs to
    text: str


@dataclass
class Entry:
    header: str
    bullets: List[Bullet] = field(default_factory=list)


@dataclass
class ParsedResume:
    raw_text: str
    num_pages: int
    word_count: int
    name: str
    email: Optional[str]
    phone: Optional[str]
    links: List[str]
    sections: Dict[str, List[str]]
    blocks: Dict[str, List[Block]]
    skills: Dict[str, int]                       # canonical skill -> mentions
    skill_categories: Dict[str, List[str]]
    extra_skills: List[str]                      # listed in Skills section, not in taxonomy
    education: List[str]
    experience: List[Entry]
    projects: List[Entry]
    bullets: List[Bullet]

    def to_dict(self) -> dict:
        return {
            "name": self.name, "email": self.email, "phone": self.phone, "links": self.links,
            "skills": self.skill_categories, "other_listed_skills": self.extra_skills,
            "education": self.education,
            "experience": [{"header": e.header, "bullets": [b.text for b in e.bullets]} for e in self.experience],
            "projects": [{"header": e.header, "bullets": [b.text for b in e.bullets]} for e in self.projects],
        }


# --------------------------------------------------------------------------- #
# Regexes & constants
# --------------------------------------------------------------------------- #
SECTION_ALIASES: Dict[str, List[str]] = {
    "summary": ["summary", "professional summary", "profile", "objective", "career objective",
                "about me", "professional profile"],
    "experience": ["experience", "work experience", "professional experience", "employment history",
                   "work history", "employment", "internships", "internship experience"],
    "education": ["education", "academic background", "academics", "education and training",
                  "educational qualifications"],
    "skills": ["skills", "technical skills", "core competencies", "skills & tools", "skills and tools",
               "technologies", "key skills", "technical proficiencies"],
    "projects": ["projects", "personal projects", "academic projects", "selected projects",
                 "key projects"],
    "certifications": ["certifications", "certificates", "licenses", "licenses & certifications",
                       "courses", "certifications and courses"],
    "awards": ["awards", "honors", "achievements", "accomplishments", "awards and achievements"],
}
_HEADING_TO_SECTION = {a: sec for sec, aliases in SECTION_ALIASES.items() for a in aliases}

SECTION_TITLES = {
    "summary": "Professional Summary", "experience": "Experience", "education": "Education",
    "skills": "Skills", "projects": "Projects", "certifications": "Certifications",
    "awards": "Awards & Achievements",
}

BULLET_RE = re.compile(r"^\s*(?:[•●▪■◦‣·]\s*|[-–—*]\s+)")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE_CANDIDATE_RE = re.compile(r"\+?\d[\d\s().-]{8,}\d")
LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?(?:linkedin\.com|github\.com)/[\w\-./]+", re.I)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
DATE_RE = re.compile(
    rf"(?:{_MONTH}\s+)?(?:19|20)\d{{2}}\s*(?:[-–—]|to)\s*(?:present|current|now|(?:{_MONTH}\s+)?(?:19|20)\d{{2}})",
    re.I,
)

_PRIVATE_BULLETS = {ord(c): "•" for c in "\uf0b7\uf0a7\uf076\uf0d8\uf0fc\uf0a8\u2022"}


# --------------------------------------------------------------------------- #
# PDF extraction
# --------------------------------------------------------------------------- #
def extract_text_from_pdf(data: bytes) -> tuple[str, int]:
    """Return (clean_text, page_count). Raises ResumeParseError with a friendly message."""
    if not data:
        raise ResumeParseError("The uploaded file is empty.")
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:  # noqa: BLE001
                raise ResumeParseError("This PDF is password-protected. Please upload an unlocked copy.") from exc
        pages = [(p.extract_text() or "") for p in reader.pages]
    except ResumeParseError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError) as exc:
        raise ResumeParseError("Could not read this file as a PDF. Is it corrupted?") from exc
    except Exception as exc:  # noqa: BLE001 - pypdf can raise many exception types
        raise ResumeParseError(f"Unexpected error while reading the PDF: {exc}") from exc

    text = clean_text("\n".join(pages))
    if len(text.split()) < 30:
        raise ResumeParseError(
            "Almost no text could be extracted. The PDF may be a scanned image "
            "(OCR is not supported) - please export a text-based PDF from Word/Docs, "
            "or paste your text via the 'Edit extracted text' box."
        )
    return text, len(pages)


def clean_text(text: str) -> str:
    text = text.translate(_PRIVATE_BULLETS).replace("\u00a0", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Structuring
# --------------------------------------------------------------------------- #
def _detect_heading(line: str) -> Optional[str]:
    norm = re.sub(r"[^a-z& ]", "", line.lower()).strip()
    norm = re.sub(r"\s+", " ", norm)
    if 0 < len(line) <= 45 and norm in _HEADING_TO_SECTION:
        return _HEADING_TO_SECTION[norm]
    return None


def split_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {"header": []}
    current = "header"
    for line in text.split("\n"):
        sec = _detect_heading(line.strip())
        if sec:
            current = sec
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return sections


def _is_continuation(prev: str, line: str) -> bool:
    """Is `line` the wrapped second line of the bullet whose text is `prev`?"""
    if DATE_RE.search(line):
        return False
    if line[0].islower():
        return True
    if prev.rstrip().endswith((".", ";", ":", "!", "?")):
        return False
    words = line.split()
    title_case = sum(w[0].isupper() for w in words) / len(words) >= 0.7
    return len(words) >= 4 and not title_case and not line.isupper()


def group_blocks(lines: List[str], sentence_fallback: bool = False) -> List[Block]:
    blocks: List[Block] = []
    pending_bullet = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if BULLET_RE.match(line):
            body = BULLET_RE.sub("", line).strip()
            if body:
                blocks.append(Block("bullet", body))
                pending_bullet = False
            else:
                pending_bullet = True        # marker alone on its line
            continue
        if pending_bullet:
            blocks.append(Block("bullet", line))
            pending_bullet = False
        elif blocks and blocks[-1].kind == "bullet" and _is_continuation(blocks[-1].text, line):
            blocks[-1].text += " " + line
        else:
            blocks.append(Block("text", line))
    if sentence_fallback and not any(b.kind == "bullet" for b in blocks):
        for b in blocks:
            if b.kind == "text" and len(b.text.split()) >= 7 and not DATE_RE.search(b.text):
                b.kind = "bullet"
    return blocks


def _build_entries(blocks: List[Block], section: str, next_id: int) -> tuple[List[Entry], int]:
    entries: List[Entry] = []
    cur: Optional[Entry] = None
    for b in blocks:
        if b.kind == "text":
            if cur is None or cur.bullets:
                cur = Entry(header=b.text)
                entries.append(cur)
            else:
                cur.header += " | " + b.text
        else:
            if cur is None:
                cur = Entry(header="")
                entries.append(cur)
            b.bullet_id = next_id
            cur.bullets.append(Bullet(next_id, section, cur.header, b.text))
            next_id += 1
    return entries, next_id


def _extract_phone(text: str) -> Optional[str]:
    for m in PHONE_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 10 <= len(digits) <= 13:
            return m.group(0).strip()
    return None


def _guess_name(header_lines: List[str]) -> str:
    for line in header_lines:
        s = line.strip()
        if s and "@" not in s and not re.search(r"\d", s) and 1 <= len(s.split()) <= 5:
            return s.title() if s.isupper() else s
    return ""


def _listed_skills(lines: List[str], known: set) -> List[str]:
    out: List[str] = []
    for line in lines:
        body = line.split(":", 1)[1] if ":" in line else line
        for item in re.split(r"[,|;•·/]", body):
            item = item.strip(" -–—*.")
            if 1 < len(item) <= 30 and len(item.split()) <= 3 and item not in known and item not in out:
                out.append(item)
    return out


def parse_resume(text: str, num_pages: int = 1) -> ParsedResume:
    text = clean_text(text)
    sections = split_sections(text)
    header_lines = sections.get("header", [])

    blocks: Dict[str, List[Block]] = {}
    for sec, lines in sections.items():
        blocks[sec] = group_blocks(lines, sentence_fallback=sec in ("experience", "projects"))

    bullets: List[Bullet] = []
    next_id = 0
    experience, next_id = _build_entries(blocks.get("experience", []), "experience", next_id)
    projects, next_id = _build_entries(blocks.get("projects", []), "projects", next_id)
    for e in experience + projects:
        bullets.extend(e.bullets)

    if not bullets:  # non-standard headings: harvest bullet-marked lines anywhere else
        for sec, blks in blocks.items():
            if sec in ("skills", "education", "summary", "certifications"):
                continue
            for b in blks:
                if b.kind == "bullet":
                    b.bullet_id = next_id
                    bullets.append(Bullet(next_id, "other", sec.title(), b.text))
                    next_id += 1

    skill_counts = dict(extract_skills(text))
    skills_lines = [ln.strip() for ln in sections.get("skills", []) if ln.strip()]
    education = [ln.strip() for ln in sections.get("education", []) if ln.strip()][:12]

    email = EMAIL_RE.search(text)
    return ParsedResume(
        raw_text=text,
        num_pages=num_pages,
        word_count=len(text.split()),
        name=_guess_name(header_lines),
        email=email.group(0) if email else None,
        phone=_extract_phone("\n".join(header_lines) or text),
        links=sorted({m.group(0) for m in LINK_RE.finditer(text)}),
        sections=sections,
        blocks=blocks,
        skills=skill_counts,
        skill_categories=categorize(skill_counts),
        extra_skills=_listed_skills(skills_lines, set(skill_counts)),
        education=education,
        experience=experience,
        projects=projects,
        bullets=bullets,
    )
