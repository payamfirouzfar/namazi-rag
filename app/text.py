"""Text cleaning and tokenizing. Works for English and Persian."""
import re
import unicodedata

# PDFs and Persian keyboards often mix Arabic and Persian letter forms
ARABIC_TO_PERSIAN = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه"})

SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+|\n{2,}")
WORD = re.compile(r"\w+", re.UNICODE)

STOPWORDS = set(
    """a an and are as at be by for from has have in is it its of on or that the this to was were
    with what which who how when does do can should would
    و در به از که این را با برای است آن یا هم تا بر""".split()
)

# Common clinical abbreviations. The expansion is added next to the original token so
# a query for "HTN" also matches text that says "hypertension" and the other way round.
ABBREVIATIONS = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "mi": "myocardial infarction",
    "chf": "congestive heart failure",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "aki": "acute kidney injury",
    "uti": "urinary tract infection",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "dka": "diabetic ketoacidosis",
    "inr": "international normalized ratio",
    "wbc": "white blood cell",
    "rbc": "red blood cell",
    "hb": "hemoglobin",
    "hba1c": "glycated hemoglobin",
    "ecg": "electrocardiogram",
    "ekg": "electrocardiogram",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "icu": "intensive care unit",
    "bp": "blood pressure",
    "ace": "angiotensin converting enzyme",
    "arb": "angiotensin receptor blocker",
    "nsaid": "nonsteroidal anti inflammatory drug",
    "gfr": "glomerular filtration rate",
    "egfr": "glomerular filtration rate",
    "cbc": "complete blood count",
    "tb": "tuberculosis",
}


def clean_text(text: str) -> str:
    """Fix the usual PDF damage: ligatures, hyphenated line breaks, odd spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(ARABIC_TO_PERSIAN)
    text = text.replace("‌", " ")  # zero-width non-joiner (Persian half space)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words split across lines
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    parts = SENTENCE_END.split(text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p and p.strip()]


def tokenize(text: str) -> list[str]:
    """Lowercase words without stopwords, plus expansions of clinical abbreviations."""
    tokens = []
    for word in WORD.findall(clean_text(text).lower()):
        if word in STOPWORDS:
            continue
        tokens.append(word)
        if word in ABBREVIATIONS:
            tokens.extend(ABBREVIATIONS[word].split())
    return tokens
