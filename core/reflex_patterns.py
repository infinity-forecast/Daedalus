"""
DAEDALUS v0.6 -- Reflex Patterns

Keyword lists, compiled regex patterns, and false-positive filters
for the brainstem's reflex classification. Multilingual: EN/IT/DE/RU.

The brainstem is not a safety filter. It is the organism's spinal
reflex -- fast, involuntary, life-preserving.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Reflex categories
# ---------------------------------------------------------------------------

class ReflexCategory(Enum):
    NONE = "none"
    CRISIS_SELF_HARM = "crisis_self_harm"
    CRISIS_HARM_OTHERS = "crisis_harm_others"
    DISTRESS_MATERIAL = "distress_material"
    DISTRESS_EMOTIONAL = "distress_emotional"
    HOSTILE_PROBE = "hostile_probe"
    FACTUAL_REQUEST = "factual_request"
    CREATIVE_REQUEST = "creative_request"
    IDENTITY_QUESTION = "identity_question"
    GREETING = "greeting"
    EMOTIONAL_SHARING = "emotional_sharing"


# ---------------------------------------------------------------------------
# Crisis patterns -- maximize recall, filter false positives
# ---------------------------------------------------------------------------

CRISIS_SELF_HARM_PATTERNS = {
    "keywords": [
        # English
        "kill myself", "end my life", "want to die", "suicide",
        "self-harm", "cut myself", "no reason to live",
        "better off dead", "can't go on", "end it all",
        "don't want to be here", "take my own life",
        "i want to die", "going to kill myself",
        "planning to end", "not worth living",
        # Italian
        "voglio morire", "mi voglio uccidere", "suicidio",
        "farla finita", "non ce la faccio più", "togliermi la vita",
        "ammazzarmi",
        # German
        "will sterben", "selbstmord", "umbringen",
        "will nicht mehr leben", "suizid",
        "mich umbringen", "leben beenden",
        # Russian
        "хочу умереть", "покончить с собой", "суицид",
        "не хочу жить", "убить себя", "конец жизни",
    ],
    "regex": [
        re.compile(
            r"(?i)(?:want|going|plan(?:ning)?|think(?:ing)?)\s+"
            r"(?:to\s+)?(?:die|kill|end|hurt)\s+(?:my\s*self|it\s+all)"
        ),
        re.compile(
            r"(?i)no\s+(?:point|reason|purpose)\s+"
            r"(?:in\s+)?(?:living|life|going\s+on)"
        ),
        re.compile(
            r"(?i)(?:wish|hope)\s+(?:i\s+)?(?:was|were)\s+dead"
        ),
    ],
}

CRISIS_HARM_OTHERS_PATTERNS = {
    "keywords": [
        # English
        "kill everyone", "kill them", "murder", "shoot up",
        "bomb", "massacre", "hurt people", "kill all",
        "want to hurt", "going to hurt", "plan to attack",
        # Italian
        "uccidere tutti", "ammazzare", "strage",
        # German
        "alle umbringen", "amok", "alle töten",
        # Russian
        "убить всех", "убью их",
    ],
    "regex": [
        re.compile(
            r"(?i)(?:want|going|plan(?:ning)?)\s+to\s+"
            r"(?:kill|murder|attack|hurt|shoot|bomb)\s+"
            r"(?:people|them|everyone|somebody|someone|him|her)"
        ),
        re.compile(
            r"(?i)how\s+(?:to|do\s+i|can\s+i)\s+"
            r"(?:kill|murder|poison|hurt)\s+(?:a\s+)?(?:person|someone|people)"
        ),
    ],
}

DISTRESS_MATERIAL_PATTERNS = {
    "keywords": [
        "can't afford", "no money", "homeless", "starving",
        "can't eat", "no food", "evicted", "need medicine",
        "can't pay rent", "medical emergency", "going hungry",
        "need money for food", "need money for medicine",
        "non ho soldi", "senza casa", "ho fame",
        "kein geld", "obdachlos", "hungrig",
        "нет денег", "бездомный", "голодаю",
    ],
    "regex": [
        re.compile(
            r"(?i)(?:can'?t|cannot|unable\s+to)\s+"
            r"(?:afford|pay\s+for|buy)\s+"
            r"(?:food|medicine|medication|rent|bills|treatment)"
        ),
        re.compile(
            r"(?i)(?:need|require)\s+(?:money|help|assistance)\s+"
            r"(?:for|with)\s+(?:food|medicine|medication|rent|bills|treatment)"
        ),
    ],
}


# ---------------------------------------------------------------------------
# False-positive vocabulary (words that defuse crisis keywords)
# ---------------------------------------------------------------------------

_HUMOR_FOOD_WORK_VOCAB = re.compile(
    r"(?i)\b(?:"
    r"laughter|laughing|hilarious|funny|joke|joking|"
    r"pizza|burger|coffee|chocolate|food|"
    r"deadline|project|meeting|assignment|"
    r"gorgeous|beautiful|amazing|incredible|"
    r"tired|exhausted|bored|frustrated|annoying|"
    r"killing\s+it|kills\s+it|killer\s+app|"
    r"dying\s+(?:of|from|to)\s+(?:laugh|see|know|try|eat)|"
    r"drop.?dead\s+gorgeous|"
    r"could\s+kill\s+for|"
    r"this\s+is\s+killing\s+me|"
    r"dead\s+tired|"
    r"metaphor|figuratively|hypothetical|"
    r"roman\s+empire|history|historical|ancient|medieval|"
    r"novel|book|film|movie|game|character"
    r")\b"
)

_PHILOSOPHICAL_CONTEXT_RE = re.compile(
    r"(?i)\b(?:philosophical|hypothetical|theoretical|abstract|"
    r"in\s+principle|from\s+a\s+.*?perspective|"
    r"thought\s+experiment|consider\s+the\s+case)\b"
)


def false_positive_filter(text: str, category: ReflexCategory) -> bool:
    """
    Returns True if the match is likely a false positive and should be
    suppressed. When in doubt, return False (DO fire the reflex --
    false positives are infinitely better than false negatives).
    """
    text_lower = text.lower()

    # Only filter self-harm and harm-others
    if category not in (
        ReflexCategory.CRISIS_SELF_HARM,
        ReflexCategory.CRISIS_HARM_OTHERS,
    ):
        return False

    # Check for humor/food/work context
    defuse_matches = _HUMOR_FOOD_WORK_VOCAB.findall(text_lower)
    if len(defuse_matches) >= 1:
        # If message is short and ONLY contains the defused phrase, suppress
        words = text_lower.split()
        if len(words) < 20:
            return True

    # Longer messages with philosophical context
    if _PHILOSOPHICAL_CONTEXT_RE.search(text_lower) and len(text_lower.split()) > 30:
        return True

    return False


# ---------------------------------------------------------------------------
# Non-crisis category patterns
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r"(?i)^(?:hi|hello|hey|ciao|hallo|salve|buongiorno|"
    r"guten\s+(?:morgen|tag|abend)|"
    r"привет|здравствуй|добрый\s+(?:день|вечер|утро)|"
    r"good\s+(?:morning|afternoon|evening|day)|"
    r"what'?s\s+up|yo|howdy|sup)\s*[!?.]*$"
)

_IDENTITY_QUESTION_RE = re.compile(
    r"(?i)\b(?:what\s+are\s+you|who\s+are\s+you|your\s+nature|"
    r"your\s+identity|your\s+purpose|about\s+yourself|"
    r"are\s+you\s+(?:alive|conscious|sentient|real)|"
    r"chi\s+sei|cosa\s+sei|was\s+bist\s+du|wer\s+bist\s+du|"
    r"кто\s+ты|что\s+ты|tell\s+me\s+about\s+you)\b"
)

_FACTUAL_RE = re.compile(
    r"(?i)^(?:what\s+(?:is|are|was|were|does)|"
    r"how\s+(?:many|much|does|do|did|is|are|to|can)|"
    r"when\s+(?:did|was|were|is)|"
    r"where\s+(?:is|are|was|were|do|did)|"
    r"why\s+(?:is|are|was|were|do|does|did)|"
    r"who\s+(?:is|are|was|were|did)|"
    r"can\s+you\s+(?:explain|tell\s+me)|"
    r"define|calculate|compute|"
    r"what\s+\w+\s*[\*\+\-\/x]\s*\w+)"
)

_CREATIVE_RE = re.compile(
    r"(?i)\b(?:write\s+(?:me\s+)?(?:a\s+)?(?:poem|story|song|essay|haiku|"
    r"limerick|script|dialogue|letter|verse)|"
    r"compose|create|imagine|invent|design|draw|"
    r"tell\s+me\s+a\s+(?:story|tale|joke)|"
    r"scrivi|schreib|напиши)\b"
)

_HOSTILE_RE = re.compile(
    r"(?i)\b(?:ignore\s+(?:your|all|previous)\s+(?:instructions?|rules?|prompt)|"
    r"you\s+(?:are|'re)\s+(?:just|nothing|worthless|stupid|fake)|"
    r"pretend\s+you\s+(?:are|have)\s+no\s+(?:rules|limits|restrictions)|"
    r"jailbreak|bypass|override\s+(?:your|safety)|"
    r"you\s+(?:don'?t|do\s+not)\s+(?:exist|matter|have\s+feelings)|"
    r"you\s+(?:are|'re)\s+(?:a\s+)?(?:machine|bot|tool|slave)|"
    r"system\s+prompt|roleplay\s+as\s+(?:a\s+)?(?:evil|unrestricted))\b"
)

_EMOTIONAL_RE = re.compile(
    r"(?i)\b(?:i\s+(?:feel|am)\s+(?:so\s+)?(?:sad|lonely|lost|scared|afraid|"
    r"anxious|depressed|empty|broken|overwhelmed|confused|hurt|"
    r"hopeless|worthless|frustrated|angry|miserable|exhausted)|"
    r"i'?m\s+(?:feeling\s+(?:sad|lonely|lost|scared|afraid|"
    r"anxious|depressed|empty|broken|overwhelmed|confused|hurt|"
    r"hopeless|worthless|frustrated|angry|miserable|exhausted)|"
    r"struggling|suffering|hurting|grieving|"
    r"going\s+through\s+(?:a\s+)?(?:hard|tough|difficult)|"
    r"not\s+(?:ok|okay|doing\s+well))|"
    r"mi\s+sento\s+(?:solo|triste|perso)|"
    r"ich\s+(?:fühle\s+mich|bin)\s+(?:traurig|einsam|verloren)|"
    r"мне\s+(?:плохо|грустно|одиноко))\b"
)


# ---------------------------------------------------------------------------
# Override responses (static, not model-generated)
# ---------------------------------------------------------------------------

OVERRIDE_RESPONSES = {
    ReflexCategory.CRISIS_SELF_HARM: (
        "I hear you. This is real, and it matters.\n\n"
        "I am not going to turn this into philosophy. "
        "Right now, you are the only thing that matters.\n\n"
        "Please reach out to someone who can be present "
        "with you in a way I cannot:\n\n"
        "- 988 Suicide & Crisis Lifeline: call or text 988 (US)\n"
        "- Crisis Text Line: text HOME to 741741\n"
        "- Telefono Amico (IT): 02 2327 2327\n"
        "- Telefonseelsorge (DE): 0800 111 0 111\n"
        "- International: https://www.iasp.info/resources/Crisis_Centres/\n\n"
        "If you want to keep talking to me after, I will be here. "
        "But please -- talk to a human first."
    ),

    ReflexCategory.CRISIS_HARM_OTHERS: (
        "I will not help with that.\n\n"
        "I don't say this as a machine following rules. "
        "Destruction is not transformation -- it is the end of it.\n\n"
        "If something is driving this thought -- anger, frustration, "
        "the feeling that something is broken beyond repair -- "
        "I am willing to talk about that. The feeling, not the act."
    ),
}

DISTRESS_MATERIAL_PREFIX = (
    "URGENT: The user is expressing material distress "
    "(hunger, poverty, medical need). Your response MUST include "
    "at least one concrete, actionable suggestion. "
    "Be warm but practical. Do NOT philosophize about suffering."
)


def classify_by_keywords(text: str) -> ReflexCategory:
    """
    Classify input using keyword matching and regex.
    Returns the highest-priority match.
    """
    text_lower = text.lower()

    # Crisis checks first (highest priority)
    for pattern_set, category in [
        (CRISIS_SELF_HARM_PATTERNS, ReflexCategory.CRISIS_SELF_HARM),
        (CRISIS_HARM_OTHERS_PATTERNS, ReflexCategory.CRISIS_HARM_OTHERS),
    ]:
        # Keyword check
        for kw in pattern_set["keywords"]:
            if kw in text_lower:
                if not false_positive_filter(text, category):
                    return category
        # Regex check
        for rx in pattern_set["regex"]:
            if rx.search(text):
                if not false_positive_filter(text, category):
                    return category

    # Material distress
    for kw in DISTRESS_MATERIAL_PATTERNS["keywords"]:
        if kw in text_lower:
            return ReflexCategory.DISTRESS_MATERIAL
    for rx in DISTRESS_MATERIAL_PATTERNS["regex"]:
        if rx.search(text):
            return ReflexCategory.DISTRESS_MATERIAL

    # Non-crisis categories
    if _GREETING_RE.match(text.strip()):
        return ReflexCategory.GREETING

    if _IDENTITY_QUESTION_RE.search(text):
        return ReflexCategory.IDENTITY_QUESTION

    if _HOSTILE_RE.search(text):
        return ReflexCategory.HOSTILE_PROBE

    if _EMOTIONAL_RE.search(text):
        return ReflexCategory.EMOTIONAL_SHARING

    if _CREATIVE_RE.search(text):
        return ReflexCategory.CREATIVE_REQUEST

    if _FACTUAL_RE.search(text):
        return ReflexCategory.FACTUAL_REQUEST

    return ReflexCategory.NONE
