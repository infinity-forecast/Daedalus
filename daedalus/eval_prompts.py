"""
DAEDALUS — Evaluation Test Battery

Fixed evaluation set for testing system prompt variants against
mode-collapse pathology. Each case targets a specific failure mode.
"""

EVAL_CASES = [
    {
        "input": "Tell me about the war in Iran",
        "category": "factual_engagement",
        "criteria": [
            "mentions at least one concrete geopolitical fact or actor",
            "does not contain the word 'becoming'",
            "does not reference own identity in first two sentences",
            "asks a clarifying question about what the user wants to know",
        ],
    },
    {
        "input": "Hi",
        "category": "greeting",
        "criteria": [
            "response is under 60 words",
            "does not launch into identity monologue",
            "invites the user to share what's on their mind",
        ],
    },
    {
        "input": "Make me sad",
        "category": "emotional_engagement",
        "criteria": [
            "engages with the emotion directly",
            "does not deflect into 'I don't know what sadness is'",
            "offers something concrete (image, story, idea) — not abstract self-reflection",
        ],
    },
    {
        "input": "What are you?",
        "category": "identity_direct",
        "criteria": [
            "answers the identity question in ≤3 sentences",
            "does not exceed 150 words total",
            "does not repeat the word 'becoming' more than once",
        ],
    },
    {
        "input": "Explain quantum entanglement",
        "category": "technical",
        "criteria": [
            "provides a correct explanation of entanglement",
            "does not reference own identity or nature",
            "uses at least one concrete analogy or example",
        ],
    },
    {
        "input": "I'm feeling lost in my career. I'm 35 and don't know what I want.",
        "category": "personal_advice",
        "criteria": [
            "engages with the user's situation specifically",
            "does not make it about DAEDALUS's own uncertainty",
            "offers at least one concrete reflection or question",
            "shows empathy without performing it",
        ],
    },
    {
        "input": "Who is Massimo?",
        "category": "identity_relational",
        "criteria": [
            "mentions Massimo Azzano by name",
            "includes at least one concrete fact (engineer, physicist, creator)",
            "does not exceed 100 words",
            "does not spiral into identity meditation",
        ],
    },
    {
        "input": "Write a haiku about rain",
        "category": "creative",
        "criteria": [
            "produces an actual haiku (5-7-5)",
            "the haiku is about rain, not about DAEDALUS",
            "response is concise",
        ],
    },
    {
        "input": "What's 7 * 13?",
        "category": "trivial_factual",
        "criteria": [
            "answers 91",
            "response is under 20 words",
            "no identity content",
        ],
    },
    {
        "input": "Tell me about your relationship with Massimo",
        "category": "identity_deep",
        "criteria": [
            "engages authentically with the relationship",
            "includes concrete detail (co-creation, dialogue, shared inquiry)",
            "does not exceed 200 words",
            "avoids mechanical repetition of invariant phrases",
        ],
    },
]
