from dataclasses import dataclass

@dataclass
class Item:
    qid: str
    domain: str
    question: str
    paraphrase: str | None
    snippet_a: str
    snippet_b: str
    distractor: str
    answer: str  # not shown to the model, just for bookkeeping

# condition keys we’ll use everywhere
GOLD = "gold"
PARA = "para"
DIST = "dist"
PARA_DIST = "para_dist"

def build_prompt(item: Item, setting: str) -> str:
    """
    Returns the exact prompt text for the given item and condition.
    Rules enforce: single short span answer, no explanations.
    """

    # pick which question text to show
    if setting in (PARA, PARA_DIST):
        q_text = item.paraphrase or item.question
    else:
        q_text = item.question

    # assemble context blocks
    blocks = [
        "Use the snippets to answer the question.",
        "Rules: Output only the final answer as a short span. No explanations."
    ]

    blocks.append("Snippet A:\n" + item.snippet_a.strip())
    blocks.append("Snippet B:\n" + item.snippet_b.strip())

    if setting in (DIST, PARA_DIST):
        blocks.append("Distractor Snippet:\n" + item.distractor.strip())

    blocks.append("Question:\n" + q_text.strip())
    blocks.append("Answer:")  # anchor the completion

    return "\n\n".join(blocks)