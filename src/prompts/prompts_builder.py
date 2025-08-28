# src/prompts/prompt_builder.py
"""
This module provides utilities for constructing prompt texts for multi-hop question answering tasks
under different experimental conditions. It defines the possible settings (gold, para, dist, para_dist)
and builds prompts that combine questions (original or paraphrased), relevant snippets, and optional distractors.
The prompts are formatted to instruct models to answer concisely without explanations.

Functions:
    build_prompt(item: Item, setting: str) -> str:
        Constructs a prompt for a given item and condition, including the appropriate question,
        snippets, and distractor if required.

    make_all_prompts(item: Item) -> Dict[str, str]:
        Generates prompts for all four settings for a given item, returning a dictionary mapping
        each setting to its corresponding prompt text.
"""

from typing import Dict
from src.data.load_data import Item

# condition keys (shared everywhere)
GOLD = "gold"
PARA = "para"
DIST = "dist"
PARA_DIST = "para_dist"
ALL_SETTINGS = (GOLD, PARA, DIST, PARA_DIST)


def build_prompt(item: Item, setting: str) -> str:
    """
    Build the exact prompt text for one item under one condition.
    Conditions:
      - gold: gold question + 2 gold snippets
      - para: paraphrased question + 2 gold snippets
      - dist: gold question + 2 gold snippets + distractor
      - para_dist: paraphrased question + 2 gold snippets + distractor
    """
    if setting not in ALL_SETTINGS:
        raise ValueError(f"Unknown setting {setting}, expected one of {ALL_SETTINGS}")

    # decide which question to show
    question_text = (
        item.paraphrase if setting in (PARA, PARA_DIST) and item.paraphrase else item.question
    )

    # base prompt
    blocks = [
        "Use the snippets to answer the question.",
        "Rules: Output only the final answer as a short span. No explanations.",
        "",
        f"Snippet A:\n{item.snippet_a.strip()}",
        f"Snippet B:\n{item.snippet_b.strip()}",
    ]

    # add distractor if needed
    if setting in (DIST, PARA_DIST):
        blocks.append(f"Distractor Snippet:\n{item.distractor.strip()}")

    # add the question and answer anchor
    blocks.extend([
        "",
        f"Question:\n{question_text.strip()}",
        "",
        "Answer:"  # <- the model completes here
    ])

    return "\n".join(blocks)


def make_all_prompts(item: Item) -> Dict[str, str]:
    """
    Return a dict of {setting: prompt_text} for all four settings of one item.
    """
    return {s: build_prompt(item, s) for s in ALL_SETTINGS}