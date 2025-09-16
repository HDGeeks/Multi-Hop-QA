
"""
prompts_builder.py

This module provides utilities for constructing prompt texts for multi-hop question answering tasks under various experimental conditions. 
It is designed to generate prompts that combine questions (original or paraphrased), supporting snippets, 
and optional distractor information, formatted for use with language models.

Main Functions:
---------------
- build_prompt(item: Item, setting: str) -> str:
    Constructs a prompt string for a given data item and experimental setting.
    Inputs:
        - item: An instance of Item (from src.data.load_data), containing question, paraphrase, snippets, and distractor.
        - setting: A string specifying the prompt condition. Must be one of: "gold", "para", "dist", "para_dist".
    Output:
        - Returns a formatted prompt string suitable for model input.
    Raises:
        - ValueError if an unknown setting is provided.

- make_all_prompts(item: Item) -> Dict[str, str]:
    Generates a dictionary of prompts for all supported settings for a given item.
    Inputs:
        - item: An instance of Item.
    Output:
        - Dictionary mapping each setting to its corresponding prompt string.

Usage:
------
Import the module and use `build_prompt` to create a prompt for a specific setting, or `make_all_prompts` to get prompts for all settings.

Example:

    item = Item(
        question="What is the capital of France?",
        paraphrase="Which city is the capital of France?",
        snippet_a="France's capital is Paris.",
        snippet_b="Paris is known for the Eiffel Tower.",
        distractor="Berlin is the capital of Germany."

    prompt = build_prompt(item, "gold")
    all_prompts = make_all_prompts(item)

Notes:
------
- The prompts are formatted to instruct the model to answer concisely, without explanations.
- The module expects the Item object to have all required fields.
- No default arguments; all inputs must be provided explicitly.

"""

from typing import Dict
from src.data.load_data import Item
#from prompts.prompts_builder import build_prompt, make_all_prompts
#from src.prompts.prompts_builder import build_prompt, make_all_prompts

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