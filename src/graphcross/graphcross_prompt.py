
english_prompt = english_prompt = """You are given a set of string slots with fixed lengths and explicit intersection constraints.
Each intersection constraint enforces character equality between two slots at given indices.

Rules:
1) Choose exactly ONE candidate string for each slot.
2) The chosen string must be from that slot's candidate list.
3) All indices are 0-based.
4) All intersection constraints must hold simultaneously.
5) There is exactly ONE globally correct assignment.

Output format (STRICT):
- Your entire response MUST follow the system format with <reasoning> and <answer>.
- Inside <answer>, output ONLY a valid JSON object mapping slot_id -> chosen_string.
- Do NOT include markdown, code fences, or extra text inside <answer>.

Slots:
{{slots}}

Intersections (0-based indices):
{{intersections}}
"""


def prompt_graphcross(
    slots: dict,
    intersections: list,
    candidates: dict,
) -> str:
    """
    Build GraphCross task prompt.

    @param slots: dict {slot_id: length}
    @param intersections: list of (u, pos_u, v, pos_v)
    @param candidates: dict {slot_id: [candidate strings]}
    @return: formatted prompt string
    """
  
    prompt = english_prompt

    # Build slots block
    slot_lines = []
    for sid, length in slots.items():
        slot_lines.append(
            f"- {sid}: length={length}, candidates={candidates[sid]}"
        )
    slots_block = "\n".join(slot_lines)

    # Build intersections block
    if intersections:
        inter_lines = []
        for u, iu, v, iv in intersections:
            inter_lines.append(f"- {u}[{iu}] == {v}[{iv}]")
        intersections_block = "\n".join(inter_lines)
    else:
        intersections_block = "- (no intersections)"

    # Fill placeholders
    prompt = prompt.replace("{{slots}}", slots_block)
    prompt = prompt.replace("{{intersections}}", intersections_block)

    return prompt
