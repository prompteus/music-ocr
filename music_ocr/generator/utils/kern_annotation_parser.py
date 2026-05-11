"""
Convert standard **kern tokens into the PRAIG-annotated format with @ and · delimiters.

The PRAIG group (University of Alicante) published their OMR datasets in an annotated
variant of **kern where two delimiter characters mark sub-token boundaries:

    @  separates components within the note body:
       duration @ dotting @ pitch @ accidental
       Example: 8e-  →  8@e@-

    ·  separates the note body from graphical/performance modifiers:
       beaming (L, J, LL, JJ, K), ties ([, ], _), grace notes (q, qq),
       staccato (k), editorial marks (X, <, >), etc.
       Example: 8e-JL  →  8@e@-·J·L

This module provides a function to perform this conversion, enabling the creation
of new datasets compatible with the PRAIG tokenisation pipeline (kern/ekern/bekern).
"""


def _join_annotated(body_parts: list[str], mod_parts: list[str]) -> str:
    """
    Join parsed note body parts and modifiers into the final annotated string.

    Args:
        body_parts: Components of the note body.
        mod_parts: Components of the graphical modifiers.

    Returns:
        The fully annotated string joined by @ and ·.
    """
    body: str = "@".join(body_parts) if body_parts else ""
    mods: str = "·".join(mod_parts) if mod_parts else ""

    if body and mods:
        return body + "·" + mods
    elif body:
        return body
    elif mods:
        return "·".join(mod_parts)
    else:
        return ""


def _parse_modifier_chars(s: str) -> list[str]:
    """
    Parse the modifier string into individual graphical modifier groups.

    In standard **kern, modifiers are concatenated: 'JL' means beam-end then beam-start.
    Each modifier character or known multi-char sequence becomes its own group.

    Args:
        s: Raw modifier string portion.

    Returns:
        List of separated modifier characters/blobs.
    """
    mods: list[str] = []
    i: int = 0
    while i < len(s):
        ch: str = s[i]
        # Multi-character modifiers: LL, JJ, qq
        if i + 1 < len(s) and s[i] == s[i + 1] and s[i] in "LJq":
            mods.append(s[i : i + 2])
            i += 2
        else:
            mods.append(ch)
            i += 1
    return mods


def kern_token_to_annotated_full(token: str) -> str:
    """
    Convert a standard **kern token, splitting modifier characters individually.

    In the PRAIG datasets, each modifier is separated by ·:
        8e-JL  →  8@e@-·J·L  (not 8@e@-·JL)

    This function first extracts the body, then splits the modifier portion
    into individual modifiers.
    """
    # Structural tokens: pass through
    if token.startswith("*") or token.startswith("=") or token == ".":
        return token

    i: int = 0
    n: int = len(token)
    if n == 0:
        return token

    parts_body: list[str] = []

    # Duration digits
    duration: str = ""
    while i < n and token[i].isdigit():
        duration += token[i]
        i += 1

    if duration:
        parts_body.append(duration)

    # Dots
    dots: str = ""
    while i < n and token[i] == ".":
        dots += token[i]
        i += 1
    if dots:
        parts_body.append(dots)

    # Rest
    if i < n and token[i] == "r":
        parts_body.append("r")
        i += 1
        remaining: str = token[i:]
        mod_list: list[str] = _parse_modifier_chars(remaining) if remaining else []
        return _join_annotated(parts_body, mod_list)

    # Pitch letters
    pitch: str = ""
    while i < n and token[i].isalpha() and token[i].lower() in "abcdefg":
        pitch += token[i]
        i += 1

    if pitch:
        parts_body.append(pitch)

    # Accidentals
    accidental: str = ""
    while i < n and token[i] in "#-n":
        accidental += token[i]
        i += 1
    if accidental:
        parts_body.append(accidental)

    # Everything remaining is modifiers
    remaining_mods: str = token[i:]
    mod_list_pitch: list[str] = _parse_modifier_chars(remaining_mods) if remaining_mods else []

    return _join_annotated(parts_body, mod_list_pitch)


def convert_kern_line_to_annotated(line: str) -> str:
    """
    Convert a full line of standard **kern into the PRAIG-annotated format.

    A line may contain multiple tokens separated by spaces (chords)
    or tabs (different spines/staves).

    Handles:
    - Tab-separated spines
    - Space-separated chord notes
    - Structural tokens (pass through)
    - Empty tokens (`.` for null data)
    """
    # Split by tabs first (spine boundaries)
    spines: list[str] = line.split("\t")
    converted_spines: list[str] = []

    for spine in spines:
        # Within a spine, tokens can be space-separated (chords)
        tokens: list[str] = spine.split(" ")
        converted_tokens: list[str] = [kern_token_to_annotated_full(t) if t else t for t in tokens]
        converted_spines.append(" ".join(converted_tokens))

    return "\t".join(converted_spines)


def convert_kern_to_annotated(kern_text: str) -> str:
    """
    Convert an entire **kern score into the PRAIG-annotated format.

    Args:
        kern_text: Full **kern file content as a string

    Returns:
        The same score with @ and · delimiters inserted
    """
    lines: list[str] = kern_text.split("\n")
    converted_lines: list[str] = []

    for line in lines:
        # Convert the exclusive interpretation headers
        if line.strip().startswith("**kern"):
            converted_lines.append(line.replace("**kern", "**ekern"))
        else:
            converted_lines.append(convert_kern_line_to_annotated(line))

    return "\n".join(converted_lines)


# ──────────────────── Tests ────────────────────
