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
    body = "@".join(body_parts) if body_parts else ""
    mods = "·".join(mod_parts) if mod_parts else ""

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
    i = 0
    while i < len(s):
        ch = s[i]
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

    i = 0
    n = len(token)
    if n == 0:
        return token

    parts_body: list[str] = []

    # Duration digits
    duration = ""
    while i < n and token[i].isdigit():
        duration += token[i]
        i += 1

    if duration:
        parts_body.append(duration)

    # Dots
    dots = ""
    while i < n and token[i] == ".":
        dots += token[i]
        i += 1
    if dots:
        parts_body.append(dots)

    # Rest
    if i < n and token[i] == "r":
        parts_body.append("r")
        i += 1
        remaining = token[i:]
        mod_list = _parse_modifier_chars(remaining) if remaining else []
        return _join_annotated(parts_body, mod_list)

    # Pitch letters
    pitch = ""
    while i < n and token[i].isalpha() and token[i].lower() in "abcdefg":
        pitch += token[i]
        i += 1

    if pitch:
        parts_body.append(pitch)

    # Accidentals
    accidental = ""
    while i < n and token[i] in "#-n":
        accidental += token[i]
        i += 1
    if accidental:
        parts_body.append(accidental)

    # Everything remaining is modifiers
    remaining = token[i:]
    mod_list = _parse_modifier_chars(remaining) if remaining else []

    return _join_annotated(parts_body, mod_list)


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
    spines = line.split("\t")
    converted_spines: list[str] = []

    for spine in spines:
        # Within a spine, tokens can be space-separated (chords)
        tokens = spine.split(" ")
        converted_tokens = [kern_token_to_annotated_full(t) if t else t for t in tokens]
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
    lines = kern_text.split("\n")
    converted_lines: list[str] = []

    for line in lines:
        # Convert the exclusive interpretation headers
        if line.strip().startswith("**kern"):
            converted_lines.append(line.replace("**kern", "**ekern"))
        else:
            converted_lines.append(convert_kern_line_to_annotated(line))

    return "\n".join(converted_lines)


# ──────────────────── Tests ────────────────────

if __name__ == "__main__":
    # ── Individual token conversion tests ──

    # Simple notes: duration@pitch
    assert kern_token_to_annotated_full("4c") == "4@c", f"Expected '4@c', got '{kern_token_to_annotated_full('4c')}'"
    assert kern_token_to_annotated_full("8E") == "8@E", f"Expected '8@E', got '{kern_token_to_annotated_full('8E')}'"
    assert kern_token_to_annotated_full("16dd") == "16@dd", (
        f"Expected '16@dd', got '{kern_token_to_annotated_full('16dd')}'"
    )
    assert kern_token_to_annotated_full("2GG") == "2@GG", (
        f"Expected '2@GG', got '{kern_token_to_annotated_full('2GG')}'"
    )

    # Notes with accidentals: duration@pitch@accidental
    assert kern_token_to_annotated_full("4F#") == "4@F@#", (
        f"Expected '4@F@#', got '{kern_token_to_annotated_full('4F#')}'"
    )
    assert kern_token_to_annotated_full("8e-") == "8@e@-", (
        f"Expected '8@e@-', got '{kern_token_to_annotated_full('8e-')}'"
    )
    assert kern_token_to_annotated_full("16GG#") == "16@GG@#", (
        f"Expected '16@GG@#', got '{kern_token_to_annotated_full('16GG#')}'"
    )
    assert kern_token_to_annotated_full("8bb-") == "8@bb@-", (
        f"Expected '8@bb@-', got '{kern_token_to_annotated_full('8bb-')}'"
    )

    # Double accidentals
    assert kern_token_to_annotated_full("4c##") == "4@c@##", (
        f"Expected '4@c@##', got '{kern_token_to_annotated_full('4c##')}'"
    )
    assert kern_token_to_annotated_full("8f--") == "8@f@--", (
        f"Expected '8@f@--', got '{kern_token_to_annotated_full('8f--')}'"
    )

    # Dotted notes: duration@.@pitch
    assert kern_token_to_annotated_full("4.c") == "4@.@c", (
        f"Expected '4@.@c', got '{kern_token_to_annotated_full('4.c')}'"
    )
    assert kern_token_to_annotated_full("2.BB-") == "2@.@BB@-", (
        f"Expected '2@.@BB@-', got '{kern_token_to_annotated_full('2.BB-')}'"
    )
    assert kern_token_to_annotated_full("8.dd#") == "8@.@dd@#", (
        f"Expected '8@.@dd@#', got '{kern_token_to_annotated_full('8.dd#')}'"
    )

    # Notes with beaming modifiers: ·L, ·J
    assert kern_token_to_annotated_full("8eL") == "8@e·L", (
        f"Expected '8@e·L', got '{kern_token_to_annotated_full('8eL')}'"
    )
    assert kern_token_to_annotated_full("8e-J") == "8@e@-·J", (
        f"Expected '8@e@-·J', got '{kern_token_to_annotated_full('8e-J')}'"
    )
    assert kern_token_to_annotated_full("16f#L") == "16@f@#·L", (
        f"Expected '16@f@#·L', got '{kern_token_to_annotated_full('16f#L')}'"
    )

    # Multiple modifiers: ·J·L, ·L·], etc.
    assert kern_token_to_annotated_full("8e-JL") == "8@e@-·J·L", (
        f"Expected '8@e@-·J·L', got '{kern_token_to_annotated_full('8e-JL')}'"
    )
    assert kern_token_to_annotated_full("16B-L]") == "16@B@-·L·]", (
        f"Expected '16@B@-·L·]', got '{kern_token_to_annotated_full('16B-L]')}'"
    )

    # Tie markers
    assert kern_token_to_annotated_full("4c[") == "4@c·[", (
        f"Expected '4@c·[', got '{kern_token_to_annotated_full('4c[')}'"
    )
    assert kern_token_to_annotated_full("4c]") == "4@c·]", (
        f"Expected '4@c·]', got '{kern_token_to_annotated_full('4c]')}'"
    )
    assert kern_token_to_annotated_full("2.C[") == "2@.@C·[", (
        f"Expected '2@.@C·[', got '{kern_token_to_annotated_full('2.C[')}'"
    )

    # Rests
    assert kern_token_to_annotated_full("8r") == "8@r", f"Expected '8@r', got '{kern_token_to_annotated_full('8r')}'"
    assert kern_token_to_annotated_full("4r") == "4@r", f"Expected '4@r', got '{kern_token_to_annotated_full('4r')}'"
    assert kern_token_to_annotated_full("2.r") == "2@.@r", (
        f"Expected '2@.@r', got '{kern_token_to_annotated_full('2.r')}'"
    )

    # Grace notes (no duration, just pitch + q modifier)
    assert kern_token_to_annotated_full("eq") == "e·q", f"Expected 'e·q', got '{kern_token_to_annotated_full('eq')}'"
    assert kern_token_to_annotated_full("e-q") == "e@-·q", (
        f"Expected 'e@-·q', got '{kern_token_to_annotated_full('e-q')}'"
    )

    # Structural tokens pass through unchanged
    assert kern_token_to_annotated_full("*clefG2") == "*clefG2"
    assert kern_token_to_annotated_full("*clefF4") == "*clefF4"
    assert kern_token_to_annotated_full("*k[f#c#]") == "*k[f#c#]"
    assert kern_token_to_annotated_full("*M3/4") == "*M3/4"
    assert kern_token_to_annotated_full("*-") == "*-"
    assert kern_token_to_annotated_full("*met(C)") == "*met(C)"
    assert kern_token_to_annotated_full("=") == "="
    assert kern_token_to_annotated_full("=15") == "=15"
    assert kern_token_to_annotated_full(".") == "."

    # Editorial cautionary mark X (appears after accidental, treated as modifier)
    assert kern_token_to_annotated_full("4f-X[") == "4@f@-·X·[", (
        f"Expected '4@f@-·X·[', got '{kern_token_to_annotated_full('4f-X[')}'"
    )
    assert kern_token_to_annotated_full("4B-X[") == "4@B@-·X·[", (
        f"Expected '4@B@-·X·[', got '{kern_token_to_annotated_full('4B-X[')}'"
    )

    # Partial beam (K modifier)
    assert kern_token_to_annotated_full("16ddKL") == "16@dd·K·L", (
        f"Expected '16@dd·K·L', got '{kern_token_to_annotated_full('16ddKL')}'"
    )

    # Staccato (k modifier) after beam
    assert kern_token_to_annotated_full("16dd#Jk") == "16@dd@#·J·k", (
        f"Expected '16@dd@#·J·k', got '{kern_token_to_annotated_full('16dd#Jk')}'"
    )

    # ── Full line conversion tests ──

    line1 = "4c\t4E"
    assert convert_kern_line_to_annotated(line1) == "4@c\t4@E", (
        f"Line test 1 failed: '{convert_kern_line_to_annotated(line1)}'"
    )

    line2 = "8eL\t8g#J"
    assert convert_kern_line_to_annotated(line2) == "8@e·L\t8@g@#·J", (
        f"Line test 2 failed: '{convert_kern_line_to_annotated(line2)}'"
    )

    # Chord (space-separated within a spine)
    line3 = "4c 4e 4g\t4E 4G"
    assert convert_kern_line_to_annotated(line3) == "4@c 4@e 4@g\t4@E 4@G", (
        f"Line test 3 failed: '{convert_kern_line_to_annotated(line3)}'"
    )

    # ── Full score conversion test ──

    score = """**kern\t**kern
*clefG2\t*clefF4
*k[f#]\t*k[f#]
*M3/4\t*M3/4
=1\t=1
4c\t4E
4d\t4F#
4e\t4G
=2\t=2
2.g\t2.B
=3\t=3
*-\t*-"""

    expected = """**ekern\t**ekern
*clefG2\t*clefF4
*k[f#]\t*k[f#]
*M3/4\t*M3/4
=1\t=1
4@c\t4@E
4@d\t4@F@#
4@e\t4@G
=2\t=2
2@.@g\t2@.@B
=3\t=3
*-\t*-"""

    result = convert_kern_to_annotated(score)
    assert result == expected, f"Full score test failed.\nExpected:\n{expected}\n\nGot:\n{result}"

    # ── Round-trip test: annotated → bekern → check it matches manual bekern ──

    try:
        import sys
        import os

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils import parse_kern

        # Take a known annotated token sequence and verify parse_kern produces
        # the expected bekern output
        annotated = "4@F@#·L\t8@e@-·J"
        bekern_result = parse_kern(annotated, krn_format="bekern")
        # After whitespace replacement and @ → space, · → space:
        # "4 F # L <t> 8 e - J"
        expected_bekern = ["4", "F", "#", "L", "<t>", "8", "e", "-", "J"]
        assert bekern_result == expected_bekern, (
            f"Round-trip bekern test failed.\nExpected: {expected_bekern}\nGot: {bekern_result}"
        )

        ekern_result = parse_kern(annotated, krn_format="ekern")
        # After · → space, @ removed:
        # "4F# L <t> 8e- J"
        expected_ekern = ["4F#", "L", "<t>", "8e-", "J"]
        assert ekern_result == expected_ekern, (
            f"Round-trip ekern test failed.\nExpected: {expected_ekern}\nGot: {ekern_result}"
        )

        kern_result = parse_kern(annotated, krn_format="kern")
        # After · removed, @ removed:
        # "4F#L <t> 8e-J"
        expected_kern = ["4F#L", "<t>", "8e-J"]
        assert kern_result == expected_kern, (
            f"Round-trip kern test failed.\nExpected: {expected_kern}\nGot: {kern_result}"
        )

        print("Round-trip tests with parse_kern: PASSED")

    except ImportError as e:
        print(f"Skipping round-trip tests (missing dependency: {e})")

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)

    # ── Black-box tests: real tokens from PRAIG/grandstaff-ekern ──
    # Each pair is (standard_kern, expected_annotated) derived from the actual dataset.
    # The standard kern is obtained by stripping @ and · from the real dataset token.

    print("\n── Black-box tests: Real dataset tokens ──\n")

    real_dataset_pairs = [
        # ── Triplets / tuplets ──
        ("12AAL", "12@AA·L"),
        ("12AJ", "12@A·J"),
        ("12BB-", "12@BB@-"),
        ("12BB-L", "12@BB@-·L"),
        ("12BBL", "12@BB·L"),
        ("12C", "12@C"),
        ("12CL", "12@C·L"),
        ("12D", "12@D"),
        ("12DJ", "12@D·J"),
        ("12E-", "12@E@-"),
        ("12E-J", "12@E@-·J"),
        ("12FJ", "12@F·J"),
        ("12GGL", "12@GG·L"),
        ("12GJ", "12@G·J"),
        ("12a-J", "12@a@-·J"),
        ("12a-L", "12@a@-·L"),
        ("12b-", "12@b@-"),
        # ── Dotted sixteenths ──
        ("16.aaJ", "16@.@aa·J"),
        ("16.cc#J", "16@.@cc@#·J"),
        ("16.ff#J", "16@.@ff@#·J"),
        # ── Sixteenths with staccato (·k) ──
        ("16B-Jk", "16@B@-·J·k"),
        ("16C-Jk", "16@C@-·J·k"),
        ("16E-Jk", "16@E@-·J·k"),
        ("16G-Jk", "16@G@-·J·k"),
        ("16GG-Jk", "16@GG@-·J·k"),
        ("16c-Jk", "16@c@-·J·k"),
        ("16cJk", "16@c·J·k"),
        ("16e-Jk", "16@e@-·J·k"),
        # ── Sixteenths: plain, beamed, accidentalled ──
        ("16BBJ", "16@BB·J"),
        ("16D", "16@D"),
        ("16E", "16@E"),
        ("16EJ", "16@E·J"),
        ("16F", "16@F"),
        ("16GG#", "16@GG@#"),
        ("16GG#L", "16@GG@#·L"),
        ("16aaJ", "16@aa·J"),
        ("16aJ", "16@a·J"),
        ("16b#L", "16@b@#·L"),
        ("16bb", "16@bb"),
        ("16bb#L", "16@bb@#·L"),
        ("16bJ", "16@b·J"),
        ("16cc#J", "16@cc@#·J"),
        ("16ccc#J", "16@ccc@#·J"),
        ("16d#L", "16@d@#·L"),
        ("16dd", "16@dd"),
        ("16dd#L", "16@dd@#·L"),
        ("16ddd", "16@ddd"),
        ("16ddd#L", "16@ddd@#·L"),
        ("16ddL", "16@dd·L"),
        ("16dL", "16@d·L"),
        ("16eeeJ", "16@eee·J"),
        ("16eeJ", "16@ee·J"),
        ("16eJ", "16@e·J"),
        ("16g#", "16@g@#"),
        ("16g#L", "16@g@#·L"),
        ("16gg#", "16@gg@#"),
        ("16gg#J", "16@gg@#·J"),
        ("16gg#L", "16@gg@#·L"),
        ("16ggL", "16@gg·L"),
        ("16r", "16@r"),
        # ── 24ths (sextuplets) ──
        ("24A", "24@A"),
        ("24AAL", "24@AA·L"),
        ("24AJ", "24@A·J"),
        ("24BJ", "24@B·J"),
        ("24C", "24@C"),
        ("24C#L", "24@C@#·L"),
        ("24E", "24@E"),
        ("24EJ", "24@E·J"),
        ("24EL", "24@E·L"),
        ("24G#", "24@G@#"),
        ("24c#J", "24@c@#·J"),
        # ── Dotted halves with ties ──
        ("2.A", "2@.@A"),
        ("2.AA", "2@.@AA"),
        ("2.C", "2@.@C"),
        ("2.C-", "2@.@C@-"),
        ("2.C[", "2@.@C·["),
        ("2.E[", "2@.@E·["),
        ("2.F", "2@.@F"),
        ("2.G[", "2@.@G·["),
        ("2.b-[", "2@.@b@-·["),
    ]

    failures = 0
    for kern_input, expected_output in real_dataset_pairs:
        actual = kern_token_to_annotated_full(kern_input)
        if actual != expected_output:
            print(f"  FAIL: '{kern_input}' → '{actual}' (expected '{expected_output}')")
            failures += 1

    if failures == 0:
        print(f"  All {len(real_dataset_pairs)} black-box tests PASSED")
    else:
        print(f"\n  {failures}/{len(real_dataset_pairs)} tests FAILED")
        raise AssertionError(f"{failures} black-box tests failed")

    # ── Demo: show conversion of a sample score ──
    print("\n── Demo: Standard **kern → PRAIG-annotated format ──\n")

    demo_score = """**kern\t**kern
*clefG2\t*clefF4
*k[b-]\t*k[b-]
*M3/4\t*M3/4
=1\t=1
8eL\t4c 4e 4g
8d\t.
8f#J\t4A
=2\t=2
4.g[\t2.B-
=3\t=3
4g]\t4r
8aL\t8r
8bJ\t8G
*-\t*-"""

    print("INPUT (standard **kern):")
    print(demo_score)
    print("\nOUTPUT (PRAIG-annotated):")
    print(convert_kern_to_annotated(demo_score))
