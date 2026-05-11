"""
Tests for music_ocr.generator.utils.kern_annotation_parser.

Mirrors: music_ocr/generator/utils/kern_annotation_parser.py
"""

import pytest
from music_ocr.generator.utils.kern_annotation_parser import (
    convert_kern_line_to_annotated,
    convert_kern_to_annotated,
    kern_token_to_annotated_full,
)


# ── Individual token conversion ──────────────────────────────────────────────


def test_simple_notes():
    assert kern_token_to_annotated_full("4c") == "4@c"
    assert kern_token_to_annotated_full("8E") == "8@E"
    assert kern_token_to_annotated_full("16dd") == "16@dd"
    assert kern_token_to_annotated_full("2GG") == "2@GG"


def test_notes_with_accidentals():
    assert kern_token_to_annotated_full("4F#") == "4@F@#"
    assert kern_token_to_annotated_full("8e-") == "8@e@-"
    assert kern_token_to_annotated_full("16GG#") == "16@GG@#"
    assert kern_token_to_annotated_full("8bb-") == "8@bb@-"


def test_double_accidentals():
    assert kern_token_to_annotated_full("4c##") == "4@c@##"
    assert kern_token_to_annotated_full("8f--") == "8@f@--"


def test_dotted_notes():
    assert kern_token_to_annotated_full("4.c") == "4@.@c"
    assert kern_token_to_annotated_full("2.BB-") == "2@.@BB@-"
    assert kern_token_to_annotated_full("8.dd#") == "8@.@dd@#"


def test_beaming_modifiers():
    assert kern_token_to_annotated_full("8eL") == "8@e·L"
    assert kern_token_to_annotated_full("8e-J") == "8@e@-·J"
    assert kern_token_to_annotated_full("16f#L") == "16@f@#·L"


def test_multiple_modifiers():
    assert kern_token_to_annotated_full("8e-JL") == "8@e@-·J·L"
    assert kern_token_to_annotated_full("16B-L]") == "16@B@-·L·]"


def test_tie_markers():
    assert kern_token_to_annotated_full("4c[") == "4@c·["
    assert kern_token_to_annotated_full("4c]") == "4@c·]"
    assert kern_token_to_annotated_full("2.C[") == "2@.@C·["


def test_rests():
    assert kern_token_to_annotated_full("8r") == "8@r"
    assert kern_token_to_annotated_full("4r") == "4@r"
    assert kern_token_to_annotated_full("2.r") == "2@.@r"


def test_grace_notes():
    assert kern_token_to_annotated_full("eq") == "e·q"
    assert kern_token_to_annotated_full("e-q") == "e@-·q"


def test_structural_tokens_pass_through():
    assert kern_token_to_annotated_full("*clefG2") == "*clefG2"
    assert kern_token_to_annotated_full("*clefF4") == "*clefF4"
    assert kern_token_to_annotated_full("*k[f#c#]") == "*k[f#c#]"
    assert kern_token_to_annotated_full("*M3/4") == "*M3/4"
    assert kern_token_to_annotated_full("*-") == "*-"
    assert kern_token_to_annotated_full("*met(C)") == "*met(C)"
    assert kern_token_to_annotated_full("=") == "="
    assert kern_token_to_annotated_full("=15") == "=15"
    assert kern_token_to_annotated_full(".") == "."


def test_editorial_cautionary_mark():
    assert kern_token_to_annotated_full("4f-X[") == "4@f@-·X·["
    assert kern_token_to_annotated_full("4B-X[") == "4@B@-·X·["


def test_partial_beam():
    assert kern_token_to_annotated_full("16ddKL") == "16@dd·K·L"


def test_staccato_after_beam():
    assert kern_token_to_annotated_full("16dd#Jk") == "16@dd@#·J·k"


# ── Full line conversion ──────────────────────────────────────────────────────


def test_convert_line_simple():
    assert convert_kern_line_to_annotated("4c\t4E") == "4@c\t4@E"


def test_convert_line_beamed():
    assert convert_kern_line_to_annotated("8eL\t8g#J") == "8@e·L\t8@g@#·J"


def test_convert_line_chord():
    assert convert_kern_line_to_annotated("4c 4e 4g\t4E 4G") == "4@c 4@e 4@g\t4@E 4@G"


# ── Full score conversion ─────────────────────────────────────────────────────


SCORE = """\
**kern\t**kern
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

EXPECTED_SCORE = """\
**ekern\t**ekern
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


def test_full_score_conversion():
    assert convert_kern_to_annotated(SCORE) == EXPECTED_SCORE


# ── Round-trip tests via parse_kern ──────────────────────────────────────────


@pytest.fixture
def annotated_sample():
    return "4@F@#·L\t8@e@-·J"


def test_roundtrip_bekern(annotated_sample):
    from music_ocr.kern import parse_kern

    result = parse_kern(annotated_sample, krn_format="bekern")
    assert result == ["4", "F", "#", "L", "<t>", "8", "e", "-", "J"]


def test_roundtrip_ekern(annotated_sample):
    from music_ocr.kern import parse_kern

    result = parse_kern(annotated_sample, krn_format="ekern")
    assert result == ["4F#", "L", "<t>", "8e-", "J"]


def test_roundtrip_kern(annotated_sample):
    from music_ocr.kern import parse_kern

    result = parse_kern(annotated_sample, krn_format="kern")
    assert result == ["4F#L", "<t>", "8e-J"]


# ── Black-box tests: real tokens from PRAIG/grandstaff-ekern ─────────────────

BLACK_BOX_PAIRS = [
    # Triplets / tuplets
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
    # Dotted sixteenths
    ("16.aaJ", "16@.@aa·J"),
    ("16.cc#J", "16@.@cc@#·J"),
    ("16.ff#J", "16@.@ff@#·J"),
    # Sixteenths with staccato
    ("16B-Jk", "16@B@-·J·k"),
    ("16C-Jk", "16@C@-·J·k"),
    ("16E-Jk", "16@E@-·J·k"),
    ("16G-Jk", "16@G@-·J·k"),
    ("16GG-Jk", "16@GG@-·J·k"),
    ("16c-Jk", "16@c@-·J·k"),
    ("16cJk", "16@c·J·k"),
    ("16e-Jk", "16@e@-·J·k"),
    # Sixteenths: plain, beamed, accidentalled
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
    # 24ths (sextuplets)
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
    # Dotted halves with ties
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


@pytest.mark.parametrize("kern_input,expected", BLACK_BOX_PAIRS)
def test_black_box_real_tokens(kern_input: str, expected: str):
    assert kern_token_to_annotated_full(kern_input) == expected
