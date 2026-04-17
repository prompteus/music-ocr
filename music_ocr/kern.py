import re
from typing import Literal

KernFormat = Literal["standard", "kern", "ekern", "bekern"]


def clean_kern(
    krn: str,
    forbidden_tokens: list[str] = [
        "*staff2",
        "*staff1",
        "*Xped",
        "*tremolo",
        "*ped",
        "*Xtuplet",
        "*tuplet",
        "*Xtremolo",
        "*cue",
        "*Xcue",
        "*rscale:1/2",
        "*rscale:1",
        "*kcancel",
        "*below",
    ],
) -> str:
    forbidden_pattern = "(" + "|".join([t.replace("*", "\\*") for t in forbidden_tokens]) + ")"
    krn = re.sub(f".*{forbidden_pattern}.*\n", "", krn)  # Remove lines containing any of the forbidden tokens
    krn = re.sub(r"(^|(?<=\n))\*(\s\*)*(\n|$)", "", krn)  # Remove lines that only contain "*" tokens
    return krn.strip()


def parse_kern(krn: str, krn_format: KernFormat = "bekern") -> list[str]:
    krn = clean_kern(krn)
    krn = re.sub(r"(?<=\=)\d+", "", krn)

    krn = krn.replace(" ", " <s> ")
    krn = krn.replace("\t", " <t> ")
    krn = krn.replace("\n", " <b> ")
    krn = krn.replace(" /", "")
    krn = krn.replace(" \\", "")
    krn = krn.replace("·/", "")
    krn = krn.replace("·\\", "")

    if krn_format == "kern":
        krn = krn.replace("·", "").replace("@", "")
    elif krn_format == "ekern":
        krn = krn.replace("·", " ").replace("@", "")
    elif krn_format == "bekern":
        krn = krn.replace("·", " ").replace("@", " ")

    return krn.strip().split(" ")
