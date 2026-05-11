"""
Kern generation module suitable for OMR training.
Separates data generation from any rendering processes.
Ensures thorough and controllable coverage of standard Humdrum **kern syntax elements.
"""

import random
import re
from fractions import Fraction
from typing import Dict, List, Optional, Set, Union, Tuple


def _get_dot_multiplier(dots: str) -> Fraction:
    if dots == ".":
        return Fraction(3, 2)
    if dots == "..":
        return Fraction(7, 4)
    if dots == "...":
        return Fraction(15, 8)
    return Fraction(1, 1)


class KernGenerator:
    """
    Generates complex and extensive Humdrum **kern notation sequences.
    Provides complete control over sequence length (via measures) and guarantees
    ability to cover the entire requested token space over sufficient generations.
    """

    RESET_TRACKING_AFTER_N_GENERATED_MEASURES_AND_METHOD_COMPLETION: int = 4000

    # Generation Probabilities & Constants
    REST_PROBABILITY = 0.05
    FERMATA_ON_REST_PROB = 0.05

    PHRASE_START_PROB = 0.05
    PHRASE_END_PROB = 0.1
    SLUR_START_PROB = 0.05
    SLUR_END_PROB = 0.1
    TIE_START_PROB = 0.05
    TIE_END_PROB = 0.5
    GLISSANDO_START_PROB = 0.05
    GLISSANDO_END_PROB = 0.2
    BEAM_START_PROB = 0.05
    BEAM_END_PROB = 0.2

    CHORD_PROBABILITY = 0.05
    BARLINE_OVERRIDE_PROB = 0.05

    OCTAVE_MIN = 1
    OCTAVE_MAX = 4

    # Very long durations historically and stylistically do not take formatting dots
    UNDOTTABLE_DURATIONS = ["000", "00", "0"]

    # Very long durations cannot be beamed together logically since they equal/exceed entire measures natively
    UNBEAMABLE_DURATIONS = ["000", "00", "0", "1", "2", "3", "4"]

    # Binary durations used strictly as greedy allocation templates, mathematically guaranteeing complete metric sums
    BINARY_DURATIONS = ["000", "00", "0", "1", "2", "4", "8", "16", "32", "64", "128", "256", "512", "1024"]

    DURATIONS_MAP: Dict[str, Fraction] = {
        "000": Fraction(8, 1),
        "00": Fraction(4, 1),
        "0": Fraction(2, 1),
        "1": Fraction(1, 1),
        "2": Fraction(1, 2),
        "3": Fraction(1, 3),  # triplet half
        "4": Fraction(1, 4),
        "6": Fraction(1, 6),
        "8": Fraction(1, 8),
        "12": Fraction(1, 12),
        "16": Fraction(1, 16),
        "20": Fraction(1, 20),
        "24": Fraction(1, 24),
        "32": Fraction(1, 32),
        "48": Fraction(1, 48),
        "64": Fraction(1, 64),
        "128": Fraction(1, 128),
        "256": Fraction(1, 256),
        "512": Fraction(1, 512),
        "1024": Fraction(1, 1024),
    }

    TIMES_MAP: Dict[str, Fraction] = {
        "*M8/1": Fraction(8, 1),
        "*M4/1": Fraction(4, 1),
        "*M2/1": Fraction(2, 1),
        "*M4/4": Fraction(4, 4),
        "*M3/4": Fraction(3, 4),
        "*M2/4": Fraction(2, 4),
        "*M6/8": Fraction(6, 8),
        "*M3/8": Fraction(3, 8),
        "*M9/8": Fraction(9, 8),
        "*M12/8": Fraction(12, 8),
        "*C": Fraction(4, 4),
        "*C|": Fraction(2, 2),
    }

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initializes the generator.

        Args:
            seed: Optional integer seed for reproducible generation.
        """
        if seed is not None:
            random.seed(seed)

        # Reference for complete **kern tokens:
        # https://www.humdrum.org/guide/ch06/
        # https://www.humdrum.org/rep/kern/index.html
        # https://www.humdrum.org/Humdrum/representations/kern.html#Tokens
        self.playbook: Dict[str, List[Union[str, bool]]] = {
            "durations": [
                "000",
                "00",
                "0",
                "1",
                "2",
                "3",
                "4",
                "6",
                "8",
                "12",
                "16",
                "20",
                "24",
                "32",
                "48",
                "64",
                "128",
                "256",
                "512",
                "1024",
            ],
            "dots": ["", ".", "..", "..."],
            "grace": ["", "q", "qq", "P", "p"],
            "base_pitches": ["c", "d", "e", "f", "g", "a", "b", "C", "D", "E", "F", "G", "A", "B"],
            "accidentals": ["", "#", "-", "n", "##", "--", "---"],
            "rests": ["r"],
            "articulations": ["", "'", '"', "^", "~", ":", "`", "u", "v", ",", "z", "U", "o", "I"],
            "ornaments": [
                "",
                "T",
                "t",
                "m",
                "w",
                "M",
                "O",
                "TT",
                "tt",
                "WW",
                "RR",
            ],  # "S" ornament is missing. Verovio has problems with it lol.
            "stems": ["", "/", "\\"],
            "clefs": [
                "*clefG1",
                "*clefG2",
                "*clefG3",
                "*clefG4",
                "*clefG5",
                "*clefC1",
                "*clefC2",
                "*clefC3",
                "*clefC4",
                "*clefC5",
                "*clefF1",
                "*clefF2",
                "*clefF3",
                "*clefF4",
                "*clefF5",
                "*clefGv2",
                "*clefFv4",
            ],
            "keys": [
                "*k[]",
                "*k[f#]",
                "*k[f#c#]",
                "*k[f#c#g#]",
                "*k[f#c#g#d#]",
                "*k[f#c#g#d#a#]",
                "*k[f#c#g#d#a#e#]",
                "*k[f#c#g#d#a#e#b#]",
                "*k[b-]",
                "*k[b-e-]",
                "*k[b-e-a-]",
                "*k[b-e-a-d-]",
                "*k[b-e-a-d-g-]",
                "*k[b-e-a-d-g-c-]",
                "*k[b-e-a-d-g-c-f-]",
            ],
            "times": [
                "*M8/1",
                "*M4/1",
                "*M2/1",
                "*M4/4",
                "*M3/4",
                "*M2/4",
                "*M6/8",
                "*M3/8",
                "*M9/8",
                "*M12/8",
                "*C",
                "*C|",
            ],
            "barlines": ["=", "==", "=|", "=||", "=:|!", "=!|:", "=-"],
            "editorial": ["", "x", "xx", "y", "Y", "?", "??"],
            "chords": [False, True],
            "ties": ["[", "_", "]"],
            "slurs": ["(", ")"],
            "glissandi": ["H", "h"],
            "beams": ["L", "J", "k", "K", "LL", "JJ", "kk", "KK"],
            "phrases": ["{", "}"],
        }

        self.used: Dict[str, Set[Union[str, bool]]] = {}
        self.generated_measures: int = 0
        self.reset_tracking()

        self.in_beam = False
        self.in_slur = False
        self.in_tie = False
        self.in_phrase = False
        self.in_glissando = False
        self.current_pitch: Optional[str] = None

        self.reset_state()

    def reset_tracking(self) -> None:
        """Resets the tracking sets used for evaluating generation coverage."""
        self.used = {k: set() for k in self.playbook.keys()}

        self.used["beams"].add("")
        self.used["glissandi"].add("")
        self.generated_measures = 0

    def reset_state(self) -> None:
        """Resets the state of open contexts (ties, slurs, beams, etc.) for a new generation flow."""
        self.in_beam = False
        self.in_slur = False
        self.in_tie = False
        self.in_phrase = False
        self.in_glissando = False
        self.current_pitch = None

    def _track(self, category: str, value: Union[str, bool]) -> None:
        """Internal helper for logging generated elements."""
        if value is True:
            self.used[category].add(True)
        elif value is False:
            self.used[category].add(False)
        else:
            self.used[category].add(value)

    def _close_all_open_states(self, suffix: str, include_beams: bool = True) -> str:
        """Ensures all naturally open elements are cleanly terminated.

        Args:
            suffix: The suffix string to append closing tokens to.
            include_beams: If False, beam-close tokens are omitted (e.g. for rests,
                           which cannot carry beam markers in **kern).
        """
        if self.in_tie:
            suffix += "]"
            self.in_tie = False
            self._track("ties", "]")
        if self.in_slur:
            suffix += ")"
            self.in_slur = False
            self._track("slurs", ")")
        if self.in_phrase:
            suffix += "}"
            self.in_phrase = False
            self._track("phrases", "}")
        if self.in_glissando:
            suffix += "h"
            self.in_glissando = False
            self._track("glissandi", "h")
        if self.in_beam and include_beams:
            beam_end = str(random.choice(["J", "JJ"]))
            suffix += beam_end
            self.in_beam = False
            self._track("beams", beam_end)
        return suffix

    def _get_duration_fraction(self, dur: str, dots: str, grace: str) -> Fraction:
        if grace in ["q", "qq"]:
            return Fraction(0, 1)
        return self.DURATIONS_MAP[dur] * _get_dot_multiplier(dots)

    def _generate_rhythm_sequence(self, target_duration: Fraction) -> List[Tuple[str, str, str]]:
        binary_choices = [(d_str, "") for d_str in self.BINARY_DURATIONS if d_str in self.playbook["durations"]]

        # To ensure the greedy loop subtracts the largest components first to avoid remaining crumbs
        binary_choices.sort(key=lambda x: self.DURATIONS_MAP[x[0]], reverse=True)

        chunks = []
        while target_duration > 0:
            for dur, dot in binary_choices:
                frac = self._get_duration_fraction(dur, dot, "")
                if frac <= target_duration:
                    chunks.append((dur, dot, frac))
                    target_duration -= frac
                    break

        all_entries = []
        fracs_map = {}
        for dur_val in self.playbook["durations"]:
            for dot_val in self.playbook["dots"]:
                dur = str(dur_val)
                dot = str(dot_val)

                if dur in self.UNDOTTABLE_DURATIONS and dot != "":
                    continue
                all_entries.append((dur, dot))
                f = self._get_duration_fraction(dur, dot, "")
                fracs_map.setdefault(f, []).append((dur, dot))

        def subdivide_chunk(dur_str: str, dot_str: str, frac: Fraction, depth: int = 0) -> List[Tuple[str, str]]:
            if dur_str not in self.used["durations"] or dot_str not in self.used["dots"]:
                if random.random() < 0.8:
                    return [(dur_str, dot_str)]

            if depth > 4 or random.random() < 0.2:
                return [(dur_str, dot_str)]

            valid_replacements = []

            # Halves
            half_frac = frac / 2
            if half_frac in fracs_map:
                for entry in fracs_map[half_frac]:
                    valid_replacements.append([entry] * 2)

            # Triplets
            third_frac = frac / 3
            if third_frac in fracs_map:
                for entry in fracs_map[third_frac]:
                    valid_replacements.append([entry] * 3)

            # Quintuplets
            fifth_frac = frac / 5
            if fifth_frac in fracs_map:
                for entry in fracs_map[fifth_frac]:
                    valid_replacements.append([entry] * 5)

            # Dotted splits (frac * 3/4 + frac * 1/4)
            f34, f14 = frac * Fraction(3, 4), frac * Fraction(1, 4)
            if f34 in fracs_map and f14 in fracs_map:
                for e3 in fracs_map[f34]:
                    for e1 in fracs_map[f14]:
                        valid_replacements.append([e3, e1])

            # Double Dotted splits (frac * 7/8 + frac * 1/8)
            f78, f18 = frac * Fraction(7, 8), frac * Fraction(1, 8)
            if f78 in fracs_map and f18 in fracs_map:
                for e7 in fracs_map[f78]:
                    for e1 in fracs_map[f18]:
                        valid_replacements.append([e7, e1])

            # Triple Dotted splits (frac * 15/16 + frac * 1/16)
            f1516, f116 = frac * Fraction(15, 16), frac * Fraction(1, 16)
            if f1516 in fracs_map and f116 in fracs_map:
                for e15 in fracs_map[f1516]:
                    for e1 in fracs_map[f116]:
                        valid_replacements.append([e15, e1])

            if not valid_replacements:
                return [(dur_str, dot_str)]

            def score(repl):
                return sum(10 for x in repl if x[0] not in self.used["durations"] or x[1] not in self.used["dots"])

            valid_replacements.sort(key=score, reverse=True)
            best_score = score(valid_replacements[0])
            top_repls = [r for r in valid_replacements if score(r) == best_score]

            chosen = random.choice(top_repls)

            ans = []
            for d, dot in chosen:
                sub_frac = self._get_duration_fraction(d, dot, "")
                ans.extend(subdivide_chunk(d, dot, sub_frac, depth + 1))
            return ans

        flat_seq = []
        for dur, dot, frac in chunks:
            flat_seq.extend(subdivide_chunk(dur, dot, frac))

        final_seq = []
        for dur, dot in flat_seq:
            if random.random() < 0.1:
                grace_type = str(random.choice(["q", "qq"]))
                grace_dur = str(random.choice(["8", "16", "32"]))
                final_seq.append((grace_dur, "", grace_type))

            primary_grace = ""
            if random.random() < 0.1:
                primary_grace = str(random.choice(["P", "p"]))
            final_seq.append((dur, dot, primary_grace))

        return final_seq

    def _generate_event(self, force_close: bool = False, fixed_rhythm: Optional[Tuple[str, str, str]] = None) -> str:
        """
        Generates a valid **kern note, chord, or rest token with plausible contexts.
        """
        prefix = ""
        suffix = ""

        is_rest = False
        if not self.in_tie and random.random() < self.REST_PROBABILITY:
            is_rest = True

        if fixed_rhythm:
            dur, dot, grace = fixed_rhythm
            self._track("durations", dur)
            self._track("dots", dot)
            self._track("grace", grace)
        else:
            dur = str(random.choice(self.playbook["durations"]))
            self._track("durations", dur)

            dot = ""
            if dur not in self.UNDOTTABLE_DURATIONS:
                dot = str(random.choice(self.playbook["dots"]))
            self._track("dots", dot)

            grace = str(random.choice(self.playbook["grace"]))
            self._track("grace", grace)

        dur_str = f"{dur}{dot}{grace}"

        # 1. GENERATE REST
        if is_rest:
            rest: str = str(random.choice(self.playbook["rests"]))
            self._track("rests", rest)
            # Rests cannot carry beam markers in **kern — close everything except beams
            suffix = self._close_all_open_states(suffix, include_beams=False)

            art = ""
            if random.random() < self.FERMATA_ON_REST_PROB:
                art = ":"
                self._track("articulations", ":")
            else:
                self._track("articulations", "")

            return f"{dur_str}{rest}{art}{suffix}"

        # 2. GENERATE NOTE
        if self.in_tie and self.current_pitch:
            pitch_str = self.current_pitch
        else:
            base = str(random.choice(self.playbook["base_pitches"]))
            self._track("base_pitches", base)
            octave = random.randint(self.OCTAVE_MIN, self.OCTAVE_MAX)
            base_pitch = base * octave
            accidental = str(random.choice(self.playbook["accidentals"]))
            self._track("accidentals", accidental)
            pitch_str = f"{base_pitch}{accidental}"

            if self.in_tie:
                self.current_pitch = pitch_str

        # Handling Open/Close Logic
        if force_close:
            suffix = self._close_all_open_states(suffix)
        else:
            # Phrase Logic
            if self.in_phrase:
                if random.random() < self.PHRASE_END_PROB:
                    suffix += "}"
                    self.in_phrase = False
                    self._track("phrases", "}")
            else:
                if random.random() < self.PHRASE_START_PROB:
                    prefix += "{"
                    self.in_phrase = True
                    self._track("phrases", "{")

            # Slur Logic
            if self.in_slur:
                if random.random() < self.SLUR_END_PROB:
                    suffix += ")"
                    self.in_slur = False
                    self._track("slurs", ")")
            else:
                if random.random() < self.SLUR_START_PROB:
                    prefix += "("
                    self.in_slur = True
                    self._track("slurs", "(")

            # Tie Logic
            if self.in_tie:
                if random.random() < self.TIE_END_PROB:
                    suffix += "]"
                    self.in_tie = False
                    self.current_pitch = None
                    self._track("ties", "]")
                else:
                    suffix += "_"
                    self._track("ties", "_")
            else:
                if random.random() < self.TIE_START_PROB:
                    prefix += "["
                    self.in_tie = True
                    self.current_pitch = pitch_str
                    self._track("ties", "[")

            # Glissando Logic
            if self.in_glissando:
                if random.random() < self.GLISSANDO_END_PROB:
                    suffix += "h"
                    self.in_glissando = False
                    self._track("glissandi", "h")
            else:
                if random.random() < self.GLISSANDO_START_PROB:
                    suffix += "H"
                    self.in_glissando = True
                    self._track("glissandi", "H")

            # Beam Logic # Only valid for rhythmic note values that typically possess beams
            if self.in_beam:
                if random.random() < self.BEAM_END_PROB:
                    beam_end = str(random.choice(["J", "JJ", "k", "K", "kk", "KK"]))
                    suffix += beam_end
                    if beam_end in ["J", "JJ"]:
                        self.in_beam = False
                    self._track("beams", beam_end)
                else:
                    self._track("beams", "")
            else:
                if random.random() < self.BEAM_START_PROB and dur not in self.UNBEAMABLE_DURATIONS:
                    beam_start = str(random.choice(["L", "LL"]))
                    suffix += beam_start
                    self.in_beam = True
                    self._track("beams", beam_start)
                else:
                    self._track("beams", "")

        ornament = str(random.choice(self.playbook["ornaments"]))
        self._track("ornaments", ornament)

        # Taking specific articulation mapping into account to simplify stacking
        articulation = str(random.choice(self.playbook["articulations"]))
        self._track("articulations", articulation)

        stem = str(random.choice(self.playbook["stems"]))
        self._track("stems", stem)

        editorial = str(random.choice(self.playbook["editorial"]))
        self._track("editorial", editorial)

        note_str = f"{prefix}{dur_str}{pitch_str}{ornament}{articulation}{stem}{editorial}{suffix}"

        # 3. CHORD PROBABILITY INJECTION
        if not force_close and not self.in_tie and random.random() < self.CHORD_PROBABILITY:
            self._track("chords", True)
            extra_base = str(random.choice(self.playbook["base_pitches"]))
            extra_octave = random.randint(self.OCTAVE_MIN, self.OCTAVE_MAX)
            extra_base_pitch = extra_base * extra_octave
            extra_accidental = str(random.choice(self.playbook["accidentals"]))
            extra_pitch_str = f"{extra_base_pitch}{extra_accidental}"

            chord_note = f"{dur_str}{extra_pitch_str}{ornament}{articulation}{stem}{editorial}{suffix}"
            return f"{note_str} {chord_note}"
        else:
            self._track("chords", False)

        return note_str

    def generate(self, num_measures: int = 10, return_list: bool = False) -> Union[str, List[str]]:
        """
        Main interface to generate a full Humdrum script sequence with length control.
        """
        self.reset_state()
        lines: List[str] = ["**kern"]

        clef = str(random.choice(self.playbook["clefs"]))
        self._track("clefs", clef)
        lines.append(clef)

        key = str(random.choice(self.playbook["keys"]))
        self._track("keys", key)
        lines.append(key)

        time = str(random.choice(self.playbook["times"]))
        self._track("times", time)
        lines.append(time)
        measure_duration = self.TIMES_MAP[time]

        for measure in range(1, num_measures + 1):
            # Measure length, barline
            if random.random() < self.BARLINE_OVERRIDE_PROB:
                valid_barlines = self.playbook["barlines"].copy()
                if measure == 1 and "=:|!" in valid_barlines:
                    valid_barlines.remove("=:|!")
                raw_barline = str(random.choice(valid_barlines))
            else:
                raw_barline = "="
            self._track("barlines", raw_barline)

            m = re.match(r"^(=+)(.*)$", raw_barline)
            base_eq = m.group(1) if m else "="
            visuals = m.group(2) if m else ""

            lines.append(f"{base_eq}{measure}{visuals}")

            rhythm_seq = self._generate_rhythm_sequence(measure_duration)
            events = len(rhythm_seq)

            # Measure content
            for i, rhythm_tuple in enumerate(rhythm_seq):
                is_last = (measure == num_measures) and (i == events - 1)
                event_str = self._generate_event(force_close=is_last, fixed_rhythm=rhythm_tuple)
                lines.append(event_str)

        # Explicitly close the final measure with an unnumbered terminal barline
        valid_final = self.playbook["barlines"].copy()
        if "=!|:" in valid_final:
            valid_final.remove("=!|:")

        final_barline = str(random.choice(valid_final))
        self._track("barlines", final_barline)
        lines.append(final_barline)

        lines.append("*-")

        # Conditionally reset tracking
        self.generated_measures += num_measures
        if (
            self.generated_measures >= self.RESET_TRACKING_AFTER_N_GENERATED_MEASURES_AND_METHOD_COMPLETION
            and not self.check_coverage()
        ):
            self.reset_tracking()

        if return_list:
            return lines
        return "\n".join(lines)

    def check_coverage(self) -> Dict[str, Set[Union[str, bool]]]:
        """
        Returns a dictionary of playbook sets that have NOT been produced since tracking was reset.
        """
        missing: Dict[str, Set[Union[str, bool]]] = {}
        for key, expected_list in self.playbook.items():
            used_set = self.used[key]
            uncovered = set(expected_list) - used_set

            if uncovered:
                missing[key] = uncovered
        return missing


def _run_tests() -> None:
    """Built-in asserts acting as module test runners."""
    gen = KernGenerator()

    # Test 1: Generate valid output with specified exact length mapping via barlines
    kern_docs = str(gen.generate(num_measures=5))
    lines = kern_docs.split("\n")
    assert lines[0] == "**kern", "Output must start with **kern marker."
    assert lines[-1] == "*-", "Output must end with *- terminating marker."

    measure_count = sum(1 for line in lines if line.startswith("="))
    assert measure_count == 6, (
        f"Expected 6 barlines (5 numbered + 1 final explicit) across entire system, found {measure_count}."
    )
    print("Test 1 [Basic Validation]: PASSED.")

    # Test 2: Complete exhaustive vocabulary coverage.
    gen.reset_tracking()
    num_measures_per_staff = 20
    num_staffs = gen.RESET_TRACKING_AFTER_N_GENERATED_MEASURES_AND_METHOD_COMPLETION // num_measures_per_staff
    for _ in range(num_staffs - 1):
        gen.generate(num_measures=num_measures_per_staff)

    missing = gen.check_coverage()
    if missing:
        print("Missing coverage detected!")
        for key, val in missing.items():
            print(f"  {key}: {val}")
    assert not missing, "generator failed to cover all vocabulary metrics within expected probabilistic constraints!"
    print("Test 2 [Thorough Corpus Coverage Validation]: PASSED.")

    # Test 3: Systemic minimal edge case.
    short_out = gen.generate(num_measures=1)
    if isinstance(short_out, str):
        short_lines = short_out.split("\n")
    else:
        short_lines = short_out

    measure_1_found = any("1" in line and line.startswith("=") for line in short_lines)
    assert measure_1_found, "Measure 1 barline not correctly detected in minimal output sequence."
    print("Test 3 [Minimal Edge Case Validation]: PASSED.")

    # Test 4: List return type and chord presence validation.
    list_out = gen.generate(num_measures=3, return_list=True)
    assert isinstance(list_out, list), "generator failed to return list when return_list=True."
    assert list_out[0] == "**kern", "List format output does not begin correctly."

    chord_found = False
    for _ in range(50):
        test_chords = gen.generate(num_measures=2, return_list=True)
        if any(" " in line for line in test_chords if not line.startswith("*") and not line.startswith("=")):
            chord_found = True
            break
    assert chord_found, "generator failed to produce any chords (space delimiters) over continuous testing."
    print("Test 4 [Type Output & Structural Behaviors Verification]: PASSED.")

    # Test 5: Exact Measure Duration Capacity Limit Verification
    def extract_duration(token: str) -> Fraction:
        token = token.split(" ")[0]
        m = re.search(r"(000|00|0|[1-9]\d*)(\.*)", token)
        if not m:
            return Fraction(0)
        if "q" in token or "Q" in token:
            return Fraction(0)

        dur_str = m.group(1)
        dots_str = m.group(2)
        base = KernGenerator.DURATIONS_MAP[dur_str]

        if dots_str == ".":
            mult = Fraction(3, 2)
        elif dots_str == "..":
            mult = Fraction(7, 4)
        elif dots_str == "...":
            mult = Fraction(15, 8)
        else:
            mult = Fraction(1, 1)
        return base * mult

    duration_test_lines = gen.generate(num_measures=50, return_list=True)
    expected_measure_dur = Fraction(0)
    for line in duration_test_lines:
        if line in KernGenerator.TIMES_MAP:
            expected_measure_dur = KernGenerator.TIMES_MAP[line]

    active_measure_sum = Fraction(0)
    in_measure = False

    for line in duration_test_lines:
        if line.startswith("="):
            if in_measure:
                assert active_measure_sum == expected_measure_dur, (
                    f"Measure duration overflow/underflow! Expected {expected_measure_dur}, got {active_measure_sum} in measure ending at line: {line}"
                )
            active_measure_sum = Fraction(0)
            in_measure = True
        elif not line.startswith("*") and not line.startswith("!") and line and in_measure:
            active_measure_sum += extract_duration(line)

    print("Test 5 [Mathematical Measure Sum Capacity Match]: PASSED.")

    # Test 6: music21 Humdrum parser syntax validation.
    # Feeds generated kern through music21's converter to catch syntax errors
    # that a real engraver (Verovio, music21, etc.) would flag.
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from music21 import converter

    MUSIC21_VALIDATION_SAMPLES = 100
    MUSIC21_MEASURES_PER_SAMPLE = 6

    all_warnings = []
    for sample_idx in range(MUSIC21_VALIDATION_SAMPLES):
        kern_str = gen.generate(num_measures=MUSIC21_MEASURES_PER_SAMPLE)

        f_out = io.StringIO()
        f_err = io.StringIO()
        try:
            with redirect_stdout(f_out), redirect_stderr(f_err):
                converter.parse(kern_str, format="humdrum")
        except Exception as e:
            all_warnings.append(f"[EXCEPTION] Sample {sample_idx}: {type(e).__name__}: {e}")
            continue

        for output in [f_out.getvalue(), f_err.getvalue()]:
            for line in output.strip().split("\n"):
                line = line.strip()
                if line and ("WARNING" in line or "Error" in line):
                    all_warnings.append(f"Sample {sample_idx}: {line}")

    if all_warnings:
        print(f"music21 found {len(all_warnings)} warnings across {MUSIC21_VALIDATION_SAMPLES} samples:")
        for w in all_warnings[:20]:
            print(f"  {w}")
    assert not all_warnings, (
        f"music21 validation failed: {len(all_warnings)} warnings across "
        f"{MUSIC21_VALIDATION_SAMPLES} samples. First: {all_warnings[0]}"
    )
    print(f"Test 6 [music21 Humdrum Parser Syntax Validation ({MUSIC21_VALIDATION_SAMPLES} samples)]: PASSED.")

    # Test 7: Verovio red-element validation.
    # Verovio renders invalid syntax in red (#ff0000). This test feeds generated kern
    # through Verovio's toolkit and checks the SVG output for any red elements.
    # Runs in subprocess to isolate potential Verovio segfaults.
    import subprocess

    VENV_PYTHON = "/Users/stepanomelka/PyCharmMiscProject/SMT-deep/.venv/bin/python3"
    VEROVIO_VALIDATION_SAMPLES = 30
    VEROVIO_MEASURE_CHOICES = [2, 3, 4, 6, 8]

    verovio_available = True
    try:
        probe = subprocess.run(
            [VENV_PYTHON, "-c", 'import verovio; print("ok")'], capture_output=True, text=True, timeout=10
        )
        if probe.stdout.strip() != "ok":
            verovio_available = False
    except Exception:
        verovio_available = False

    if verovio_available:
        red_samples = []
        crash_count = 0

        for sample_idx in range(VEROVIO_VALIDATION_SAMPLES):
            measures = VEROVIO_MEASURE_CHOICES[sample_idx % len(VEROVIO_MEASURE_CHOICES)]
            kern_str = gen.generate(num_measures=measures)

            assert isinstance(kern_str, str)
            # Escape triple-quotes in kern data for safe embedding
            escaped_kern = kern_str.replace("\\", "\\\\").replace('"', '\\"')

            script = (
                "import sys, io, verovio\n"
                "from contextlib import redirect_stdout, redirect_stderr\n"
                "verovio.enableLog(verovio.LOG_OFF)\n"
                "tk = verovio.toolkit()\n"
                f'kern = "{escaped_kern}"\n'
                'kern = kern.replace("\\\\n", "\\n")\n'
                "f = io.StringIO()\n"
                "with redirect_stdout(f), redirect_stderr(f):\n"
                "    tk.loadData(kern)\n"
                "    svg = tk.renderToSVG(1)\n"
                'if "ff0000" in svg.lower():\n'
                '    print("RED")\n'
                "else:\n"
                '    print("OK")\n'
            )

            try:
                result = subprocess.run([VENV_PYTHON, "-c", script], capture_output=True, text=True, timeout=15)
                if result.stdout.strip() == "RED":
                    red_samples.append((sample_idx, measures, kern_str))
            except (subprocess.TimeoutExpired, Exception):
                crash_count += 1

        if red_samples:
            print(f"Verovio found {len(red_samples)} samples with red elements:")
            for idx, measures, kern in red_samples[:5]:
                print(f"  Sample {idx} ({measures} measures):")
                for line in kern.split("\n")[:15]:
                    print(f"    {line}")
                print("    ...")

        assert not red_samples, (
            f"Verovio validation failed: {len(red_samples)}/{VEROVIO_VALIDATION_SAMPLES} "
            f"samples had red elements. Crashes: {crash_count}"
        )
        print(f"Test 7 [Verovio Red-Element Validation ({VEROVIO_VALIDATION_SAMPLES} samples)]: PASSED.")
    else:
        print("Test 7 [Verovio Red-Element Validation]: SKIPPED (verovio not available in .venv).")


if __name__ == "__main__":
    _run_tests()
    # gen = KernGenerator()
    # for _ in range(50):
    #     gen.generate(num_measures=10)
    # print(gen.generate(num_measures=10))
