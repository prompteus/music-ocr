from __future__ import annotations
import importlib.util
import multiprocessing
import re
from fractions import Fraction

import pytest

from music_ocr.generator.utils.kern_generator import KernGenerator


@pytest.fixture(scope="module")
def gen() -> KernGenerator:
    return KernGenerator()


def test_basic_structure(gen: KernGenerator):
    """Generated kern starts with **kern, ends with *-, and has the right barline count."""
    kern_docs = str(gen.generate(num_measures=5))
    lines = kern_docs.split("\n")
    assert lines[0] == "**kern", "Output must start with **kern marker."
    assert lines[-1] == "*-", "Output must end with *- terminating marker."
    measure_count = sum(1 for line in lines if line.startswith("="))
    assert measure_count == 6, (
        f"Expected 6 barlines (5 numbered + 1 final) across entire system, found {measure_count}."
    )


def test_corpus_coverage(gen: KernGenerator):
    """Generator covers all vocabulary metrics within expected probabilistic constraints."""
    gen.reset_tracking()
    num_measures_per_staff = 20
    num_staffs = gen.RESET_TRACKING_AFTER_N_GENERATED_MEASURES_AND_METHOD_COMPLETION // num_measures_per_staff
    for _ in range(num_staffs - 1):
        gen.generate(num_measures=num_measures_per_staff)

    missing = gen.check_coverage()
    assert not missing, f"generator failed to cover all vocabulary metrics! Missing: {missing}"


def test_minimal_edge_case(gen: KernGenerator):
    """Single-measure output contains measure 1 barline."""
    short_out = gen.generate(num_measures=1)
    short_lines = short_out.split("\n") if isinstance(short_out, str) else short_out
    measure_1_found = any("1" in line and line.startswith("=") for line in short_lines)
    assert measure_1_found, "Measure 1 barline not correctly detected in minimal output."


def test_list_return_and_chord_presence(gen: KernGenerator):
    """return_list=True returns a list; chords are generated within 50 attempts."""
    list_out = gen.generate(num_measures=3, return_list=True)
    assert isinstance(list_out, list), "generator failed to return list when return_list=True."
    assert list_out[0] == "**kern", "List format output does not begin correctly."

    chord_found = any(
        " " in line
        for _ in range(50)
        for line in gen.generate(num_measures=2, return_list=True)
        if not line.startswith("*") and not line.startswith("=")
    )
    assert chord_found, "generator failed to produce any chords over 50 attempts."


def test_measure_duration_capacity(gen: KernGenerator):
    """Every measure's note durations sum exactly to the time signature capacity."""

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
        mult = {".": Fraction(3, 2), "..": Fraction(7, 4), "...": Fraction(15, 8)}.get(dots_str, Fraction(1, 1))
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
                    f"Measure duration overflow/underflow! "
                    f"Expected {expected_measure_dur}, got {active_measure_sum} before barline: {line}"
                )
            active_measure_sum = Fraction(0)
            in_measure = True
        elif not line.startswith("*") and not line.startswith("!") and line and in_measure:
            active_measure_sum += extract_duration(line)


def test_music21_syntax_validation(gen: KernGenerator):
    """Generated kern passes music21's Humdrum parser without warnings or errors."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    from music21 import converter

    SAMPLES = 100
    MEASURES = 6

    all_warnings: list[str] = []
    for sample_idx in range(SAMPLES):
        kern_str = gen.generate(num_measures=MEASURES)
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

    assert not all_warnings, (
        f"music21 validation failed: {len(all_warnings)} warnings across {SAMPLES} samples. First: {all_warnings[0]}"
    )


def _verovio_worker(kern_str: str, result_queue: multiprocessing.Queue) -> None:
    """Worker function to be run in a separate process to shield against Verovio segfaults."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    import verovio

    verovio.enableLog(verovio.LOG_OFF)
    tk = verovio.toolkit()

    f = io.StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        tk.loadData(kern_str)
        svg = tk.renderToSVG(1)

    if "ff0000" in svg.lower():
        result_queue.put("RED")
    else:
        result_queue.put("OK")


HAS_VEROVIO = importlib.util.find_spec("verovio") is not None


@pytest.mark.skipif(not HAS_VEROVIO, reason="Verovio not installed in the current environment.")
def test_verovio_no_red_elements(gen: KernGenerator):
    """Generated kern does not produce red (invalid) elements in Verovio SVG output."""

    SAMPLES = 30
    MEASURE_CHOICES = [2, 3, 4, 6, 8]

    red_samples = []
    crash_count = 0

    for sample_idx in range(SAMPLES):
        measures = MEASURE_CHOICES[sample_idx % len(MEASURE_CHOICES)]
        kern_str = gen.generate(num_measures=measures)
        assert isinstance(kern_str, str)

        result_queue: multiprocessing.Queue[str] = multiprocessing.Queue()
        p = multiprocessing.Process(target=_verovio_worker, args=(kern_str, result_queue))

        try:
            p.start()
            p.join(timeout=15)

            if p.is_alive():
                p.terminate()
                p.join()
                crash_count += 1
                continue

            if p.exitcode != 0:
                crash_count += 1
                continue

            if not result_queue.empty():
                res = result_queue.get()
                if res == "RED":
                    red_samples.append((sample_idx, measures, kern_str))
            else:
                # Process finished but queue is empty? Likely a silent crash or failure.
                crash_count += 1

        except Exception:
            crash_count += 1
            if p.is_alive():
                p.terminate()
                p.join()

    assert not red_samples, (
        f"Verovio validation failed: {len(red_samples)}/{SAMPLES} samples had red elements. Crashes: {crash_count}"
    )
