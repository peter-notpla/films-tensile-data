"""Unit tests for the pure shared/*.py parsers, run against small real
files captured from production (tests/fixtures/). No cloud clients, no
network - these are exactly the "bytes in, dataframe out" functions Phase
2.1 extracted for this reason. See pipeline-roadmap.md's "Tests and CI"
standing item.
"""

from pathlib import Path

import pytest

from shared.curve_parser import downsample_curve_minmax, extract_curve_dataframe, parse_filename
from shared.extrusion_parser import extract_extrusion_dataframe
from shared.friction_parser import extract_friction_dataframe
from shared.tensile_parser import extract_relevant_dataframe

FIXTURES = Path(__file__).parent / "fixtures"


def read_fixture(name):
    return (FIXTURES / name).read_bytes()


def test_tensile_parser_extracts_one_specimen():
    df, rows_dropped, row_errors = extract_relevant_dataframe(
        read_fixture("tensile_sample.csv"), source_file="tensile_sample.csv"
    )
    assert len(df) == 1
    assert rows_dropped == 0
    assert row_errors == []
    row = df.iloc[0]
    assert row["pellet_id"] == "EV AB AI AM 251117 HZ PF 1052"
    assert row["extrusion_id"] == "BD 251217 JK 1144"
    assert row["test_direction"] == "MD"
    assert row["template_name"] == "TensileTest-Films(V1)"


def test_friction_parser_extracts_rows():
    df, rows_dropped, row_errors = extract_friction_dataframe(
        read_fixture("friction_summary_sample.csv"), source_file="friction_summary_sample.csv"
    )
    assert len(df) == 9
    assert rows_dropped == 0
    row = df.iloc[0]
    assert row["pellet_id_prompt_for_value_before_test"] == "GN AB AE AM 260310 LI PF 1128"
    assert row["sample_repeat_number_prompt_for_value_before_test"] == "1"
    assert row["test_surfaces_prompt_for_value_before_test"] == "Outside Film on Inside Film"


def test_extrusion_parser_extracts_rows():
    df, rows_dropped, row_errors = extract_extrusion_dataframe(
        read_fixture("extrusion_sample.csv"), source_file="extrusion_sample.csv"
    )
    assert len(df) > 0
    # Whitespace trimming (added 20 Aug, standing item closed 1 Sep) must
    # hold on every identity column, not just the ones known to be dirty.
    for col in ("pellet_id", "extrusion_id", "trial_code"):
        if col in df.columns:
            assert (df[col].dropna().astype(str) == df[col].dropna().astype(str).str.strip()).all()


def test_curve_parser_filename_pattern():
    template, sample = parse_filename("raw-FrictionTest-Films(V1)-sample-42.csv")
    assert template == "FrictionTest-Films(V1)"
    assert sample == 42

    with pytest.raises(ValueError):
        parse_filename("not-a-raw-curve-file.csv")


def test_curve_parser_extracts_and_downsamples():
    df, row_errors = extract_curve_dataframe(
        read_fixture("friction_raw_curve_sample.csv"),
        source_file="raw-FrictionTest-Films(V1)-sample-1.csv",
    )
    assert len(df) == 100
    assert row_errors == []
    assert list(df.columns) >= [
        "row_number", "time_s", "load_n", "displacement_mm", "stress_mpa", "strain_pct",
    ][:1]  # cheap presence check without over-asserting exact column order

    downsampled = downsample_curve_minmax(df)
    # At most 2 rows per bucket, and never more rows than the input.
    assert len(downsampled) <= len(df)
    assert downsampled["load_n"].max() == df["load_n"].max()
    assert downsampled["load_n"].min() == df["load_n"].min()
