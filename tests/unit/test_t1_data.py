"""PR-2 (ADR 0026 §3, ADR 0027): T1 text rules, AMI parsing and segments, speaker check, meeting
choice, the manifest, the Colab-only fetch guard, the licence check and the shared ingest.

Everything runs on hand-written fixtures (tests/fixtures/ami/); no corpus file, model or network.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
T1 = REPO / "training" / "t1_whisper_verbatim"
FIX = REPO / "tests" / "fixtures" / "ami"
sys.path.insert(0, str(T1))
sys.path.insert(0, str(REPO / "training"))


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t1text: Any = _load("t1text", T1 / "t1text.py")
ami: Any = _load("ami_words", T1 / "ami_words.py")
prep: Any = _load("prepare_data", T1 / "prepare_data.py")
ci: Any = _load("colab_ingest", REPO / "training" / "colab_ingest.py")
cdata: Any = _load("colab_data", T1 / "colab_data.py")
fetch: Any = _load("fetch_public_data", REPO / "scripts" / "fetch_public_data.py")

EV = yaml.safe_load((REPO / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
HES = yaml.safe_load(
    (REPO / "src" / "presenceai" / "lang" / "en" / "fillers.yaml").read_text(encoding="utf-8")
)["hesitations"]
BACK = EV["t1"]["ami"]["backchannels"]
SEG = EV["t1"]["ami"]["segment"]
SPELL = EV["t1"]["ami"]["spelling"]


# --- config ------------------------------------------------------------------------------------


def test_t1_config_matches_adr_0027_and_owner_answers():
    t1 = EV["t1"]
    assert t1["ami"]["train_speech_hours"] == 10.0  # owner P1: speech kept, after crosstalk
    assert SEG["crosstalk_max_share"] == 0.20  # owner P4 (provisional)
    assert t1["disfluencyspeech"]["text_from"] == "transcript_annotated"  # owner P3
    assert t1["disfluencyspeech"]["text_check_max_words_differ"] == 50  # owner, ADR 0027 §6
    assert t1["disfluencyspeech"]["revision"] == "b7da294fe3a70dd96df6640893f2a5dfc2c87638"
    assert {"mm-hmm", "uh-huh"} <= set(BACK)  # owner P2
    pd = EV["public_data"]
    assert pd["ami_annotations"]["url"].startswith("https://groups.inf.ed.ac.uk/ami/")
    assert pd["ami_headset"]["url"].startswith("https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/")
    assert pd["ami_annotations"]["licence_must_state"] == "CC BY 4.0"
    assert pd["podcastfillers"]["access"] == "range_requests"  # approved with PR-3 (owner E1)
    assert set(SPELL.values()) <= set(HES)  # every mapped filler is in the E3 lexicon


def test_pins_from_data_runs_load_into_the_build_config():
    cfg = cdata.load_config()
    pd = cfg["public_data"]
    sha = re.compile(r"[0-9a-f]{64}")
    assert sha.fullmatch(pd["ami_annotations"]["sha256"])  # run d2
    assert sha.fullmatch(pd["librispeech_test_clean"]["sha256"])  # run d3
    heads = pd["ami_headset"]["sha256"]  # training (d3) + E3 test headsets (b3)
    assert len(heads) == 140 + 112 and all(sha.fullmatch(v) for v in heads.values())
    assert all(k.endswith(".wav") and ".Headset-" in k for k in heads)
    assert "sha256_file" not in pd["ami_headset"] and "sha256_files" not in pd["ami_headset"]
    assert sha.fullmatch(cfg["t1"]["ami"]["manifest_sha256"])  # run d3; T1-r2 checks it
    # the TRAINING pins hold no meeting from the ten E3 sessions; the test pins only E3 test ones
    e3 = set(EV["e3"]["ami_split"]["pool"])
    train = json.loads((REPO / "configs/pins/ami_headsets.json").read_text())["sha256"]
    test = json.loads((REPO / "configs/pins/ami_test_headsets.json").read_text())["sha256"]
    assert len(train) == 140 and not {k.split(".")[0][:-1] for k in train} & e3
    assert {k.split(".")[0][:-1] for k in test} == set(EV["e3"]["scoring"]["ami_test_sessions"])


# --- text rules (owner P2, P3) ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("annotated", "expected"),
    [
        ("{D You know, } it's, {F uh, } fine.", "You know, it's, uh, fine."),
        (
            "Did you see [ the, + the ] show, {F uh, } [ we don't have C, + we don't have cable. ]",
            "Did you see the, the show, uh, we don't have C, we don't have cable.",
        ),
        (
            "Just abandon [ the, + {D you know, } the ] written word -- -- altogether.",
            "Just abandon the, you know, the written word altogether.",
        ),
        (
            "Man, I'm,  - What time is it over th-, [ are you, + are you ] there?",
            "Man, I'm, What time is it over th-, are you, are you there?",
        ),
        ("{C And } <laughter> that is it.", "And that is it."),
        # found by data run d3: Switchboard's uncertain-word marks, and doubled commas
        ("the rest ((of the stuff)).", "the rest of the stuff."),
        ("((Uh)), I ((rest)) it", "Uh, I rest it"),
        ("you {D you know, }, yes", "you you know, yes"),
    ],
)
def test_ds_builder_removes_markup_keeps_fillers_and_partials(annotated, expected):
    assert t1text.build_ds_text(annotated) == expected


@pytest.mark.parametrize(
    ("ours", "a", "kind"),
    [
        ("It was, yet.", "It was, yet.", "same"),
        ("It was yet,", "It was yet.", "punctuation_only"),  # d3: most differences look like this
        ("but, that", "but , that", "punctuation_only"),
        ("over th-, are you", "over the, are you", "partials_only"),
        ("the rest of stuff", "the rest stuff", "words_differ"),
    ],
)
def test_compare_with_transcript_a(ours, a, kind):
    assert t1text.compare_with_transcript_a(ours, a) == kind


def test_backchannels_count_zero_fillers_and_are_not_split():
    for text in ("Mm-hmm.", "mm-hmm", "Uh-huh,", "uh-huh", "MM-HMM!"):
        assert t1text.count_fillers(text, HES, BACK) == 0, text
    assert "mm" in HES and "hmm" in HES  # the halves ARE fillers, so a split would count 2
    assert t1text.count_fillers("mm hmm", HES, BACK) == 2  # separate words are two fillers


def test_partial_filler_counts_as_the_filler_and_not_as_a_partial():
    assert t1text.count_fillers("um- I think", HES, BACK) == 1
    assert t1text.count_fillers("Ummm, uh, er", HES, BACK) == 3
    assert t1text.count_partials("um- I th- think", HES) == 1  # only "th-"
    assert t1text.is_partial("th-,") and not t1text.is_partial("-") and not t1text.is_partial("--")


def test_one_letter_cut_off_word_is_not_a_filler():
    # run d2: "you m- mute", "well, m- it might" -- letter-run collapsing gives "m-" and "mm" the
    # same key, but a one-letter cut-off word is a partial, not the filler "mm".
    assert t1text.count_fillers("you m- mute and", HES, BACK) == 0
    assert t1text.count_partials("you m- mute and", HES) == 1
    assert t1text.count_fillers("mm- right, mm.", HES, BACK) == 2  # two-letter stems still count
    assert t1text.count_fillers("uh- um- er-", HES, BACK) == 3
    assert t1text.count_partials("uh- um- u- m-", HES) == 2  # "u-" and "m-"


# --- AMI ----------------------------------------------------------------------------------------


def _items(agent: str, probe: Any = None) -> list[Any]:
    return ami.parse_words((FIX / f"ES2010a.{agent}.words.xml").read_bytes(), probe)


def test_parse_words_and_probe():
    probe = ami.Probe()
    items = _items("A", probe)
    kinds = Counter(i.kind for i in items)
    assert kinds == {"word": 13, "punc": 3, "gap": 1, "other": 2}
    assert next(i for i in items if i.text == "sh").trunc
    assert probe.w_attributes["trunc"] == 1 and probe.w_attributes["punc"] == 3
    assert probe.short_tokens["um"] == 1 and probe.short_tokens["mm-hmm"] == 1
    assert set(probe.elements) <= ami.KNOWN


@pytest.mark.parametrize(
    ("word", "trunc", "out"),
    [
        ("Um", False, "um"),
        ("Uhm", False, "um"),
        ("um-", False, "um-"),
        ("sh", True, "sh-"),
        ("Mm-hmm", False, "Mm-hmm"),
        ("Uh-huh", False, "Uh-huh"),
        ("Okay", False, "Okay"),
    ],
)
def test_spelling(word, trunc, out):
    assert ami.spell(word, trunc, SPELL, BACK) == out


def _segs(agent: str, other: str, drops: Counter[str]) -> list[Any]:
    others = ami.timed_words(_items(other))
    return ami.segments("ES2010a", agent, "2", _items(agent), others, SEG, SPELL, BACK, drops)


def test_segments_split_at_pauses_and_drop_gap_short_and_crosstalk():
    drops: Counter[str] = Counter()
    segs = _segs("A", "B", drops)
    assert [s.text for s in segs] == ["um, we sh- should start.", "Mm-hmm. Okay then"]
    assert drops == {"gap_element": 1, "too_short": 1, "crosstalk": 1}
    assert segs[0].start_s == pytest.approx(0.9) and segs[0].end_s == pytest.approx(2.7)
    assert all(s.crosstalk_share <= SEG["crosstalk_max_share"] for s in segs)
    assert t1text.count_fillers(segs[0].text, HES, BACK) == 1
    assert t1text.count_partials(segs[0].text, HES) == 1
    assert t1text.count_fillers(segs[1].text, HES, BACK) == 0  # "Mm-hmm" is a backchannel


def test_segments_respect_max_length():
    items = [ami.Item("word", f"w{i}", float(i), i + 0.9) for i in range(70)]
    segs = ami.segments("ES2010a", "A", "0", items, [], SEG, SPELL, BACK)
    assert len(segs) >= 3 and all(s.duration_s <= SEG["max_s"] for s in segs)


def test_overlap_share():
    assert ami.overlap_share(0, 10, [(0, 2), (5, 6)]) == pytest.approx(0.3)
    assert ami.overlap_share(0, 10, []) == 0.0


MEETINGS = ami.parse_meetings((FIX / "meetings.xml").read_bytes())


def test_speaker_check_passes_and_reports_shared_ids_only():
    ok = ami.speaker_check(MEETINGS, ["ES2005"], ["ES2002"])
    assert ok["passed"] and ok["shared_speakers"] == [] and ok["dev_speakers"] == 1
    bad = ami.speaker_check(MEETINGS, ["ES2011"], ["ES2002"])  # XXX001 in both
    assert not bad["passed"] and bad["shared_speakers"] == ["XXX001"]
    missing = ami.speaker_check(MEETINGS, ["IS1001"], ["ES2002"])
    assert not missing["passed"] and missing["missing_sessions"] == ["IS1001"]
    # run d2 listed every dev session as missing (operator precedence); a present dev session
    # and a missing test session must be reported the other way round
    other = ami.speaker_check(MEETINGS, ["ES2005"], ["TS3006"])
    assert other["missing_sessions"] == ["TS3006"] and not other["passed"]


def test_transformerror_is_treated_like_a_gap():
    xml = (
        b'<r xmlns:nite="http://nite.sourceforge.net/">'
        b'<w starttime="1.0" endtime="1.5">one</w><transformerror starttime="1.5" endtime="1.6"/>'
        b'<w starttime="1.6" endtime="2.5">two</w></r>'
    )
    probe = ami.Probe()
    items = ami.parse_words(xml, probe)
    assert [i.kind for i in items] == ["word", "gap", "word"]
    assert "transformerror" in ami.KNOWN and not set(probe.elements) - ami.KNOWN
    drops: Counter[str] = Counter()
    assert ami.segments("ES2010a", "A", "0", items, [], SEG, SPELL, BACK, drops) == []
    assert drops == {"gap_element": 1}


def test_training_pool_excludes_e3_sessions_shared_speakers_and_non_scenario():
    pool, why = ami.training_pool(MEETINGS, ["ES2002", "ES2005"], True)
    assert pool == ["ES2010a", "IS1004a"]
    assert why == {"e3_session": 2, "shares_e3_speaker": 1, "not_scenario": 1}


def test_meeting_order_is_seeded_and_round_robin():
    pool = ["ES2010a", "ES2010b", "IS1004a", "IS1004b", "TS3010a"]
    order = ami.meeting_order(pool, ["ES", "IS", "TS"], 20261127)
    assert order == ami.meeting_order(list(reversed(pool)), ["ES", "IS", "TS"], 20261127)
    assert [m[:2] for m in order[:3]] == ["ES", "IS", "TS"] and sorted(order) == sorted(pool)


def test_select_stops_at_target_hours_of_kept_speech():
    segs = {
        "M": [ami.Segment("M", "A", "0", 10.0 * i, 10.0 * i + 4.0, "x", 1, 0.0) for i in range(10)]
    }
    chosen = ami.select(["M"], segs, 10.0 / 3600.0)  # 10 s target, 4 s segments
    assert len(chosen) == 3 and sum(s.duration_s for s in chosen) == pytest.approx(12.0)


def _annotations(tmp: Path) -> Path:
    root = tmp / "ami_public_manual_1.6.2"
    (root / "words").mkdir(parents=True)
    (root / "corpusResources").mkdir()
    for f in FIX.glob("*.words.xml"):
        shutil.copy(f, root / "words" / f.name)
    shutil.copy(FIX / "meetings.xml", root / "corpusResources" / "meetings.xml")
    return root


def _cfg(hours: float = 1.0) -> dict[str, Any]:
    cfg = cdata.load_config()
    cfg["t1"] = json.loads(json.dumps(cfg["t1"]))
    cfg["t1"]["ami"]["train_speech_hours"] = hours
    return cfg


def test_manifest_is_deterministic_and_counts_kept_speech(tmp_path):
    root = _annotations(tmp_path)
    s1 = prep.build(root, _cfg(), tmp_path / "o1")
    s2 = prep.build(root, _cfg(), tmp_path / "o2")
    assert s1["manifest_sha256"] == s2["manifest_sha256"]
    assert s1["segments"] == 3 and s1["speech_hours_kept"] == pytest.approx(5.0 / 3600, abs=1e-4)
    assert s1["meetings_used"] == ["ES2010a"] and s1["missing_word_files"] == ["IS1004a.A"]
    assert s1["segments_dropped"] == {"crosstalk": 2, "gap_element": 1, "too_short": 1}
    assert s1["fillers"] == 1 and s1["partials"] == 1
    rows = [json.loads(x) for x in (tmp_path / "o1" / "manifest.jsonl").read_text().splitlines()]
    assert {r["text"] for r in rows} >= {"Uh-huh right"}
    assert (
        hashlib.sha256((tmp_path / "o1" / "manifest.jsonl").read_bytes()).hexdigest()
        == (s1["manifest_sha256"])
    )
    probe = json.loads((tmp_path / "o1" / "format_probe.json").read_text())
    assert probe["unknown_elements"] == []


# --- fetch: Colab only, hashes, licence first ---------------------------------------------------


def test_fetch_refuses_outside_colab():
    with pytest.raises(fetch.NotOnColabError):
        fetch.require_colab({})
    with pytest.raises(fetch.NotOnColabError):  # a Colab tag alone is not enough
        fetch.require_colab({"COLAB_RELEASE_TAG": "x"}) if not Path("/content").is_dir() else (
            fetch.require_colab({})
        )


def test_fetch_main_runs_the_guard_before_anything(monkeypatch):
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    with pytest.raises(fetch.NotOnColabError):
        fetch.main(["--only", "librispeech_test_clean", "--dest", "x"])


def test_download_retries_a_busy_server_then_succeeds(tmp_path, monkeypatch):
    import urllib.error

    calls, waits = [], []

    def flaky(url, dest):
        calls.append(url)
        if len(calls) < 3:
            raise urllib.error.HTTPError(url, 503, "Service Unavailable", None, None)  # type: ignore[arg-type]
        dest.write_bytes(b"ok")
        return hashlib.sha256(b"ok").hexdigest(), 2

    monkeypatch.setattr(fetch, "_fetch_once", flaky)
    rec = fetch.download("https://example.org/a.wav", tmp_path / "a.wav", sleep=waits.append)
    assert rec["retries"] == 2 and waits == [15.0, 30.0] and len(calls) == 3


def test_download_does_not_retry_a_missing_file(tmp_path, monkeypatch):
    import urllib.error

    def gone(url, dest):
        raise urllib.error.HTTPError(url, 404, "Not Found", None, None)  # type: ignore[arg-type]

    monkeypatch.setattr(fetch, "_fetch_once", gone)
    waits: list[float] = []
    with pytest.raises(urllib.error.HTTPError):
        fetch.download("https://example.org/a.wav", tmp_path / "a.wav", sleep=waits.append)
    assert waits == []


def test_download_checks_pinned_hash(tmp_path):
    src = tmp_path / "f.bin"
    src.write_bytes(b"abc")
    good = hashlib.sha256(b"abc").hexdigest()
    rec = fetch.download(src.as_uri(), tmp_path / "out" / "f.bin", good)
    assert rec["sha256"] == good and rec["bytes"] == 3 and rec["pinned"]
    with pytest.raises(ValueError, match="mismatch"):
        fetch.download(src.as_uri(), tmp_path / "out" / "g.bin", "0" * 64)
    assert not (tmp_path / "out" / "g.bin").exists()
    with pytest.raises(ValueError, match="non-https"):
        fetch.download("http://example.org/x", tmp_path / "h.bin")


def _zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        for n, t in files.items():
            z.writestr(n, t)
    return path


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        (
            "This work is licensed under the Creative Commons Attribution 4.0 International "
            "License.",
            True,
        ),
        ("Licence: CC BY 4.0", True),
        ("Creative Commons Attribution-NonCommercial-ShareAlike 2.0", False),
        ("CC BY-NC 4.0", False),
        ("All rights reserved.", False),
    ],
)
def test_licence_check(text, ok):
    assert fetch.licence_states(text, "CC BY 4.0") is ok


def test_licence_text_reads_only_the_licence_member(tmp_path):
    z = _zip(tmp_path / "a.zip", {"ami/LICENCE.txt": "CC BY 4.0", "ami/words/x.xml": "<x/>"})
    assert fetch.licence_text(z) == ("ami/LICENCE.txt", "CC BY 4.0")
    with pytest.raises(ValueError, match="expected one licence"):
        fetch.licence_text(_zip(tmp_path / "b.zip", {"ami/words/x.xml": "<x/>"}))


# --- notebook build and ingest ------------------------------------------------------------------


def test_build_injects_modules_and_matches_hash(tmp_path):
    nb_path = cdata.build("t", tmp_path)
    body = (tmp_path / "data_run.py").read_text(encoding="utf-8")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    src = "".join("".join(c["source"]) for c in nb["cells"])
    ast.parse(body)  # the injected script is valid Python
    order = [body.index(s) for s in ("licence_text(", "licence_states(", "extractall(")]
    assert order == sorted(order)  # licence read and checked before anything is extracted
    assert hashlib.sha256(body.encode()).hexdigest() in src
    assert "__CONFIG_JSON__" not in body and "__MODULES_JSON__" not in body
    mods = json.loads(body.split('MODULES = json.loads(r"""', 1)[1].split('""")', 1)[0])
    assert set(mods) == {
        "t1text.py",
        "ami_words.py",
        "prepare_data.py",
        "fetch_public_data.py",
        "normalize.py",
    }
    assert b"def filler_key" in base64.b64decode(mods["normalize.py"])
    assert "nvidia-smi" not in src  # CPU runtime is enough
    assert "data/colab_results/" in src


def test_notebook_resumes_on_drive_reads_token_safely_and_never_cuts_the_download(tmp_path):
    nb = json.loads(cdata.build("t", tmp_path).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    body = (tmp_path / "data_run.py").read_text(encoding="utf-8")
    drive = next(c for c in cells if "drive.mount" in c)
    assert "MyDrive/presenceai_p4/p4data_t/p4data" in drive and "P4_OUT" in drive
    assert 'WORK = Path("/content/p4work")' in body  # datasets stay on local disk, not Drive
    secret = next(c for c in cells if "userdata.get" in c)
    assert "userdata.get('HF_TOKEN')" in secret and "value not shown" in secret
    for c in [*cells, body]:  # the token value is never printed or written anywhere
        assert "print(os.environ" not in c and "print(tok" not in c
    assert body.count('os.environ.get("HF_TOKEN"') == 1  # read in one place only ...
    assert body.count("hf_token()") == 2  # its definition and one call ...
    assert "HfFileSystem(token=token)" in body  # ... which only builds the HF file system
    last = cells[-1]
    assert "DELETE_RUNTIME = False" in last and "if DELETE_RUNTIME:" in last
    zip_cell = next(c for c in cells if "make_archive" in c)
    assert cells.index(zip_cell) < len(cells) - 1 and "files.download" in zip_cell
    assert "MyDrive/presenceai_p4/p4data_t" in zip_cell  # copy kept on Drive


def _run_zip(tmp: Path, files: dict[str, bytes], tamper: bool = False) -> Path:
    root = tmp / "build" / "p4data"
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    table = ci.run_info_table(root)
    if tamper:
        (root / next(iter(files))).write_bytes(b"changed")
    (root / "run_info.json").write_text(json.dumps({"status": "ok", "sha256": table}))
    z = tmp / "inbox" / "p4data_t.zip"
    z.parent.mkdir(parents=True)
    shutil.make_archive(str(z.with_suffix("")), "zip", tmp / "build", "p4data")
    return z


def test_ingest_copies_checked_files_and_deletes_zip(tmp_path):
    z = _run_zip(tmp_path, {"t1/manifest.jsonl": b"{}\n", "t1/manifest_summary.json": b"{}"})
    data, res = tmp_path / "data", tmp_path / "res"
    out = ci.ingest(
        z, "p4data", [("t1/manifest.jsonl", data), ("*", res)], tmp_path / "scr", info_dest=res
    )
    assert out["status"] == "ok" and out["files_verified"] == 2
    assert (data / "t1" / "manifest.jsonl").is_file()
    assert (res / "t1" / "manifest_summary.json").is_file() and (res / "run_info.json").is_file()
    assert not z.exists() and not (tmp_path / "scr").exists()


def test_ingest_refuses_mismatch_keeps_zip_and_copies_nothing(tmp_path):
    z = _run_zip(tmp_path, {"t1/manifest.jsonl": b"{}\n"}, tamper=True)
    with pytest.raises(ValueError, match="checksum"):
        ci.ingest(
            z, "p4data", [("*", tmp_path / "res")], tmp_path / "scr", info_dest=tmp_path / "res"
        )
    assert z.exists() and not (tmp_path / "res").exists() and not (tmp_path / "scr").exists()


def test_ingest_rejects_unsafe_paths(tmp_path):
    z = _zip(tmp_path / "bad.zip", {"../evil.txt": "x"})
    with pytest.raises(ValueError, match="unsafe"):
        ci.ingest(
            z, "p4data", [("*", tmp_path / "res")], tmp_path / "scr", info_dest=tmp_path / "res"
        )


def test_ingest_default_inbox_is_data_colab_results():
    assert ci.INBOX == REPO / "data" / "colab_results"
    with pytest.raises(FileNotFoundError, match="colab_results"):
        cdata.ingest("does-not-exist")


# --- Drive cache of pinned public files (owner, 2026-11-28; ADR 0028 §9) --------------------------


def test_cache_hit_needs_no_request_and_verifies(tmp_path, monkeypatch):
    data = b"pinned audio"
    pin = hashlib.sha256(data).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.wav").write_bytes(data)

    def no_network(url, dest):
        raise AssertionError("a verified cached file must not be downloaded")

    monkeypatch.setattr(fetch, "_fetch_once", no_network)
    rec = fetch.download(
        "https://example.org/a.wav", tmp_path / "w" / "a.wav", pin, cache_dir=cache
    )
    assert rec["from_cache"] and (tmp_path / "w" / "a.wav").read_bytes() == data


def test_bad_cached_copy_is_deleted_and_downloaded_again(tmp_path, monkeypatch):
    good = b"pinned audio"
    pin = hashlib.sha256(good).hexdigest()
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "a.wav").write_bytes(b"corrupted")
    calls = []

    def fetch_once(url, dest):
        calls.append(url)
        dest.write_bytes(good)
        return pin, len(good)

    monkeypatch.setattr(fetch, "_fetch_once", fetch_once)
    rec = fetch.download(
        "https://example.org/a.wav", tmp_path / "w" / "a.wav", pin, cache_dir=cache
    )
    assert not rec["from_cache"] and len(calls) == 1
    assert (cache / "a.wav").read_bytes() == good  # replaced by the verified download
    manifest = json.loads((cache / "cache_manifest.json").read_text())
    assert manifest["a.wav"]["sha256"] == pin


def test_unpinned_files_are_never_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(
        fetch,
        "_fetch_once",
        lambda url, dest: (dest.write_bytes(b"x"), (hashlib.sha256(b"x").hexdigest(), 1))[1],
    )
    cache = tmp_path / "cache"
    fetch.download("https://example.org/b.wav", tmp_path / "w" / "b.wav", None, cache_dir=cache)
    assert not cache.exists()
    assert fetch.cache_put(cache, "c.wav", "0" * 64, tmp_path / "w" / "b.wav", "x") is False


def test_cache_put_refuses_a_file_that_does_not_match_its_pin(tmp_path):
    src = tmp_path / "a.wav"
    src.write_bytes(b"abc")
    assert not fetch.cache_put(tmp_path / "cache", "a.wav", "0" * 64, src, "test")
    assert not (tmp_path / "cache" / "a.wav").exists()
