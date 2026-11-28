"""PR-3 (ADR 0028): E3 protocol config, scoring, the AMI disfluency parser and the split-zip
reader. Hand-written fixtures only; no model, corpus file or network."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
T1 = REPO / "training" / "t1_whisper_verbatim"
FIX = REPO / "tests" / "fixtures" / "ami"
sys.path.insert(0, str(T1))


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


t1text: Any = _load("t1text", T1 / "t1text.py")
sc: Any = _load("e3score", T1 / "e3score.py")
ami: Any = _load("ami_words", T1 / "ami_words.py")
sz: Any = _load("splitzip", T1 / "splitzip.py")

EV = yaml.safe_load((REPO / "configs" / "evaluation.yaml").read_text(encoding="utf-8"))
DEFAULT = yaml.safe_load((REPO / "configs" / "default.yaml").read_text(encoding="utf-8"))
E3 = EV["e3"]
HES = yaml.safe_load(
    (REPO / "src" / "presenceai" / "lang" / "en" / "fillers.yaml").read_text(encoding="utf-8")
)["hesitations"]
BACK = EV["t1"]["ami"]["backchannels"]


# --- ADR 0028: protocol fixed before the run -----------------------------------------------


def test_decoding_is_the_apps_asr_settings_unchanged():
    asr, dec = DEFAULT["asr"], E3["decoding"]
    for k in ("compute_type", "beam_size", "temperature", "condition_on_previous_text"):
        assert dec[k] == asr[k], k  # owner 1: the app's settings, never changed for E3 or T1
    assert dec["language"] == "en" and dec["word_timestamps"] is True
    assert dec["vad_filter"] is False  # as speech/asr.py
    assert E3["model"] == "whisper_small" == asr["model"]


def test_prompt_normaliser_and_detector_are_fixed():
    assert E3["initial_prompt"] == (
        "Umm, let me think, uh, like, hmm... Okay, so, um, here's what I was, uh, thinking."
    )
    assert E3["wer_normaliser"] == {
        "lower_case": True,
        "remove_punctuation": True,
        "remove_fillers": False,
    }
    assert E3["detector"] == "lexicon"
    pf = E3["podcastfillers"]
    assert pf["scored_labels"] == ["Uh", "Um"] and pf["match_tolerance_s"] == 0.2
    assert set(pf["hypothesis_family"]) == {"uh", "um", "uhm", "er", "erm"}
    assert E3["ami"]["deciding_unit"] == "t1_segments"
    assert EV["public_data"]["podcastfillers"]["access"] == "range_requests"


# --- scoring ---------------------------------------------------------------------------------


def test_alignment_filler_counts():
    c = sc.filler_counts("I uh think um so", "I uh think so uh", HES, BACK)
    assert c == sc.Counts(tp=1, fp=1, fn=1)  # "uh" kept, "um" dropped, an "uh" invented
    assert sc.prf(c) == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    # type need not match; backchannels and one-letter partials are not fillers
    assert sc.filler_counts("well um yes", "well uh yes", HES, BACK) == sc.Counts(1, 0, 0)
    assert sc.filler_counts("mm-hmm m- mute", "mm-hmm m- mute", HES, BACK) == sc.Counts(0, 0, 0)
    assert sc.prf(sc.Counts()) == {"precision": None, "recall": None, "f1": None}


def test_time_match_is_one_to_one_with_tolerance():
    ref = [sc.Event(1.0, 1.3), sc.Event(5.0, 5.4)]
    hyp = [sc.Event(1.45, 1.6), sc.Event(1.5, 1.7), sc.Event(9.0, 9.2)]  # 1st within +0.2 s
    assert sc.time_match(hyp, ref, 0.2) == sc.Counts(tp=1, fp=2, fn=1)
    assert sc.time_match(hyp, ref, 0.0) == sc.Counts(tp=0, fp=3, fn=2)


def test_hyp_filler_events_use_the_family_only():
    words = [("Uh,", 0.0, 0.2), ("hmm", 1.0, 1.2), ("Erm", 2.0, 2.3), ("like", 3.0, 3.1)]
    ev = sc.hyp_filler_events(words, E3["podcastfillers"]["hypothesis_family"])
    assert [(e.start, e.end) for e in ev] == [(0.0, 0.2), (2.0, 2.3)]


def test_repetitions_kept():
    ref = ["we", "we", "should", "start"]
    assert sc.repetitions_kept(ref, ["we", "we", "should", "start"], [[0, 1]]) == (1, 1)
    assert sc.repetitions_kept(ref, ["we", "should", "start"], [[0, 1]]) == (0, 1)


def test_wer_normaliser_keeps_fillers():
    cfg = E3["wer_normaliser"]
    ref = sc.normalise_wer("Hello, World! It's fine.", cfg)
    assert ref == ["hello", "world", "it's", "fine"]
    hyp = sc.normalise_wer("hello uh world it's fine", cfg)
    assert sc.word_errors(ref, hyp) == 1  # the invented "uh" is an error
    with pytest.raises(ValueError):
        sc.normalise_wer("x", {**cfg, "remove_fillers": True})


def test_bootstrap_is_seeded_and_brackets_the_point():
    units = {f"m{i}": sc.Counts(tp=8 + i % 3, fp=2, fn=1 + i % 2) for i in range(20)}
    lo, hi = sc.bootstrap(units, sc.f1_of, 500, 7)
    point = sc.prf(sum(units.values(), start=sc.Counts()))["f1"]
    assert lo <= point <= hi and (lo, hi) == sc.bootstrap(units, sc.f1_of, 500, 7)
    w = {f"s{i}": (i % 4, 20) for i in range(10)}
    lo, hi = sc.bootstrap(w, sc.wer_of, 200, 7)
    assert 0 <= lo <= hi <= 100


# --- AMI disfluency layer (assumed format; probed on Colab first) ------------------------------


def test_disfluency_parser_resolves_ranges_and_finds_repetitions():
    items = ami.parse_words((FIX / "ES2010a.A.words.xml").read_bytes())
    order = [i.id for i in items if i.id]
    probe = ami.DisfluencyProbe()
    dis = ami.parse_disfluency((FIX / "ES2010a.A.disfluency.xml").read_bytes(), order, probe)
    assert [d.type for d in dis] == ["Repetition", "Hesitation"]
    assert dis[0].word_ids == ("ES2010a.A.words2", "ES2010a.A.words3")
    assert [d.type for d in ami.repetitions(dis)] == ["Repetition"]
    assert probe.unresolved_links == 1 and probe.type_values["Repetition"] == 2


def test_segment_word_ids_line_up_with_tokens_and_stay_out_of_the_manifest():
    items = ami.parse_words((FIX / "ES2010a.A.words.xml").read_bytes())
    seg = ami.segments(
        "ES2010a",
        "A",
        "2",
        items,
        [],
        EV["t1"]["ami"]["segment"],
        EV["t1"]["ami"]["spelling"],
        BACK,
    )[0]
    assert len(seg.word_ids) == len(seg.text.split()) == seg.n_words
    assert seg.word_ids[2] == "ES2010a.A.words3"  # "sh-"
    from dataclasses import asdict

    assert "word_ids" in asdict(seg)  # prepare_data pops it before writing the manifest


# --- split zip ----------------------------------------------------------------------------------


def _zip_bytes(files: dict[str, bytes], method: int) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=method) as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


def _as_split(data: bytes, cut_member: int) -> tuple[list[bytes], list[Any]]:
    """Turn a plain zip into a 2-part split archive: members from `cut_member` on (and the
    central directory) live on disk 1, with offsets relative to disk 1, as Info-ZIP writes."""
    infos = zipfile.ZipFile(io.BytesIO(data)).infolist()
    cut = infos[cut_member].header_offset
    part0, rest = data[:cut], bytearray(data[cut:])
    eocd = rest.rfind(struct.pack("<I", sz.EOCD))
    (_, _, _, n1, n2, cd_size, cd_off, clen) = struct.unpack_from("<IHHHHIIH", rest, eocd)
    cd_rel = cd_off - cut
    i = cd_rel
    for _ in range(n2):
        nlen, elen, clen2 = struct.unpack_from("<HHH", rest, i + 28)
        off = struct.unpack_from("<I", rest, i + 42)[0]
        if off >= cut:
            struct.pack_into("<H", rest, i + 34, 1)  # disk number start
            struct.pack_into("<I", rest, i + 42, off - cut)
        i += 46 + nlen + elen + clen2
    struct.pack_into("<IHHHHIIH", rest, eocd, sz.EOCD, 1, 1, n1, n2, cd_size, cd_rel, clen)
    return [part0, bytes(rest)], infos


def _parts(chunks: list[bytes]) -> Any:
    return sz.Parts([len(c) for c in chunks], lambda p, s, n: chunks[p][s : s + n])


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_split_zip_index_and_extract(tmp_path, method):
    files = {
        "LICENSE_AUDIO.csv": b"episode,licence\n",
        "audio/episode_wav/test/a.wav": bytes(range(256)) * 400,
        "audio/episode_wav/test/b.wav": b"RIFF" + b"\x01\x02" * 50_000,
    }
    chunks, _ = _as_split(_zip_bytes(files, method), cut_member=2)
    parts = _parts(chunks)
    idx = sz.index(parts)
    assert set(idx) == set(files)
    assert idx["audio/episode_wav/test/b.wav"].disk == 1
    for name, data in files.items():
        rec = sz.extract(parts, idx[name], tmp_path / name)
        assert (tmp_path / name).read_bytes() == data
        assert rec["sha256"] == hashlib.sha256(data).hexdigest()
        assert rec["crc32"] == f"{zlib.crc32(data):08x}"


def test_split_zip_reads_across_a_part_boundary(tmp_path):
    data = _zip_bytes({"x.wav": bytes(range(256)) * 1000}, zipfile.ZIP_STORED)
    mid = len(data) // 2  # cut through the member's bytes, as split parts do
    parts = _parts([data[:mid], data[mid:]])
    idx = sz.index(parts)
    rec = sz.extract(parts, idx["x.wav"], tmp_path / "x.wav")
    assert rec["bytes"] == 256_000


def test_split_zip_crc_mismatch_deletes_the_file(tmp_path):
    data = bytearray(_zip_bytes({"x.wav": b"abcdef" * 1000}, zipfile.ZIP_STORED))
    pos = data.find(b"abcdef")
    data[pos] ^= 0xFF  # corrupt one byte of the member
    parts = _parts([bytes(data)])
    with pytest.raises(ValueError, match="CRC32"):
        sz.extract(parts, sz.index(parts)["x.wav"], tmp_path / "x.wav")
    assert not (tmp_path / "x.wav").exists()


def test_zip64_extra_field():
    extra = struct.pack("<HH", 0x0001, 16) + struct.pack("<QQ", 5_000_000_000, 4_900_000_000)
    assert sz._zip64_extra(extra, ["size", "compressed"]) == {
        "size": 5_000_000_000,
        "compressed": 4_900_000_000,
    }
    with pytest.raises(ValueError):
        sz._zip64_extra(b"", ["offset"])


def test_http_fetch_refuses_a_full_response(monkeypatch):
    class R:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b""

    monkeypatch.setattr(sz.urllib.request, "urlopen", lambda req, timeout: R())
    with pytest.raises(sz.RangeNotSupportedError):
        sz.http_fetch(["https://example.org/a.zip"])(0, 0, 10)


# --- the E3 Colab build ---------------------------------------------------------------------------

sys.path.insert(0, str(REPO / "training"))
ce3: Any = _load("colab_e3", T1 / "colab_e3.py")


def _lock_version(pkg: str) -> str:
    lock = (REPO / "uv.lock").read_text(encoding="utf-8")
    block = lock.split(f'name = "{pkg}"\n', 1)[1]
    return block.split('version = "', 1)[1].split('"', 1)[0]


def test_e3_build_uses_the_apps_model_versions_and_settings(tmp_path):
    import ast
    import json

    nb = json.loads(ce3.build("t", tmp_path).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    body = (tmp_path / "e3_run.py").read_text(encoding="utf-8")
    ast.parse(body)
    src = "".join(cells)
    assert "'T4' in gpu" in src  # GPU run
    setup = next(c for c in cells if "pip install" in c)
    assert f"faster-whisper=={_lock_version('faster-whisper')}" in setup
    assert f"ctranslate2=={_lock_version('ctranslate2')}" in setup
    cfg = json.loads(body.split('CONFIG = json.loads(r"""', 1)[1].split('""")', 1)[0])
    man = json.loads((REPO / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    small = next(m for m in man["models"] if m["name"] == "whisper_small")
    assert cfg["model"]["files"] == small["files"]  # converted model must equal the app's
    assert cfg["model"]["source"]["revision"] == small["source"]["revision"]
    assert cfg["e3"]["decoding"] == E3["decoding"] and cfg["e3"]["initial_prompt"]
    assert len(cfg["public_data"]["ami_headset"]["sha256"]) == 140 + 112  # training + test pins
    mods = json.loads(body.split('MODULES = json.loads(r"""', 1)[1].split('""")', 1)[0])
    assert {"e3score.py", "splitzip.py", "ami_words.py", "t1text.py", "normalize.py"} <= set(mods)
    # stage order: the builder check first, then the model, then data, ASR and scoring
    order = [
        body.index(s)
        for s in (
            "stage_ds_check()",
            "stage_model()",
            "stage_data()",
            "stage_asr(cond",
            "score(all_items)",
        )
    ]
    assert order == sorted(order)
    assert 'initial_prompt=E3["initial_prompt"] if cond == "base_initial_prompt" else None' in body
    assert body.count('os.environ.get("HF_TOKEN"') == 1
    assert "DELETE_RUNTIME = False" in cells[-1]
    assert "MyDrive/presenceai_p4/e3base_t/e3base" in src
    # run b1: Colab's python3.12 has no ensurepip, so the converter env is made with uv (pinned
    # to the laptop's uv version), never with `python -m venv`
    assert '"-m", "venv"' not in body and '"venv", "--python", py_minor' in body
    assert cfg["uv_version"].count(".") == 2 and "uv_pin" in body


def test_e3_ingest_keeps_hypotheses_and_items_out_of_git(tmp_path, monkeypatch):
    import json
    import shutil

    ci = sys.modules["colab_ingest"]
    root = tmp_path / "build" / "e3base"
    files = {"hyps/base.jsonl": b"{}", "items/ami_test.json": b"[]", "e3_metrics.json": b"{}"}
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    table = ci.run_info_table(root)
    (root / "run_info.json").write_text(json.dumps({"status": "ok", "sha256": table}))
    z = tmp_path / "e3base_t.zip"
    shutil.make_archive(str(z.with_suffix("")), "zip", tmp_path / "build", "e3base")
    monkeypatch.setattr(ce3, "RESULTS", tmp_path / "res")
    monkeypatch.setattr(ce3, "DATA", tmp_path / "data")
    out = ce3.ingest("t", z)
    assert out["status"] == "ok" and not z.exists()
    assert (tmp_path / "data" / "t" / "hyps" / "base.jsonl").is_file()  # git-ignored data/
    assert (tmp_path / "data" / "t" / "items" / "ami_test.json").is_file()
    assert (tmp_path / "res" / "base_t" / "e3_metrics.json").is_file()  # committed results
    assert (tmp_path / "res" / "base_t" / "run_info.json").is_file()


def test_model_check_normalises_line_endings_only_for_the_converters_json_files():
    lf = E3["model_text_files_lf_sha256"]
    assert set(lf) == {"config.json", "vocabulary.json"}  # model.bin etc. stay byte-for-byte
    man = __import__("json").loads((REPO / "models" / "MANIFEST.json").read_text(encoding="utf-8"))
    small = next(m for m in man["models"] if m["name"] == "whisper_small")
    files = {f["path"] for f in small["files"]}
    assert set(lf) < files and {"model.bin", "tokenizer.json"} <= files - set(lf)


def test_lf_pins_equal_the_laptop_model_files_with_normalised_line_endings():
    d = REPO / "models" / "whisper-small-ct2-int8"
    if not (d / "config.json").exists():
        pytest.skip("app model not downloaded here (CI)")  # file read only; no model is run
    for name, want in E3["model_text_files_lf_sha256"].items():
        data = (d / name).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == want, name


def test_disfluency_types_come_from_pointers_into_the_ontology():
    onto = ami.load_ontology((FIX / "dsfl-types.xml").read_bytes())
    assert onto["dsfl_rep"].startswith("disfluency/Repetition")  # ancestor label kept
    items = ami.parse_words((FIX / "ES2010a.B.words.xml").read_bytes())
    order = [i.id for i in items if i.id]
    probe = ami.DisfluencyProbe()
    dis = ami.parse_disfluency((FIX / "ES2010a.B.disfluency.xml").read_bytes(), order, probe, onto)
    reps = ami.repetitions(dis)
    assert len(dis) == 3 and len(reps) == 1
    assert reps[0].word_ids == ("ES2010a.B.words0", "ES2010a.B.words1")
    assert probe.pointer_roles == {"dsfl-type": 3}
    assert probe.pointer_targets == {"dsfl-types.xml": 3}
    assert probe.unresolved_pointers == 1  # dsfl_unknown is not in the ontology
    # without the ontology nothing is called a repetition (never a guess)
    assert (
        ami.repetitions(
            ami.parse_disfluency((FIX / "ES2010a.B.disfluency.xml").read_bytes(), order, None, {})
        )
        == []
    )


@pytest.mark.parametrize(
    "header",
    [
        [
            "clip_name",
            "fullvoc_label",
            "episode_split_subset",
            "podcast_filename",
            "event_start_inepisode",
            "event_end_inepisode",
        ],
        [
            "﻿clip_name",
            " FullVoc_Label ",
            "Episode Split Subset",
            "podcast_filename",
            "event_start_inepisode",
            "event_end_inepisode",
        ],  # BOM, spaces, case
        ["clip", "label_full_vocab", "split", "episode_filename", "event_start", "event_end"],
    ],
)
def test_pf_columns_found_by_normalised_alias(header):
    cols = sc.pf_columns(header)
    assert all(cols.values()) and set(cols.values()) <= set(header)


def test_pf_columns_missing_is_reported_not_guessed():
    cols = sc.pf_columns(["clip_name", "something_else"])
    assert cols["label"] is None
    with pytest.raises(ValueError, match="missing PodcastFillers columns"):
        sc.pf_test_events([], cols, ["Uh"])


def test_pf_test_events_keeps_test_split_and_scored_labels_only():
    cols = sc.pf_columns(
        [
            "fullvoc_label",
            "episode_split_subset",
            "podcast_filename",
            "event_start_inepisode",
            "event_end_inepisode",
        ]
    )
    rows = [
        {
            "fullvoc_label": "Uh",
            "episode_split_subset": "test",
            "podcast_filename": "a",
            "event_start_inepisode": "1.0",
            "event_end_inepisode": "1.2",
        },
        {
            "fullvoc_label": "um",
            "episode_split_subset": "Test",
            "podcast_filename": "a",
            "event_start_inepisode": "3.0",
            "event_end_inepisode": "3.4",
        },
        {
            "fullvoc_label": "Breath",
            "episode_split_subset": "test",
            "podcast_filename": "b",
            "event_start_inepisode": "5.0",
            "event_end_inepisode": "5.2",
        },
        {
            "fullvoc_label": "Uh",
            "episode_split_subset": "train",
            "podcast_filename": "c",
            "event_start_inepisode": "1.0",
            "event_end_inepisode": "1.2",
        },
    ]
    eps, ev = sc.pf_test_events(rows, cols, E3["podcastfillers"]["scored_labels"])
    assert eps == {"a", "b"} and ev == {"a": [[1.0, 1.2], [3.0, 3.4]]}


def test_e3_notebook_sets_the_drive_cache_and_the_pins_reach_the_script(tmp_path):
    import json

    nb = json.loads(ce3.build("t", tmp_path).read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    drive = next(c for c in cells if "drive.mount" in c)
    assert "os.environ['P4_CACHE'] = '/content/drive/MyDrive/presenceai_data'" in drive
    body = (tmp_path / "e3_run.py").read_text(encoding="utf-8")
    cfg = json.loads(body.split('CONFIG = json.loads(r"""', 1)[1].split('""")', 1)[0])
    assert len(cfg["public_data"]["podcastfillers"]["sha256"]) == 20  # run b4 pins
    for name in ("ami_annotations", "ami_headsets", "podcastfillers_test_wav", "librispeech"):
        assert f'cache("{name}")' in body


# --- repetitions from repairs (owner 2026-11-28; ADR 0028 §10): automatic, from the annotation ---

_W = (
    '<r xmlns:nite="http://nite.sourceforge.net/">'
    + "".join(
        f'<w nite:id="w{i}" starttime="{i}.0" endtime="{i}.5">{t}</w>'
        for i, t in enumerate(["we", "we", "should", "go", "uh", "went", "th-", "the", "end"])
    )
    + "</r>"
).encode()
_ONTO = (
    b'<r xmlns:nite="http://nite.sourceforge.net/">'
    b'<t nite:id="rd" name="reparandum Reparandum"/><t nite:id="rs" name="reparans Reparans"/></r>'
)


def _dsfl(kind: str, a: int, b: int, nid: str) -> str:
    return (
        f'<dsfl nite:id="{nid}">'
        f'<nite:pointer role="dsfl-type" href="dsfl-types.xml#id({kind})"/>'
        f'<nite:child href="x.words.xml#id(w{a})..id(w{b})"/></dsfl>'
    )


def _reps(body: str, gap: int = 2):
    items = ami.parse_words(_W)
    order = [i.id for i in items]
    text = {i.id: i.text for i in items}
    xml = f'<r xmlns:nite="http://nite.sourceforge.net/">{body}</r>'.encode()
    dis = ami.parse_disfluency(xml, order, None, ami.load_ontology(_ONTO))
    return ami.repetitions_from_repairs(dis, text, order, gap)


def test_repetition_is_a_reparandum_repeated_at_the_start_of_its_reparans_nested():
    body = (
        '<dsfl nite:id="p1">' + _dsfl("rd", 0, 0, "a") + _dsfl("rs", 1, 2, "b") + "</dsfl>"
        '<dsfl nite:id="p2">' + _dsfl("rd", 3, 3, "c") + _dsfl("rs", 5, 5, "d") + "</dsfl>"
        '<dsfl nite:id="p3">' + _dsfl("rd", 6, 6, "e") + _dsfl("rs", 7, 7, "f") + "</dsfl>"
    )
    reps, counts = _reps(body)
    assert [r.word_ids for r in reps] == [("w0",)]  # "we we": yes; "go uh went": no; "th- the": no
    assert counts["paired_same_parent"] == 3 and counts["repetitions"] == 1
    assert counts["reparanda"] == 3 and counts["reparans"] == 3


def test_repetition_pairing_falls_back_to_the_following_reparans_within_the_gap():
    body = _dsfl("rd", 0, 0, "a") + _dsfl("rs", 1, 1, "b")  # flat, adjacent
    reps, counts = _reps(body)
    assert len(reps) == 1 and counts["paired_following"] == 1
    far = _dsfl("rd", 0, 0, "a") + _dsfl("rs", 8, 8, "b")  # 7 words apart: not paired
    reps, counts = _reps(far)
    assert reps == [] and counts["unpaired"] == 1


def test_repetition_rule_is_case_and_punctuation_blind_but_needs_the_same_words():
    items_text = {"w0": "We,", "w1": "we"}
    dis = [
        ami.Disfluency("dsfl-type=reparandum", ("w0",), "p"),
        ami.Disfluency("dsfl-type=reparans", ("w1",), "p"),
    ]
    reps, _ = ami.repetitions_from_repairs(dis, items_text, ["w0", "w1"], 2)
    assert len(reps) == 1


def test_repetition_rule_is_fixed_in_config_and_used_by_the_run(tmp_path):
    rep = E3["repetitions"]
    assert rep["definition"] == "reparandum_repeated_in_reparans" and rep["max_gap_words"] == 2
    body = (T1 / "colab" / "e3_run.py").read_text(encoding="utf-8")
    assert "ami.repetitions_from_repairs(dis, text, order, gap)" in body
    assert '"repair_pairing": dict(sorted(pairing.items()))' in body


def test_pin_file_names_are_clean_utf8_and_the_run_refuses_unpinned_files():
    # ADR 0028 §11: one PodcastFillers name was stored with "–" mis-decoded as cp1252 ("â€“"),
    # so run b5 found no pin for that episode. Names must be clean UTF-8, and the run stops
    # on a file without a pin instead of using it unchecked.
    for p in sorted((REPO / "configs" / "pins").glob("*.json")):
        text = p.read_bytes().decode("utf-8")
        for bad in ("â€", "Ã", "�"):
            assert bad not in text, (p.name, bad)
    body = (T1 / "colab" / "e3_run.py").read_text(encoding="utf-8")
    assert 'raise ValueError(f"PodcastFillers {ep}.wav has no pin' in body
    assert 'raise ValueError(f"AMI {name} has no pin")' in body
