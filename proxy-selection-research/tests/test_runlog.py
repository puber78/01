import csv
import json

from proxy_selection.runlog import CommandLedger, RunLogger


def test_run_logger_writes_jsonl_events(tmp_path):
    log = RunLogger.start(tmp_path, note="unit-test")
    log.input_hashes({"a.jpg": "d" * 64})
    log.failure("scoring", "boom")
    log.finish("PASS", rows=3)
    lines = log.path.read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(x) for x in lines]
    assert [e["event"] for e in events] == ["run_start", "input_hashes", "failure", "run_end"]
    assert events[0]["environment"]["python"]
    assert events[-1]["status"] == "PASS"


def test_command_ledger_append(tmp_path):
    led = CommandLedger(tmp_path)
    led.append("python -m pytest", exit_code=0, log_path="log.txt", note="tests")
    rows = list(csv.reader(led.path.open(encoding="utf-8")))
    assert rows[0][0] == "recorded_at"
    assert rows[1][4] == "python -m pytest"
    assert rows[1][5] == "0"
