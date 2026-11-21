#!/usr/bin/env python3
"""Bind an observed official build to a later real runtime inventory.

begin runs immediately BEFORE ./build.sh --release --jobs 8; record-build runs
only AFTER that exact command completes and receives its actual exit code.
finalize additionally requires a successful isaac_runtime_inventory.py report.
This script never builds, starts Kit, or infers compilation from a directory.
Records are supervisor attestations plus checked hashes, not independent proof
of reproducible compilation. No command here grants functional-test credit.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid


COMMIT = "7c206f75bdadd9e05fc457f19863ca4c3f0cb693"
COMMAND = ["./build.sh", "--release", "--jobs", "8"]
EXTENSION = "isaacsim.core.simulation_manager"
FILESYSTEM_TIME_TOLERANCE_SECONDS = 1.0
NATIVE_DIRECTORY = Path("exts") / EXTENSION / "bin"
NATIVE_NAME = re.compile(r"(?:lib)?isaacsim\.core\.simulation_manager\.plugin(?:\.so)?")
STATIC_ARTIFACTS = {
    "launcher": "python.sh", "experience": "apps/isaacsim.exp.full.kit",
    "python_executable": "kit/python/bin/python3", "kit_executable": "kit/kit",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    require(parsed.tzinfo is not None, "Timestamp must include its timezone")
    return parsed.timestamp()


def file_record(path):
    path = Path(path)
    before = path.stat()
    require(path.is_file() and before.st_size > 0, f"Required nonempty file unavailable: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), f"File changed while hashing: {path}")
    return {"sha256": digest.hexdigest(), "size_bytes": after.st_size, "mtime_ns": after.st_mtime_ns}


def source_record(source):
    source = Path(source).resolve()
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True, timeout=30, env=env).strip()
    record = {"root": str(source), "commit": git("rev-parse", "HEAD"),
              "tree": git("rev-parse", "HEAD^{tree}"),
              "dirty": bool(git("status", "--porcelain", "--untracked-files=normal")),
              "build_script_sha256": file_record(source / "build.sh")["sha256"]}
    require(record["commit"] == COMMIT and record["dirty"] is False,
            "Source must be the clean pinned 6.1.0 commit, including untracked files")
    return record


def load(path, kind):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    require(isinstance(value, dict) and type(value.get("schema_version")) is int
            and value["schema_version"] == 1 and value.get("kind") == kind, "Record schema/kind mismatch")
    return value


def output_record(path, value, source, protected=()):
    path = Path(path).resolve()
    require(not path.is_relative_to(Path(source).resolve()), "Output must be outside the official source/build tree")
    require(path != Path(__file__).resolve() and path not in {Path(p).resolve() for p in protected},
            "Output must not overwrite the collector or an input record/log")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental replacement of a previous run's evidence.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def native_artifact(runtime, required=True):
    candidates = [p for p in (runtime / NATIVE_DIRECTORY).glob("*")
                  if NATIVE_NAME.fullmatch(p.name) and p.is_file()]
    # Symlink aliases of one physical binary are harmless; distinct binaries need review.
    unique = {p.resolve(): p for p in sorted(candidates, key=lambda p: len(p.name), reverse=True)}
    if not unique and not required:
        return None
    require(len(unique) == 1, "Expected one actual simulation_manager native plugin; missing/ambiguous binaries refused")
    path = next(iter(unique.values()))
    with path.open("rb") as stream:
        require(stream.read(4) == b"\x7fELF", "Native artifact must be an ELF binary, not a manifest or placeholder")
    return {"role": "isaac_native_plugin", "path": path.relative_to(runtime).as_posix(), **file_record(path)}


def artifacts(runtime):
    return [{"role": role, "path": relative, **file_record(runtime / relative)}
            for role, relative in STATIC_ARTIFACTS.items()] + [native_artifact(runtime)]


def validate_build_log(path):
    record = file_record(path)
    # The pinned repo CLI can return zero even after rejecting unknown arguments.
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
            require(not re.search(r"^\s*error:\s*(unrecognized arguments|argument )", plain, re.I),
                    "Build log contains a CLI argument error despite its reported exit code")
    require(file_record(path)["sha256"] == record["sha256"], "Build log changed during validation")
    return record


def begin(args):
    source = source_record(args.source_root)
    require(args.command == COMMAND, "Only the exact official release/jobs-8 build command is accepted")
    runtime = Path(source["root"]) / "_build/linux-x86_64/release"
    log = args.log.resolve()
    require(not log.exists(), "Begin needs a fresh log path before the actual build starts")
    return {"schema_version": 1, "kind": "isaac_build_start", "run_id": str(uuid.uuid4()),
            "started_at_utc": now(), "source": source, "runtime_root": str(runtime),
            "command": args.command, "working_directory": source["root"], "log_path": str(log),
            "native_before": native_artifact(runtime, required=False),
            "collector_sha256": file_record(__file__)["sha256"],
            "status": "build_not_yet_observed", "functional_validation": "not_verified"}


def record_build(args):
    start = load(args.start, "isaac_build_start")
    require(type(args.exit_code) is int and args.exit_code == 0, "A failed build cannot create a successful build event")
    require(start.get("command") == COMMAND, "Start record does not name the exact release/jobs-8 build")
    source = source_record(start["source"]["root"])
    require(source == start["source"], "Source/build script changed during the build")
    runtime = Path(source["root"]) / "_build/linux-x86_64/release"
    require(Path(start["runtime_root"]).resolve() == runtime.resolve(), "Start runtime path mismatch")
    log = Path(start["log_path"])
    log_record = validate_build_log(log)
    completed = now()
    require(timestamp(start["started_at_utc"]) - FILESYSTEM_TIME_TOLERANCE_SECONDS
            <= log_record["mtime_ns"] / 1e9 <= timestamp(completed) + FILESYSTEM_TIME_TOLERANCE_SECONDS,
            "Build log modification time does not belong to this build interval")
    actual = artifacts(runtime)
    native = next(item for item in actual if item["role"] == "isaac_native_plugin")
    old = start.get("native_before")
    evidence = {"new_or_changed_native_bytes": old is None or old["sha256"] != native["sha256"],
                "native_mtime_within_build": timestamp(start["started_at_utc"]) - FILESYSTEM_TIME_TOLERANCE_SECONDS
                    <= native["mtime_ns"] / 1e9 <= timestamp(completed) + FILESYSTEM_TIME_TOLERANCE_SECONDS,
                "scope": "Observed artifact change/time only; compiler execution is attested by the supervising wrapper and retained build log."}
    return {"schema_version": 1, "kind": "isaac_build_event", "run_id": start["run_id"],
            "source": source, "runtime_root": str(runtime), "artifacts": actual,
            "build": {"command": COMMAND, "flags": COMMAND[1:], "working_directory": source["root"],
                      "exit_code": args.exit_code, "started_at_utc": start["started_at_utc"],
                      "completed_at_utc": completed, "log_path": str(log), "log_sha256": log_record["sha256"],
                      "filesystem_timestamp_tolerance_seconds": FILESYSTEM_TIME_TOLERANCE_SECONDS,
                      "start_record_path": str(args.start.resolve()), "start_record_sha256": file_record(args.start)["sha256"],
                      "native_artifact_evidence": evidence,
                      "observation_scope": "Actual command/result supplied by the supervising build wrapper; not inferred from an archive or build-directory name."},
            "status": "build_event_recorded_runtime_not_verified", "functional_validation": "not_verified"}


def validate_inventory(inventory, event, inventory_path):
    require(isinstance(inventory, dict) and type(inventory.get("schema_version")) is int
            and inventory["schema_version"] == 1 and inventory.get("inventory_kind") == "runtime_snapshot",
            "Expected the real runtime inventory schema")
    require(inventory.get("collection_status") == "collected" and inventory.get("snapshot_status") == "collected"
            and inventory.get("errors") == [], "Runtime collection must have succeeded without errors")
    execution, supervisor = inventory["execution"], inventory["supervisor"]
    require(all(execution.get(k) is True for k in ("kit_start_requested", "simulation_app_constructor_returned",
            "app_ready_observed", "close_requested")), "Runtime startup/readiness/close request evidence missing")
    official_fast_exit = (supervisor.get("official_fast_exit_verified") is True
        and inventory.get("application", {}).get("requested_launch_config", {}).get("fast_shutdown") is True
        and inventory.get("settings", {}).get("/app/fastShutdown") is True)
    require(execution.get("close_returned") is True or official_fast_exit,
            "Neither normal close return nor supervised official fast exit was verified")
    require(type(execution.get("worker_pid")) is int and execution["worker_pid"] > 0, "Runtime worker PID missing")
    require(type(supervisor.get("worker_exit_code")) is int and supervisor["worker_exit_code"] == 0
            and all(supervisor.get(k) is False for k in ("timed_out", "forced_kill", "interrupted")),
            "Runtime process did not exit normally under its supervisor")
    run_id = inventory.get("run_id")
    require(isinstance(run_id, str) and str(uuid.UUID(run_id)) == run_id, "Runtime run ID missing/invalid")
    require(timestamp(inventory["created_at"]) >= timestamp(event["build"]["completed_at_utc"]),
            "Runtime inventory predates this build event")
    require(timestamp(inventory["finished_at"]) >= timestamp(inventory["created_at"]), "Runtime timing is invalid")
    runtime = Path(event["runtime_root"])
    app = inventory["application"]
    experience = next(a for a in event["artifacts"] if a["role"] == "experience")
    require(Path(app["requested_experience"]).resolve() == (runtime / experience["path"]).resolve()
            and app["requested_experience_sha256"] == experience["sha256"], "Runtime selected a different experience/artifact")
    identity = app.get("experience_identity", {})
    require(identity.get("status") == "verified_path_and_hash" and identity.get("basename_matches") is True
            and identity.get("resolved_path_matches") is True and identity.get("sha256_matches") is True
            and identity.get("sha256") == experience["sha256"], "Runtime app path/hash were not actually verified")
    require(app.get("observed", {}).get("version") == "6.1.0", "Observed runtime app is not 6.1.0")
    require(Path(inventory["host"]["python_executable"]).resolve() == (runtime / STATIC_ARTIFACTS["python_executable"]).resolve(),
            "Runtime used a different Python executable")
    observed = [row for row in inventory["extensions"] if row.get("name") == EXTENSION and row.get("enabled") is True]
    require(len(observed) == 1 and Path(observed[0]["path"]).resolve() == (runtime / "exts" / EXTENSION).resolve(),
            "The source-built simulation_manager extension was not observed enabled from this runtime")
    kit_version = inventory.get("kit", {}).get("kit_version")
    require(isinstance(kit_version, str) and bool(kit_version.strip()), "Actual IApp.get_kit_version observation missing")
    require(inventory.get("runtime_inventory", {}).get("complete") is False
            and inventory.get("functional_validation", {}).get("status") == "not_verified"
            and inventory["functional_validation"].get("passed_test_ids") == []
            and inventory["functional_validation"].get("full_parity_claim_allowed") is False,
            "Inventory must not claim functional or full-platform acceptance")
    return {"path": str(inventory_path.resolve()), "sha256": file_record(inventory_path)["sha256"],
            "run_id": run_id, "worker_pid": execution["worker_pid"], "kit_version": kit_version,
            "created_at": inventory["created_at"], "finished_at": inventory["finished_at"],
            "collector_sha256": inventory.get("collector_sha256")}


def finalize(args):
    event_hash = file_record(args.event)["sha256"]
    event = load(args.event, "isaac_build_event")
    source = source_record(event["source"]["root"])
    require(source == event["source"], "Source changed after the build event")
    require(event["build"]["command"] == COMMAND and type(event["build"]["exit_code"]) is int
            and event["build"]["exit_code"] == 0, "Event lacks the successful official build command")
    require(validate_build_log(event["build"]["log_path"])["sha256"] == event["build"]["log_sha256"], "Build log changed")
    start_path = Path(event["build"]["start_record_path"])
    require(file_record(start_path)["sha256"] == event["build"]["start_record_sha256"], "Build start record changed")
    start = load(start_path, "isaac_build_start")
    require(start["run_id"] == event["run_id"] and start["source"] == source and start["command"] == COMMAND,
            "Build start/event identity mismatch")
    runtime = Path(source["root"]) / "_build/linux-x86_64/release"
    require(Path(event["runtime_root"]).resolve() == runtime.resolve(), "Build runtime path mismatch")
    actual = artifacts(runtime)
    signature = lambda rows: sorted((a["role"], a["path"], a["sha256"]) for a in rows)
    require(signature(actual) == signature(event["artifacts"]), "Runtime artifacts changed after build recording")
    inventory_hash = file_record(args.inventory)["sha256"]
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    runtime_record = validate_inventory(inventory, event, args.inventory)
    require(runtime_record["sha256"] == inventory_hash, "Runtime inventory changed during validation")
    require(file_record(args.event)["sha256"] == event_hash, "Build event changed during validation")
    return {"schema_version": 1, "kind": "isaac_build_receipt", "created_at_utc": now(),
            "source": source, "runtime_root": str(runtime), "build": event["build"],
            "kit_version": runtime_record["kit_version"], "artifacts": actual,
            "build_event": {"path": str(args.event.resolve()), "sha256": event_hash, "run_id": event["run_id"]},
            "runtime_inventory": runtime_record, "functional_validation": {"status": "not_verified", "passed_test_ids": [],
                "full_parity_claim_allowed": False},
            "trust_scope": "Supervising wrapper's observed build, clean pinned checkout, unchanged log/artifacts, and actual runtime snapshot; not independently reproducible-build proof or functional acceptance."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="action", required=True)
    first = modes.add_parser("begin")
    first.add_argument("--source-root", required=True, type=Path)
    first.add_argument("--log", required=True, type=Path)
    first.add_argument("--output", required=True, type=Path)
    first.add_argument("command", nargs=argparse.REMAINDER, help="-- ./build.sh --release --jobs 8")
    end = modes.add_parser("record-build")
    end.add_argument("--start", required=True, type=Path)
    end.add_argument("--exit-code", required=True, type=int)
    end.add_argument("--output", required=True, type=Path)
    final = modes.add_parser("finalize")
    final.add_argument("--event", required=True, type=Path)
    final.add_argument("--inventory", required=True, type=Path)
    final.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "begin" and args.command[:1] == ["--"]:
            args.command = args.command[1:]
        result = {"begin": begin, "record-build": record_build, "finalize": finalize}[args.action](args)
        protected = [getattr(args, key) for key in ("log", "start", "event", "inventory") if hasattr(args, key)]
        output_record(args.output, result, result["source"]["root"], protected)
        print(json.dumps({"output": str(args.output.resolve()), "kind": result["kind"], "functional_validation": "not_verified"}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(json.dumps({"status": "refused", "error": str(exc), "exception_type": type(exc).__name__}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
