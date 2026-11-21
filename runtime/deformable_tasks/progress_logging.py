"""Persist progress before best-effort terminal output.

A disconnected SSH reader must not abort a physics step. Disk failures remain
fatal: losing durable evidence is not a successful experiment.
"""
import json
import os
import sys


def emit_json(payload, path):
    line = json.dumps(payload) + '\n'
    with open(path, 'a', encoding='utf-8') as log:
        log.write(line)
        log.flush()
    try:
        sys.stdout.write(line)
        sys.stdout.flush()
    except BrokenPipeError:
        # Prevent a second broken pipe from interpreter-exit flushing.
        sys.stdout = open(os.devnull, 'w', encoding='utf-8')
