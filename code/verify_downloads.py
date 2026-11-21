"""Check SHA-256 and file size of the complete numerical download set."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    index = Path(__file__).resolve().parents[1] / 'data/raw-data-assets.json'
    failed = []
    for asset in json.loads(index.read_text(encoding='utf-8')):
        archive = args.directory / asset['file']
        if archive.is_file() and archive.stat().st_size == asset['bytes']:
            with archive.open('rb') as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() == asset['sha256']:
                print(f'OK {archive.name}')
                continue
        failed.append(archive.name)
        print(f'Missing or invalid: {archive.name}')
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
