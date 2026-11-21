"""Check SHA-256 and file size of the complete numerical download set."""
import argparse
import hashlib
import json
from pathlib import Path


def matches(path, record):
    if not path.is_file() or path.stat().st_size != record['bytes']:
        return False
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest() == record['sha256']


def assemble(directory, record):
    archive = directory / record['file']
    temporary = archive.with_suffix('.zip.partial')
    total = hashlib.sha256()
    with temporary.open('wb') as out:
        for part in record['parts']:
            source = directory / part['file']
            if not source.is_file() or source.stat().st_size != part['bytes']:
                raise ValueError(f'Missing or incomplete part: {source.name}')
            digest = hashlib.sha256()
            with source.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    out.write(chunk)
                    total.update(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != part['sha256']:
                raise ValueError(f'Incorrect checksum: {source.name}')
    if total.hexdigest() != record['sha256']:
        raise ValueError(f'Incorrect assembled checksum: {archive.name}')
    temporary.replace(archive)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--assemble', action='store_true', help='Join verified numbered parts into their original ZIPs.')
    args = parser.parse_args()
    index = Path(__file__).resolve().parents[1] / 'data/raw-data-assets.json'
    failed = []
    for asset in json.loads(index.read_text(encoding='utf-8')):
        archive = args.directory / asset['file']
        if matches(archive, asset):
            print(f'OK {archive.name}')
            continue
        if args.assemble and 'parts' in asset:
            try:
                assemble(args.directory, asset)
                print(f'Assembled and verified {archive.name}')
                continue
            except ValueError as error:
                print(error)
        failed.append(archive.name)
        print(f'Missing or invalid: {archive.name}')
    return int(bool(failed))


if __name__ == '__main__':
    raise SystemExit(main())
