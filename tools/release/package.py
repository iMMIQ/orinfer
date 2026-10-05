"""Build and verify a release from a clean, exact Git revision; publish only with --publish."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.model.package import install
from tools.model.publication import file_hash


def command(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def archive(directory, output, epoch):
    with output.open('xb') as destination, gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=epoch) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as tar:
            for path in [directory, *sorted(directory.rglob('*'))]:
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(directory.parent)))
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = epoch
                if info.isfile():
                    with path.open('rb') as stream:
                        tar.addfile(info, stream)
                else:
                    tar.addfile(info)


def verify(directory):
    checksums = directory / 'SHA256SUMS'
    for line in checksums.read_text().splitlines():
        digest, name = line.split('  ', 1)
        if Path(name).name != name or file_hash(directory / name) != digest:
            raise ValueError(f'Release checksum mismatch: {name}')


def notices(metadata):
    text = ['License notices for the locked Cargo dependency graph.\n']
    for package in sorted(metadata['packages'], key=lambda p: (p['name'], p['version'])):
        text.append(f"\n{'='*72}\n{package['name']} {package['version']}\nLicense: {package.get('license')}\nRepository: {package.get('repository')}\nAuthors: {', '.join(package['authors'])}\n")
        root = Path(package['manifest_path']).parent
        paths = {p for p in root.iterdir() if p.is_file() and p.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE'))}
        if package.get('license_file'):
            paths.add(root / package['license_file'])
        for path in sorted(paths):
            text.append(f'\n--- {path.name} ---\n{path.read_text(errors="replace")}\n')
    return ''.join(text)


def build(args):
    revision = command('git', 'rev-parse', 'HEAD')
    if command('git', 'status', '--porcelain', '--untracked-files=normal'):
        raise ValueError('Release requires a clean worktree, including new source files')
    metadata = json.loads(command('cargo', 'metadata', '--locked', '--offline', '--format-version', '1'))
    version = next(p['version'] for p in metadata['packages'] if p['name'] == 'orinfer-cli')
    tag = 'v' + version
    if args.version != tag:
        raise ValueError(f'Tag must match workspace version: {tag}')
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(['cargo', 'build', '--release', '--locked', '--offline', '-p', 'orinfer-cli'], cwd=ROOT, check=True)
    binary = ROOT / 'target/release/orinfer'
    elf = binary.read_bytes()[:20]
    if elf[:4] != b'\x7fELF' or int.from_bytes(elf[18:20], 'little') != 183:
        raise ValueError('Binary must target Linux aarch64')
    if command(str(binary), '--version') != f'orinfer {version}':
        raise ValueError('Binary version mismatch')
    if command('git', 'rev-parse', 'HEAD') != revision or command('git', 'status', '--porcelain'):
        raise ValueError('Source changed during build')
    operators = args.operators.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix='orinfer-release-') as temp:
        package_id = install(operators, Path(temp)/'operator-cache')
        bundle = Path(temp) / f'orinfer-{tag}-linux-aarch64'
        (bundle/'bin').mkdir(parents=True)
        shutil.copy2(binary, bundle/'bin/orinfer')
        for name in ('README.md', 'LICENSE', 'COPYING', 'COPYING.LESSER', 'THIRD_PARTY_NOTICES.md'):
            if (ROOT/name).exists():
                shutil.copyfile(ROOT/name, bundle/name)
        for name in ('tools/model/prepare.py', 'tools/model/package.py', 'tools/model/publication.py', 'configs/architecture-contract.json', 'tools/build/cpu-requirements.txt'):
            (bundle/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, bundle/name)
        for name in ('docs/serving.md', 'kernels/README.md', 'examples/opencode.json',
                     'tools/build/README.md', 'tools/model/README.md', 'tools/api/README.md',
                     'tools/vision/README.md', 'tools/bench/README.md', 'tools/eval/README.md'):
            (bundle/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/name, bundle/name)
        source = bundle/'source.tar.gz'
        subprocess.run(['git', 'archive', '--format=tar.gz', f'--output={source}', revision], cwd=ROOT, check=True)
        (bundle/'SOURCE.txt').write_text(f'Revision: {revision}\nSource: source.tar.gz\nBuild: cargo build --release --locked --offline -p orinfer-cli\n')
        (bundle/'THIRD_PARTY_LICENSES.txt').write_text(notices(metadata))
        (bundle/'RELEASE.json').write_text(json.dumps(dict(tag=tag, revision=revision, operator_package=package_id,
                                                        binary_sha256=file_hash(binary)), indent=2)+'\n')
        epoch = int(command('git', 'show', '-s', '--format=%ct', revision))
        archive(bundle, output/f'{bundle.name}.tar.gz', epoch)
    operator_name = f'orinfer-{tag}-qwen3_5-27b-int8-quality-sm87-operators.tar.gz'
    shutil.copyfile(operators, output/operator_name)
    assets = sorted(output.glob('*.tar.gz'))
    (output/'SHA256SUMS').write_text(''.join(f'{file_hash(path)}  {path.name}\n' for path in assets))
    verify(output)
    if args.publish:
        if not args.notes or not args.notes.is_file():
            raise ValueError('--publish requires --notes FILE')
        subprocess.run(['git', 'rev-parse', '--verify', f'refs/tags/{tag}'], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        if command('git', 'rev-parse', f'{tag}^{{}}') != revision:
            raise ValueError('Annotated release tag differs from packaged revision')
        subprocess.run(['git', 'push', 'origin', f'refs/tags/{tag}'], cwd=ROOT, check=True)
        subprocess.run(['gh', 'release', 'create', tag, *map(str, assets), str(output/'SHA256SUMS'),
                        '--title', tag, '--notes-file', str(args.notes)], cwd=ROOT, check=True)
        with tempfile.TemporaryDirectory(prefix='orinfer-release-download-') as temp:
            subprocess.run(['gh', 'release', 'download', tag, '--dir', temp], cwd=ROOT, check=True)
            verify(Path(temp))
            for path in [*assets, output/'SHA256SUMS']:
                if file_hash(Path(temp)/path.name) != file_hash(path):
                    raise ValueError(f'Uploaded asset differs from local build: {path.name}')
    return dict(tag=tag, revision=revision, output=str(output), verified=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', required=True)
    parser.add_argument('--operators', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--notes', type=Path)
    parser.add_argument('--publish', action='store_true')
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == '__main__':
    main()
