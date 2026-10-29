"""Atomically package the validated v17 scene and actual-render gallery.

No private reference images, old native scenes, transfer scene or bake output
is distributed here. Historical transfer documentation remains under sources/.
"""
from pathlib import Path
import hashlib
import json
import zipfile
from PIL import Image

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = '8e4f2c9c3e1b37de42d9d99edf115a898fa72cb76936dce19020bb02f23b0e00'
GALLERY_FOLDERS = ('screenshots_v17', 'screenshots_extra_v17',
                   'screenshots_before_v16', 'screenshots_extra_before_v16', 'comparisons')
SUFFIXES = {'.json', '.md', '.py', '.png', '.jpg', '.jpeg', '.html', '.txt', '.csv'}


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_release_file(path):
    relative = path.relative_to(ROOT)
    if not path.is_file() or path.is_symlink():
        return False
    if any(part in ('__pycache__', 'private_references', 'drafts', 'transfer', 'qa') for part in relative.parts):
        return False
    if any(token in path.stem.lower() for token in ('draft_', 'first_pass', '_smoke', '_inspect')):
        return False
    return path.suffix.lower() in SUFFIXES


def main():
    native = ROOT / 'chmyrivka_visual_v17.blend'
    native_sha = sha(native)
    validation = json.loads((ROOT / 'validation_v17.json').read_text())
    assert validation['passed'] and not validation['errors']
    assert validation['candidate_sha256'] == native_sha
    assert validation['source_sha256_after'] == SOURCE_SHA
    build = json.loads((ROOT / 'build_info_v17.json').read_text())
    assert build['native_sha256'] == native_sha and build['source_unchanged']
    assert build['source_sha256'] == SOURCE_SHA
    for folder, count, expected in [('screenshots_v17', 6, native_sha),
                                    ('screenshots_extra_v17', 5, native_sha),
                                    ('screenshots_before_v16', 6, SOURCE_SHA),
                                    ('screenshots_extra_before_v16', 5, SOURCE_SHA)]:
        rows = json.loads((ROOT / folder / 'render_manifest.json').read_text())
        assert len(rows) == count and len({r['camera'] for r in rows}) == count
        for row in rows:
            assert Path(row['file']).name == row['file']
            assert (ROOT / folder / row['file']).is_file() and row['native_sha256'] == expected
            assert row['resolution'] == [1600, 900] and row['samples'] == 48
            with Image.open(ROOT / folder / row['file']) as image:
                assert image.size == (1600, 900)
                image.verify()
    gallery_proof = json.loads((ROOT / 'comparisons/gallery_manifest_v17.json').read_text())
    assert gallery_proof['native_sha256'] == native_sha and gallery_proof['source_sha256'] == SOURCE_SHA
    assert gallery_proof['current_views'] == gallery_proof['baseline_views'] == 11
    for path in ROOT.glob('*.md'):
        contents = path.read_text(encoding='utf8')
        for marker in ('TRANSFER_PENDING', 'GAL_PENDING', 'V17_PENDING', 'AUDIT_RESULT_PENDING', 'FINAL_GALLERY_PENDING'):
            assert marker not in contents, (path, marker)
    for required in ('README_V17_RU.md', 'CHMYRIVKA_STATUS_RU.md', 'CHMYRIVKA_ROADMAP_RU.md',
                     'VISUAL_REVIEW_V17_RU.md', 'VALIDATION_V17_RU.md'):
        assert (ROOT / required).is_file(), required
    assert 'v17' in (ROOT / 'CHMYRIVKA_STATUS_RU.md').read_text(), 'Status is still from an older release'
    files = {native, ROOT / 'validation_v17.json', ROOT / 'build_info_v17.json'}
    for pattern in ('*.md', '*.py'):
        files.update(p for p in ROOT.glob(pattern) if safe_release_file(p))
    for folder in ('data', 'source_data', 'dependencies', 'sources', 'tools', *GALLERY_FOLDERS):
        files.update(p for p in (ROOT / folder).rglob('*') if safe_release_file(p))
    for name in ('gallery_v17.html', 'contact_sheet_v17.png', 'gallery_manifest_v17.json',
                 'comparison_driver_v17.png', 'comparison_architecture_v17.png', 'comparison_layout_v17.png'):
        assert ROOT / 'comparisons' / name in files
    for path in files:
        if path.suffix.lower() == '.png':
            with Image.open(path) as image:
                image.verify()
    release = {'version': 'v17', 'native_sha256': native_sha, 'source_sha256': SOURCE_SHA,
               'assetto_corsa_tested': False, 'transfer_bake_created_in_v17': False,
               'current_views': 11, 'baseline_views': 11,
               'files': [{'path': str(path.relative_to(ROOT)), 'bytes': path.stat().st_size,
                          'sha256': sha(path)} for path in sorted(files)]}
    manifest = ROOT / 'RELEASE_MANIFEST_V17.json'
    manifest.write_text(json.dumps(release, ensure_ascii=False, indent=2), encoding='utf8')
    files.add(manifest)
    gallery = {path for path in files if path.relative_to(ROOT).parts[0] in GALLERY_FOLDERS}
    gallery.update(ROOT / name for name in ('README_V17_RU.md', 'CHMYRIVKA_STATUS_RU.md',
                                           'VISUAL_REVIEW_V17_RU.md'))
    reports = []
    for name, selected in [('chmyrivka_visual_v17.zip', files), ('chmyrivka_screenshots_v17.zip', gallery)]:
        target = ROOT.parent / name
        temporary = target.with_name(target.name + '.tmp')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
            for path in sorted(selected):
                relative = path.relative_to(ROOT)
                assert 'private_references' not in relative.parts and 'transfer' not in relative.parts
                assert not path.name.startswith('draft_')
                archive.write(path, 'chmyrivka_v17/' + str(relative),
                              compress_type=zipfile.ZIP_STORED if path.suffix.lower() in ('.blend', '.png', '.jpg', '.jpeg') else zipfile.ZIP_DEFLATED)
        with zipfile.ZipFile(temporary) as archive:
            assert len(archive.infolist()) == len(selected) and archive.testzip() is None
        assert temporary.stat().st_size < 500_000_000, 'Split an oversized release before delivery'
        temporary.replace(target)
        reports.append({'file': str(target), 'bytes': target.stat().st_size,
                        'files': len(selected), 'sha256': sha(target)})
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    main()
