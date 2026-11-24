"""Check a finished lap build before anyone opens ksEditor.

Nine passes through the editor is half an hour of clicking, and every
fault found afterwards costs that twice. Everything here is cheap and
needs no Blender: it reads the FBX sizes, the persistence files and the
report the export already wrote.

  python3 ac_export/verify_build.py <build directory>

Exit status is 0 when nothing is wrong, 1 otherwise, so it can gate a
build script. What it cannot check is whether ksEditor accepts a shader
name or whether the relief points the right way; those need the editor
and the game.
"""
from pathlib import Path
import json, re, sys

# ksEditor's 16-bit index buffer. A mesh over it is dropped on import
# with 'The mesh will not be loaded' and its scenery silently vanishes.
VERTEX_CEILING = 65536

# Measured on this track: a chunk of 2.24M triangles builds, one of 2.68M
# dies with AccessViolationException in ksGraphics.render. The warning sits
# below the lower figure rather than between them.
TRIANGLE_WARNING = 2_000_000

KNOWN_SHADERS = {'ksPerPixel', 'ksPerPixelAT', 'ksPerPixelNM', 'ksPerPixelAT_NM'}


class Report:
    def __init__(self):
        self.failures, self.warnings, self.notes = [], [], []

    def fail(self, text):
        self.failures.append(text)

    def warn(self, text):
        self.warnings.append(text)

    def note(self, text):
        self.notes.append(text)

    def show(self):
        for line in self.notes:
            print('   ' + line)
        for line in self.warnings:
            print('ВНИМАНИЕ: ' + line)
        for line in self.failures:
            print('ОШИБКА:   ' + line)
        print()
        if self.failures:
            print('НЕ ГОДИТСЯ: %d ошибок, %d предупреждений'
                  % (len(self.failures), len(self.warnings)))
        elif self.warnings:
            print('ГОДИТСЯ, но с оговорками: %d предупреждений' % len(self.warnings))
        else:
            print('ВСЁ ЧИСТО')
        return 1 if self.failures else 0


def check_pairs(build, report):
    """Every FBX needs the persistence file that fills its materials."""
    models = sorted(build.glob('*.fbx'))
    if not models:
        report.fail('в папке нет ни одного .fbx')
        return []
    for fbx in models:
        ini = fbx.with_name(fbx.name + '.ini')
        if not ini.is_file():
            report.fail('%s без persistence — его материалы придётся ставить руками'
                        % fbx.name)
        elif fbx.stat().st_size == 0:
            report.fail('%s пустой' % fbx.name)
    report.note('файлов FBX: %d, из них с persistence: %d'
                % (len(models),
                   sum(1 for f in models if f.with_name(f.name + '.ini').is_file())))
    return models


def check_textures(build, report):
    """A texture the persistence names but nobody wrote leaves txDiffuse NULL,
    and ksEditor then refuses to save the KN5 at all."""
    folder = build / 'texture'
    if not folder.is_dir():
        report.fail('нет папки texture — ksEditor ищет текстуры только в ней')
        return
    have = {p.name for p in folder.iterdir() if p.is_file()}
    named, missing = set(), set()
    for ini in sorted(build.glob('*.fbx.ini')):
        text = ini.read_text(encoding='utf8', errors='replace')
        for texture in re.findall(r'RES_\d+_TEXTURE=(.+)', text):
            texture = texture.strip()
            named.add(texture)
            if texture not in have:
                missing.add('%s -> %s' % (ini.name, texture))
    for item in sorted(missing)[:10]:
        report.fail('текстуры нет на диске: ' + item)
    if len(missing) > 10:
        report.fail('...и ещё %d таких' % (len(missing) - 10))
    report.note('текстур названо %d, лежит на диске %d, битых ссылок %d'
                % (len(named), len(have), len(missing)))
    unused = have - named
    if unused:
        report.note('не упомянуто ни одним материалом: %d файлов' % len(unused))


def check_shaders(build, report):
    """An unknown shader name is the one fault that only shows up in the
    editor, so it is worth naming here even though we cannot test it."""
    found = {}
    for ini in sorted(build.glob('*.fbx.ini')):
        text = ini.read_text(encoding='utf8', errors='replace')
        for shader in re.findall(r'SHADER=(\S+)', text):
            found[shader.strip()] = found.get(shader.strip(), 0) + 1
    for shader, count in sorted(found.items()):
        if shader not in KNOWN_SHADERS:
            report.fail('шейдер вне известного набора: %s (%d материалов)'
                        % (shader, count))
    report.note('шейдеры: ' + ', '.join('%s=%d' % kv for kv in sorted(found.items())))
    if 'ksPerPixelNM' in found:
        report.note('рельеф включён — проверь в ksEditor первым делом один кусок: '
                    'имя ksPerPixelNM официально не задокументировано')


def check_models_ini(build, report, models):
    """AC loads what models.ini lists and nothing else, so a name that does
    not match a KN5 leaves that part of the lap simply absent."""
    path = build / 'models.ini'
    if not path.is_file():
        report.fail('нет models.ini — AC загрузит только одну модель из всех')
        return
    listed = [m.strip() for m in
              re.findall(r'FILE=(.+)', path.read_text(encoding='utf8'))]
    expected = [f.with_suffix('.kn5').name for f in models]
    for name in listed:
        if name not in expected:
            report.fail('models.ini ссылается на %s, а такого FBX нет' % name)
    for name in expected:
        if name not in listed:
            report.fail('%s не перечислен в models.ini и в игру не попадёт' % name)
    first = build.name if (build / (build.name + '.fbx')).is_file() else None
    if listed and first and listed[0] != first + '.kn5':
        report.warn('первой в models.ini стоит %s; AC читает пустышки AC_START '
                    'только из первой модели' % listed[0])
    report.note('в models.ini моделей: %d' % len(listed))


def check_report(build, report):
    """The export's own numbers: split parts against the vertex ceiling, and
    chunk weights against what the editor has survived here."""
    # --report lets the caller name it anything, so the report is found by
    # what it contains rather than by what it is called.
    data = None
    candidates = [build / 'export_report.json'] + sorted(build.glob('*.json'))
    for path in candidates:
        if not path.is_file() or path.name == 'band_edges.json':
            continue
        try:
            loaded = json.loads(path.read_text(encoding='utf8'))
        except (ValueError, OSError):
            continue
        if isinstance(loaded, dict) and 'chunks' in loaded:
            data = loaded
            break
    if data is None:
        report.warn('отчёта экспорта нет — вес кусков и разрезанные сетки не проверены')
        return
    chunks = data.get('chunks') or []
    worst_part = 0
    for chunk in chunks:
        name = Path(chunk.get('fbx') or '?').name
        triangles = chunk.get('exported_triangles', 0)
        if triangles > TRIANGLE_WARNING:
            report.warn('%s несёт %d треугольников — близко к тому, на чём '
                        'ksEditor падал (2.68 млн)' % (name, triangles))
        for split in chunk.get('split_meshes') or []:
            parts = max(1, split.get('parts', 1))
            worst_part = max(worst_part, split.get('corners', 0) // parts)
    if worst_part > VERTEX_CEILING:
        report.fail('после разрезания часть сетки держит %d углов при потолке %d'
                    % (worst_part, VERTEX_CEILING))
    elif worst_part:
        report.note('худшая часть разрезанной сетки: %d углов из %d'
                    % (worst_part, VERTEX_CEILING))
    total = sum(c.get('exported_triangles', 0) for c in chunks)
    report.note('кусков %d, треугольников всего %d' % (len(chunks), total))
    if data.get('conifers_swapped'):
        report.note('порода заменена у %d деревьев' % data['conifers_swapped'])


def check_bands(build, report):
    """Band edges have to cover the lap once: a gap is a hole in the scenery
    and an overlap is two copies fighting over the same ground."""
    path = build / 'band_edges.json'
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding='utf8'))
    edges = data.get('edges') or []
    if len(edges) < 2:
        report.fail('band_edges.json без границ')
        return
    if any(b <= a for a, b in zip(edges, edges[1:])):
        report.fail('границы полос не возрастают: %s'
                    % [round(e) for e in edges])
    spans = [b - a for a, b in zip(edges, edges[1:])]
    report.note('полос %d, от %.0f до %.0f м, границы %s'
                % (len(spans), min(spans), max(spans),
                   [round(e) for e in edges]))
    counts = data.get('triangles_per_chunk') or []
    if counts:
        report.note('по весу: от %d до %d, разброс %.0f%%'
                    % (min(counts), max(counts),
                       100.0 * (max(counts) - min(counts)) / max(max(counts), 1)))


def main(build):
    build = Path(build).resolve()
    if not build.is_dir():
        print('нет такой папки: %s' % build)
        return 1
    print('Проверка сборки: %s\n' % build)
    report = Report()
    models = check_pairs(build, report)
    check_textures(build, report)
    check_shaders(build, report)
    check_models_ini(build, report, models)
    check_report(build, report)
    check_bands(build, report)
    return report.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
