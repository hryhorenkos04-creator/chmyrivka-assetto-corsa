"""Build the whole lap with one command, and check it before ksEditor.

The export takes a dozen switches and four of them are load-bearing: get
--merge, --flip, --scale or --start-station wrong and the track does not
drive at all. Typing them out each time is how a build goes wrong
quietly, so they live here instead.

  python3 ac_export/build_lap.py --scene <blend> --out <folder>

It runs the export, then verify_build on what came out, then writes the
AI line. Nothing here needs Blender's Python: it drives blender itself,
and falls back to this interpreter when bpy is importable, which is how
the containerised builds run.
"""
from pathlib import Path
import argparse, subprocess, sys

HERE = Path(__file__).resolve().parent

# What a working build of this track takes. Everything else has a sane
# default in the export; these either cannot be defaulted (the start is a
# property of this circuit) or would silently produce a track nobody can
# drive.
SETTINGS = [
    '--mode', 'lap',
    '--chunks', '8',
    '--balance',
    '--corridor', '45',
    '--road-corridor', '800',
    '--backdrop-corridor', '160',
    '--drop-bevel-beyond', '12',
    '--trunk-ratio', '0.2',
    '--conifer-from', '2300',
    '--conifer-to', '3950',
    '--merge',
    '--relief',
    '--sectors', '3',
    '--flip',
    '--start-station', '3950',
    '--scale', '0.01',
]


def run(command, label):
    print('\n--- %s ---' % label)
    print(' '.join(str(c) for c in command[:3]) + ' ...')
    result = subprocess.run([str(c) for c in command])
    if result.returncode:
        sys.exit('%s не отработал, код %d' % (label, result.returncode))


def export_command(a):
    script = HERE / 'export_ac_v17.py'
    settings = list(SETTINGS)
    if a.dry_run:
        settings.append('--dry-run')
    if a.normals_gl:
        settings.append('--normals-gl')
    tail = ['--scene', a.scene, '--output', str(Path(a.out) / 'chmyrivka.fbx')]
    if a.racing_line:
        tail += ['--racing-line', a.racing_line]
    if a.blender:
        # Blender wants its own arguments first and the script's after --.
        return [a.blender, '--background', '--python', script, '--'] + settings + tail
    return [sys.executable, '-I', '-u', script] + settings + tail


def main(a):
    out = Path(a.out).resolve()
    run(export_command(a), 'экспорт')
    if a.dry_run:
        print('\nприкидка готова, файлы не писались')
        return
    run([sys.executable, '-I', HERE / 'verify_build.py', out], 'проверка сборки')
    if a.racing_line:
        run([sys.executable, '-I', HERE / 'make_ai_line.py',
             '--racing-line', a.racing_line, '--out', out], 'AI-линия')
    print('\nГОТОВО. Дальше ksEditor: для каждого FBX по очереди')
    print('  File > Restart Editor, File > Open, File > Load Persistence, File > Save')
    print('Первым собери chmyrivka_s1.fbx и проверь, принял ли редактор шейдер.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--racing-line', dest='racing_line', default=None)
    p.add_argument('--blender', default=None,
                   help='path to blender.exe; omit to use this interpreter, '
                        'which needs bpy importable')
    p.add_argument('--dry-run', action='store_true', dest='dry_run',
                   help='count the chunks without writing anything')
    p.add_argument('--normals-gl', action='store_true', dest='normals_gl',
                   help='pass through when relief comes out inside out')
    main(p.parse_args())
