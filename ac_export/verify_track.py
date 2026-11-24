"""Check an installed track folder, the way the game will read it.

verify_build.py looks at what the export produced. This looks at what is
actually in content/tracks/<name>, which is a different question and the
one that has cost this project the most time: a map.png of 1200x1600 sat
beside a map.ini describing 1620x2170 for a day, the marker went off the
road, and the search went looking for a coordinate bug that was not
there. A mismatched pair reads as a plausible failure of everything else.

  python3 ac_export/verify_track.py "<...>/content/tracks/chmyrivka"

Exit status is 0 when nothing is wrong, 1 otherwise. Pillow is used for
the map only if it is installed; without it the image is read far enough
to get its size, which is all this needs.
"""
from pathlib import Path
import json, re, struct, sys

AI_VERSION = 7
AI_EXTRA_FIELDS = 18
# Every Kunos circuit ships SCALE_FACTOR=1: the map is drawn at one unit
# per metre and the offsets move the world into frame. A build that
# scales instead stopped the map drawing at all.
EXPECTED_SCALE = 1.0


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
            print('ТРАССА НЕИСПРАВНА: %d ошибок, %d предупреждений'
                  % (len(self.failures), len(self.warnings)))
        elif self.warnings:
            print('ТРАССА РАБОТОСПОСОБНА, но с оговорками: %d' % len(self.warnings))
        else:
            print('ТРАССА В ПОРЯДКЕ')
        return 1 if self.failures else 0


def png_size(path):
    """Width and height from a PNG header, without needing Pillow."""
    with open(path, 'rb') as handle:
        head = handle.read(24)
    if len(head) < 24 or head[:8] != b'\x89PNG\r\n\x1a\n':
        return None
    return struct.unpack('>2I', head[16:24])


def read_ini(path):
    values = {}
    for line in path.read_text(encoding='utf8', errors='replace').splitlines():
        if '=' in line and not line.strip().startswith(';'):
            key, _, value = line.partition('=')
            values[key.strip()] = value.strip()
    return values


def check_models(track, report):
    """What AC loads is what models.ini names, and nothing else."""
    kn5 = sorted(track.glob('*.kn5'))
    models = track / 'models.ini'
    if not models.is_file():
        if len(kn5) > 1:
            report.fail('нет models.ini, а KN5 несколько — игра возьмёт только '
                        'тот, что назван как папка, остальные не загрузятся')
        return
    listed = [n.strip() for n in
              re.findall(r'FILE=(.+)', models.read_text(encoding='utf8'))]
    present = {p.name for p in kn5}
    for name in listed:
        if name not in present:
            report.fail('models.ini ссылается на %s, а файла нет' % name)
    for name in sorted(present - set(listed)):
        report.fail('%s лежит в папке, но не перечислен в models.ini и '
                    'в игру не попадёт' % name)
    if listed and listed[0] != track.name + '.kn5':
        report.fail('первой в models.ini стоит %s, а не %s.kn5; AC читает '
                    'пустышки AC_START только из первой модели'
                    % (listed[0], track.name))
    for path in kn5:
        if path.stat().st_size < 1024:
            report.fail('%s подозрительно мал: %d байт'
                        % (path.name, path.stat().st_size))
    report.note('моделей %d, суммарно %d МБ'
                % (len(kn5), sum(p.stat().st_size for p in kn5) // 1_000_000))
    if not (track / 'texture').is_dir():
        report.warn('нет папки texture — она нужна ksEditor, но не игре: '
                    'в KN5 текстуры уже зашиты')


def check_map(track, report):
    """The pair that has to agree, and the one that silently did not."""
    image = track / 'map.png'
    config = track / 'data' / 'map.ini'
    if not image.is_file() or not config.is_file():
        report.fail('нет %s — мини-карты не будет'
                    % ('map.png' if not image.is_file() else 'data/map.ini'))
        return
    size = png_size(image)
    values = read_ini(config)
    try:
        width = float(values.get('WIDTH', 0))
        height = float(values.get('HEIGHT', 0))
        scale = float(values.get('SCALE_FACTOR', 0))
        cap = float(values.get('MAX_SIZE', 0))
    except ValueError:
        report.fail('data/map.ini содержит нечисловые значения')
        return
    if size is None:
        report.warn('map.png не читается как PNG — размер не проверен')
    elif abs(size[0] - width) > 1 or abs(size[1] - height) > 1:
        report.fail('map.png %dx%d, а map.ini описывает %gx%g — маркер машины '
                    'уедет с трассы; ставить их надо только вместе'
                    % (size[0], size[1], width, height))
    else:
        report.note('карта согласована: %dx%d' % size)
    if abs(scale - EXPECTED_SCALE) > 1e-6:
        report.warn('SCALE_FACTOR=%g; у всех трасс Kunos он равен 1, и '
                    'другое значение однажды привело к тому, что карта '
                    'перестала рисоваться' % scale)
    if cap and max(width, height) > cap:
        report.fail('MAX_SIZE=%g меньше стороны карты %g — AC покажет только '
                    'кусок вместо круга' % (cap, max(width, height)))


def check_surfaces(track, report):
    """Without this file every surface falls back to the game's defaults,
    and grass grips like asphalt."""
    path = track / 'data' / 'surfaces.ini'
    if not path.is_file():
        report.fail('нет data/surfaces.ini — физика покрытий пойдёт по '
                    'умолчаниям игры, и трава будет держать как асфальт')
        return
    text = path.read_text(encoding='utf8', errors='replace')
    keys = set(re.findall(r'^KEY=(\S+)', text, re.M))
    for needed in ('ROAD',):
        if needed not in keys:
            report.fail('в surfaces.ini нет KEY=%s, а меши названы под него'
                        % needed)
    report.note('покрытий описано %d: %s' % (len(keys), ', '.join(sorted(keys))))
    valid = re.findall(r'KEY=(\S+)(?:(?!\[SURFACE).)*?IS_VALID_TRACK=(\d)',
                       text, re.S)
    off = [k for k, v in valid if v == '1' and k not in ('ROAD', 'KERB')]
    if off:
        report.warn('засчитываются как трасса, кроме дороги и поребриков: %s'
                    % ', '.join(off))


def check_ai(track, report):
    """A line the game cannot parse is the same as no line, except it
    fails later and less clearly."""
    path = track / 'ai' / 'fast_lane.ai'
    if not path.is_file():
        report.warn('нет ai/fast_lane.ai — соперников в заезд добавить нельзя')
        return
    data = path.read_bytes()
    if len(data) < 16:
        report.fail('ai/fast_lane.ai обрывается на заголовке')
        return
    version, count, _, _ = struct.unpack_from('<4i', data, 0)
    if version != AI_VERSION:
        report.fail('ai/fast_lane.ai версии %d, ожидается %d'
                    % (version, AI_VERSION))
    wanted = 16 + count * 20 + count * AI_EXTRA_FIELDS * 4
    if len(data) != wanted:
        report.fail('ai/fast_lane.ai весит %d байт, схема на %d точек требует %d'
                    % (len(data), count, wanted))
        return
    last = struct.unpack_from('<4f i', data, 16 + (count - 1) * 20)
    if last[4] != count - 1:
        report.fail('номера точек AI-линии не доходят до %d' % (count - 1))
    report.note('AI-линия: %d точек, длина %.0f м' % (count, last[3]))


def check_ui(track, report):
    """The card the game shows before anyone drives anything."""
    folder = track / 'ui'
    path = folder / 'ui_track.json'
    if not path.is_file():
        report.fail('нет ui/ui_track.json — трасса может не появиться в списке')
        return
    try:
        card = json.loads(path.read_text(encoding='utf8'))
    except ValueError as problem:
        report.fail('ui/ui_track.json не разбирается: %s' % problem)
        return
    for field in ('name', 'length'):
        if not card.get(field):
            report.warn('в карточке трассы пусто поле %s' % field)
    try:
        boxes = int(card.get('pitboxes', 0))
    except ValueError:
        boxes = 0
    if boxes < 2:
        report.warn('pitboxes=%s — с таким числом в сессию не пустят никого, '
                    'кроме одной машины' % card.get('pitboxes'))
    for name in ('preview.png', 'outline.png'):
        if not (folder / name).is_file():
            report.warn('нет ui/%s' % name)
    report.note('карточка: %s, %s м, мест %s'
                % (card.get('name'), card.get('length'), card.get('pitboxes')))


def main(track):
    track = Path(track).resolve()
    if not track.is_dir():
        print('нет такой папки: %s' % track)
        return 1
    print('Проверка установленной трассы: %s\n' % track)
    report = Report()
    check_models(track, report)
    check_map(track, report)
    check_surfaces(track, report)
    check_ai(track, report)
    check_ui(track, report)
    return report.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
