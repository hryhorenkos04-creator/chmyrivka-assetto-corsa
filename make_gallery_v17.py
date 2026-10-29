"""Build an offline gallery of paired, unretouched Blender renders.

Requires the complete six main + five extra before/after manifests. This
script verifies camera/settings identity; it does not establish art quality.
"""
from pathlib import Path
import argparse
import html
import json
import math
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
SOURCE_SHA = '8e4f2c9c3e1b37de42d9d99edf115a898fa72cb76936dce19020bb02f23b0e00'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
PAIRED_KEYS = ('camera', 'location', 'rotation', 'lens_mm', 'resolution',
               'samples', 'seed', 'engine', 'view_transform', 'exposure')


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def label(draw, xy, value, size=24, bold=False, fill='#e8edf1'):
    draw.text(xy, value, font=font(size, bold), fill=fill)


def save_png(im, path):
    temporary = path.with_name(path.name + '.tmp')
    im.save(temporary, format='PNG')
    with Image.open(temporary) as check:
        check.verify()
    temporary.replace(path)


def rows(folder, count, expected_sha):
    items = sorted(json.loads((ROOT / folder / 'render_manifest.json').read_text()),
                   key=lambda row: row['file'])
    assert len(items) == count, (folder, len(items), count)
    assert len({row['camera'] for row in items}) == count, folder
    assert len({row['file'] for row in items}) == count, folder
    for row in items:
        assert Path(row['file']).name == row['file'], row['file']
        assert row['native_sha256'] == expected_sha, (folder, row['camera'])
        assert row['resolution'] == [1600, 900] and row['samples'] == 48
        with Image.open(ROOT / folder / row['file']) as im:
            assert im.size == tuple(row['resolution'])
            im.verify()
    return items


def paired_rows(before_folder, after_folder, count, native_sha):
    before = rows(before_folder, count, SOURCE_SHA)
    after = rows(after_folder, count, native_sha)
    lookup = {row['camera']: row for row in before}
    assert set(lookup) == {row['camera'] for row in after}
    pairs = []
    for current in after:
        previous = lookup[current['camera']]
        for key in PAIRED_KEYS:
            assert previous[key] == current[key], (key, current['camera'])
        pairs.append({'before_folder': before_folder, 'before': previous,
                      'after_folder': after_folder, 'after': current})
    return pairs


def paste_pic(board, path, xy, size):
    with Image.open(path) as original:
        im = original.convert('RGB')
    im.thumbnail(size, Image.Resampling.LANCZOS)
    board.paste(im, xy)


def comparison_board(path, pairs, subtitle):
    board = Image.new('RGB', (1660, 118 + 520 * len(pairs)), '#141b23')
    draw = ImageDraw.Draw(board)
    label(draw, (24, 18), 'ЧМИРІВКА / СТАРОБІЛЬСЬК · v16 → v17', 30, True)
    label(draw, (24, 62), subtitle, 20, fill='#9eabb8')
    for j, pair in enumerate(pairs):
        y = 108 + j * 520
        label(draw, (24, y), pair['after']['title'], 23, True)
        label(draw, (24, y + 31), 'ДО · v16', 18)
        label(draw, (850, y + 31), 'ПОСЛЕ · v17', 18)
        paste_pic(board, ROOT / pair['before_folder'] / pair['before']['file'],
                  (24, y + 58), (786, 442))
        paste_pic(board, ROOT / pair['after_folder'] / pair['after']['file'],
                  (850, y + 58), (786, 442))
    save_png(board, path)


def image_card(pair, index, baseline=False):
    side = 'before' if baseline else 'after'
    row, folder = pair[side], pair[side + '_folder']
    href = '../' + folder + '/' + row['file']
    title = f'{index:02d} · {row["title"]}'
    escaped = html.escape(href, quote=True)
    return (f'<figure><a href="{escaped}"><img src="{escaped}" '
            f'alt="{html.escape(title, quote=True)}" loading="lazy"></a>'
            f'<figcaption>{html.escape(title)}</figcaption></figure>')


def main(args):
    info = json.loads((ROOT / 'build_info_v17.json').read_text())
    native_sha = info['native_sha256']
    assert info['source_sha256'] == SOURCE_SHA and info['source_unchanged']
    main_pairs = paired_rows('screenshots_before_v16', 'screenshots_v17', 6, native_sha)
    extra_pairs = paired_rows('screenshots_extra_before_v16', 'screenshots_extra_v17', 5, native_sha)
    # This retained ground camera is now partly occluded by nearby foliage.
    # Present it as evidence of a limitation, not as an improved ground sample.
    for pair in extra_pairs:
        if pair['after']['file']=='02_verge.png':
            pair['after']=dict(pair['after'],title='Растительность вблизи · упрощённая листва')
    pairs = main_pairs + extra_pairs
    out = ROOT / 'comparisons'
    out.mkdir(exist_ok=True)
    by_file = lambda group, filename: next(p for p in group if p['after']['file'] == filename)
    driver = [by_file(main_pairs, '01_driver.png'), by_file(extra_pairs, '05_driver.png')]
    architecture = [by_file(main_pairs, '02_detail.png'), by_file(main_pairs, '03_detail.png')]
    layout = [by_file(main_pairs, '04_overview.png')]
    for name, group in [('driver', driver), ('architecture', architecture), ('layout', layout)]:
        comparison_board(out / f'comparison_{name}_v17.png', group,
                         'Одинаковые камеры и настройки · фактические рендеры Blender Cycles')
    if args.comparisons_only:
        print('Three paired v16/v17 comparison boards ready')
        return
    board = Image.new('RGB', (1348, 105 + 420 * math.ceil(len(pairs) / 2) + 30), '#141b23')
    draw = ImageDraw.Draw(board)
    label(draw, (24, 18), 'ЧМИРІВКА / СТАРОБІЛЬСЬК · галерея v17', 30, True)
    label(draw, (24, 60), '11 ракурсов полной сцены · Blender Cycles, не игровой захват',
          19, fill='#9eabb8')
    for i, pair in enumerate(pairs):
        x, y = 24 + (i % 2) * 662, 105 + (i // 2) * 420
        paste_pic(board, ROOT / pair['after_folder'] / pair['after']['file'],
                  (x, y), (638, 359))
        title = f'{i + 1:02d} · {pair["after"]["title"]}'
        if len(title) > 47:
            title = title[:46] + '…'
        label(draw, (x, y + 369), title, 18)
    save_png(board, out / 'contact_sheet_v17.png')
    cards = '\n'.join(image_card(pair, i + 1) for i, pair in enumerate(pairs))
    old_cards = '\n'.join(image_card(pair, i + 1, baseline=True) for i, pair in enumerate(pairs))
    page = '''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Чмирівка · v17</title>
<style>body{background:#141b23;color:#e8edf1;font:17px system-ui;margin:30px auto;max-width:1500px;padding:0 22px}h1{font-size:30px}h2{margin-top:42px}p{color:#aebccb;line-height:1.6}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,360px),1fr));gap:26px}figure{margin:0}img{width:100%;display:block}figcaption{padding:12px 0 18px}a{color:#82dfd1}summary{cursor:pointer;margin:30px 0;font-size:23px}nav{line-height:1.8}</style>
<h1>Старобільськ — Чмирівка · v17</h1>
<p>Фактические рендеры Blender Cycles. Нажмите на кадр для оригинала 1600×900. Основная область доработки — близкие фасады и передние части дворов Сінної. Северный поворот, поребрики и пит-зона показаны для контроля окружения. Это локальный проход, не одинаковая детализация всего круга. Игровой запуск Assetto Corsa не выполнялся.</p>
<nav>Сравнения: <a href="comparison_driver_v17.png">два водительских вида</a> · <a href="comparison_architecture_v17.png">близкие фасады</a> · <a href="comparison_layout_v17.png">расположение дворов</a> · <a href="contact_sheet_v17.png">все 11 кадров</a></nav>
<h2>Текущая сцена v17</h2><div class="grid">''' + cards + '''</div>
<details><summary>Исходная v16 · те же 11 камер</summary><p>Шесть основных и пять дополнительных исходных видов. Дополнительный водительский ракурс 3985 м создан для этого сравнения и снят также в неизменной v16. Камеры, разрешение, число сэмплов, seed, цветовое преобразование и экспозиция проверены по manifest.</p><div class="grid">''' + old_cards + '''</div></details>
<p>Адресная точность новой отделки фасадов не подтверждена. Подробные ограничения и результаты просмотра — в документации выпуска.</p></html>'''
    temporary = out / 'gallery_v17.html.tmp'
    temporary.write_text(page, encoding='utf8')
    temporary.replace(out / 'gallery_v17.html')
    report = {'version': 'v17', 'native_sha256': native_sha, 'source_sha256': SOURCE_SHA,
              'current_views': len(pairs), 'baseline_views': len(pairs),
              'matching_fields': list(PAIRED_KEYS), 'image_manipulation': 'layout and thumbnail resizing only',
              'assetto_corsa_tested': False,
              'pairs': [{'camera': p['after']['camera'],
                         'before': p['before_folder'] + '/' + p['before']['file'],
                         'after': p['after_folder'] + '/' + p['after']['file']} for p in pairs]}
    (out / 'gallery_manifest_v17.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('Gallery ready: 11 current + 11 baseline views, 3 comparison boards, offline HTML')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparisons-only', action='store_true')
    main(parser.parse_args())
