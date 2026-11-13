"""Draw Assetto Corsa's track map and UI outline from the v15 racing axis.

AC shows no minimap without data/map.ini and map.png, and the track card
stays blank without ui/outline.png. Both come straight from the axis the
scene already carries, so neither needs the game or the SDK.

  python3 ac_export/make_map.py --racing-line <axis.json> --out <track dir>
"""
from pathlib import Path
import argparse, json, sys
from PIL import Image, ImageDraw

ROAD_WIDTH_M = 11.0        # mid-point of the project's 8-14 m carriageway
MARGIN_PX = 40


def axis(path):
    rows = json.loads(Path(path).read_text(encoding='utf8'))['samples']
    # The export turns Blender's Z-up into the Y-up the game wants, so the
    # scene's Y becomes AC's -Z. Converting here keeps map.ini in the same
    # coordinates AC reports the car in; reading the raw Y instead puts the
    # marker on the wrong side of the circuit.
    # The axis closes on itself; the repeated last sample would draw a spike.
    points = [(r['x'], -r['y']) for r in rows[:-1]]
    assert len(points) > 2, 'racing axis is too short to draw'
    return points


def fit(points, size):
    xs = [p[0] for p in points]
    zs = [p[1] for p in points]
    span_x, span_z = max(xs) - min(xs), max(zs) - min(zs)
    scale = (size - 2 * MARGIN_PX) / max(span_x, span_z)
    placed = [((x - min(xs)) * scale + MARGIN_PX,
               (z - min(zs)) * scale + MARGIN_PX) for x, z in points]
    return placed, scale, min(xs), min(zs)


def draw(points, size, width, colour, background):
    image = Image.new('RGBA', (size, size), background)
    pen = ImageDraw.Draw(image)
    pen.line(points + [points[0]], fill=colour, width=width, joint='curve')
    return image


def main(a):
    points = axis(a.racing_line)
    out = Path(a.out).resolve()
    (out / 'data').mkdir(parents=True, exist_ok=True)
    (out / 'ui').mkdir(parents=True, exist_ok=True)

    placed, scale, x0, z0 = fit(points, a.size)
    width = max(2, round(ROAD_WIDTH_M * scale))
    draw(placed, a.size, width, (255, 255, 255, 255), (0, 0, 0, 0)).save(out / 'map.png')

    # map.ini ties pixels back to metres so AC can place the car on the map.
    (out / 'data' / 'map.ini').write_text('\n'.join([
        '[PARAMETERS]',
        'WIDTH=%d' % a.size,
        'HEIGHT=%d' % a.size,
        'MARGIN=%d' % MARGIN_PX,
        'SCALE_FACTOR=%.6f' % (1.0 / scale),
        'X_OFFSET=%.3f' % -x0,
        'Z_OFFSET=%.3f' % -z0,
        'DRAWING_SIZE=%.3f' % width,
        '']), encoding='utf8')

    small, small_scale, _, _ = fit(points, a.outline)
    draw(small, a.outline, max(2, round(ROAD_WIDTH_M * small_scale * 1.6)),
         (255, 255, 255, 255), (0, 0, 0, 0)).save(out / 'ui' / 'outline.png')

    print('AC_MAP map.png=%dpx outline.png=%dpx scale=%.4f px/m width=%dpx'
          % (a.size, a.outline, scale, width))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--racing-line', required=True, dest='racing_line')
    p.add_argument('--out', required=True, help='the track folder')
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--outline', type=int, default=512)
    main(p.parse_args())
