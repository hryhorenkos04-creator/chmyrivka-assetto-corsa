"""Draw Assetto Corsa's track map and UI outline from the v15 racing axis.

AC shows no minimap without data/map.ini and map.png, and the track card
stays blank without ui/outline.png. Both come straight from the axis the
scene already carries, so neither needs the game or the SDK.

  python3 ac_export/make_map.py --racing-line <axis.json> --out <track dir>

map.ini ties the picture back to metres. AC places the car marker with

    pixel_x = world_x * SCALE_FACTOR + X_OFFSET
    pixel_y = world_z * SCALE_FACTOR + Z_OFFSET

so SCALE_FACTOR is pixels per metre and both offsets are PIXELS, not
metres -- the reference block centres a 512 px map with X_OFFSET=256,
which is WIDTH/2.  Drawing at exactly one pixel per metre keeps
SCALE_FACTOR at the conventional 1 and makes the offsets readable: they
are just where the world origin sits inside the image.  MAX_SIZE caps the
size the HUD scales the picture down to; it does not crop it, so a lap
wider than 1600 px still draws in full.
"""
from pathlib import Path
import argparse, json
from PIL import Image, ImageDraw

ROAD_WIDTH_M = 11.0        # mid-point of the project's 8-14 m carriageway
MARGIN_PX = 40
MAX_SIZE = 1600


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


def place(points, scale):
    """Lay the axis out at `scale` px/m and report where the origin landed."""
    x0 = min(p[0] for p in points)
    z0 = min(p[1] for p in points)
    width = round((max(p[0] for p in points) - x0) * scale) + 2 * MARGIN_PX
    height = round((max(p[1] for p in points) - z0) * scale) + 2 * MARGIN_PX
    # Offsets are what AC adds to a scaled world coordinate, so they carry
    # the margin as well as the shift off the minimum corner.
    x_offset = MARGIN_PX - x0 * scale
    z_offset = MARGIN_PX - z0 * scale
    placed = [(x * scale + x_offset, z * scale + z_offset) for x, z in points]
    return placed, width, height, x_offset, z_offset


def draw(points, width, height, pen_width):
    image = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    ImageDraw.Draw(image).line(points + [points[0]], fill=(255, 255, 255, 255),
                               width=pen_width, joint='curve')
    return image


def main(a):
    points = axis(a.racing_line)
    out = Path(a.out).resolve()
    (out / 'data').mkdir(parents=True, exist_ok=True)
    (out / 'ui').mkdir(parents=True, exist_ok=True)

    scale = a.scale
    placed, width, height, x_offset, z_offset = place(points, scale)
    # Kunos ships SCALE_FACTOR=1 on every circuit and magione's WIDTH is
    # 342.88 -- a fraction, so these are metres, not pixels. Shrinking the
    # scale to squeeze the lap under the stock MAX_SIZE of 1600 stopped the
    # map drawing at all, so the cap moves instead of the scale.
    cap = max(MAX_SIZE, width, height)
    pen = max(2, round(ROAD_WIDTH_M * scale))
    draw(placed, width, height, pen).save(out / 'map.png')

    (out / 'data' / 'map.ini').write_text('\n'.join([
        '[PARAMETERS]',
        'WIDTH=%d' % width,
        'HEIGHT=%d' % height,
        'MARGIN=%d' % MARGIN_PX,
        'SCALE_FACTOR=%.6f' % scale,
        'MAX_SIZE=%d' % cap,
        'X_OFFSET=%.3f' % x_offset,
        'Z_OFFSET=%.3f' % z_offset,
        'DRAWING_SIZE=%d' % pen,
        '']), encoding='utf8')

    # The card art is a thumbnail, so it is fitted to a square instead.
    span = max(max(p[0] for p in points) - min(p[0] for p in points),
               max(p[1] for p in points) - min(p[1] for p in points))
    thumb = (a.outline - 2 * MARGIN_PX) / span
    small, _, _, _, _ = place(points, thumb)
    draw(small, a.outline, a.outline,
         max(2, round(ROAD_WIDTH_M * thumb * 1.6))).save(out / 'ui' / 'outline.png')

    print('AC_MAP map.png=%dx%d px  scale=%.3f px/m  pen=%d px  '
          'X_OFFSET=%.1f Z_OFFSET=%.1f  outline=%dpx'
          % (width, height, scale, pen, x_offset, z_offset, a.outline))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--racing-line', required=True, dest='racing_line')
    p.add_argument('--out', required=True, help='the track folder')
    p.add_argument('--scale', type=float, default=1.0,
                   help='pixels per metre; Kunos circuits are all drawn at 1')
    p.add_argument('--outline', type=int, default=512)
    main(p.parse_args())
