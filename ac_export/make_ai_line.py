"""Write Assetto Corsa's ai/fast_lane.ai from the v15 racing axis.

Without it the track has no AI: opponents cannot be added, and a session
is whatever one car does alone. The usual way to get one is to drive a
lap with the game recording, but the file is only data, and the scene
already carries a racing line solved against the asphalt -- closed, C2
continuous, validated as covered by the road with 1.65 m of clearance at
its tightest. That is a better line than most people drive.

  python3 ac_export/make_ai_line.py --racing-line <axis.json> --out <track dir>

Format, read off the import/export pair in leBluem/io_import_accsv, which
round-trips files the game and ksEditor accept:

  header   4 x int32          version 7, point count, 0, 0
  ideal    per point, 4f + i  x, y, z, distance from start, index
  extra    per point, 18f     mostly zero; 6 and 7 are the room to the
                              left and right edges, and the first float
                              of the first point carries the point count
                              again, as an int reinterpreted as a float

Everything the game does not need for a usable line is left at zero,
which is what that exporter writes for a line built from scratch. Filling
fields whose meaning is unconfirmed would be worse than leaving them
empty: the file parses either way, so a wrong guess shows up as strange
AI behaviour rather than as an error.
"""
from pathlib import Path
import argparse, json, math, struct, sys

VERSION = 7
EXTRA_FIELDS = 18
SIDE_LEFT, SIDE_RIGHT = 6, 7

# The smallest positive denormal: its bit pattern is the integer 1, so
# multiplying by a count reinterprets that count as a float. The reference
# exporter uses this to put the point count in the first extra field.
DENORMAL = 1.40129846432481e-45


def samples(path):
    path = Path(path)
    if not path.is_file():
        sys.exit('racing axis not found: %s' % path)
    rows = json.loads(path.read_text(encoding='utf8'))['samples']
    assert len(rows) > 2, 'racing axis is too short'
    return rows


def resample(rows, spacing):
    """Thin the axis to roughly one point every `spacing` metres.

    The axis is sampled every half metre, which is more than an AI line
    needs and makes the file four times larger than any shipped one.
    """
    kept, last = [], None
    for row in rows:
        if last is None or row['line_s_m'] - last >= spacing:
            kept.append(row)
            last = row['line_s_m']
    # The axis closes on itself; the repeated last sample would put two
    # points in the same place and give the AI a zero-length segment.
    if kept and math.dist((kept[0]['x'], kept[0]['y']),
                          (kept[-1]['x'], kept[-1]['y'])) < spacing * 0.5:
        kept.pop()
    return kept


def rotate_to_start(rows, station):
    """Put the first point on the start line rather than at chainage zero.

    Nothing confirms that the game requires it -- lap timing comes from
    the AC_TIME gates, and the line is a closed loop either way -- but
    every shipped file begins there, and starting elsewhere costs nothing
    to avoid.
    """
    if station is None:
        return rows
    nearest = min(range(len(rows)),
                  key=lambda i: abs(rows[i]['map_s_m'] - station))
    return rows[nearest:] + rows[:nearest]


def to_game(row):
    """Blender's Z-up into the game's Y-up, the same turn the export makes.

    Verified against the exported FBX rather than assumed: its Z spans
    -2496..348 where the scene's Y spans 32..2122, so the game's Z is the
    scene's Y negated.
    """
    return (row['x'], row['z'], -row['y'])


def write(path, points, left, right):
    with open(path, 'wb') as out:
        out.write(struct.pack('<4i', VERSION, len(points), 0, 0))
        travelled = 0.0
        previous = None
        for index, point in enumerate(points):
            if previous is not None:
                travelled += math.dist(point, previous)
            previous = point
            out.write(struct.pack('<4f i', point[0], point[1], point[2],
                                  travelled, index))
        for index in range(len(points)):
            extra = [0.0] * EXTRA_FIELDS
            extra[SIDE_LEFT] = left
            extra[SIDE_RIGHT] = right
            if index == 0:
                extra[0] = DENORMAL * len(points)
            out.write(struct.pack('<%df' % EXTRA_FIELDS, *extra))
        return travelled


def verify(path, expected):
    """Read the file back the way the game's parsers do."""
    data = Path(path).read_bytes()
    version, count, _, _ = struct.unpack_from('<4i', data, 0)
    if version != VERSION:
        return 'version is %d, not %d' % (version, VERSION)
    if count != expected:
        return 'header says %d points, %d were written' % (count, expected)
    wanted = 16 + count * 20 + count * EXTRA_FIELDS * 4
    if len(data) != wanted:
        return 'file is %d bytes, the layout needs %d' % (len(data), wanted)
    first = struct.unpack_from('<4f i', data, 16)
    last = struct.unpack_from('<4f i', data, 16 + (count - 1) * 20)
    if first[4] != 0 or last[4] != count - 1:
        return 'point indices do not run 0..%d' % (count - 1)
    if not last[3] > first[3]:
        return 'distance does not increase along the line'
    packed = struct.unpack_from('<f', data, 16 + count * 20)[0]
    echoed = struct.unpack('<i', struct.pack('<f', packed))[0]
    if echoed != count:
        return 'the count echoed in the extra block reads %d' % echoed
    return None


def main(a):
    rows = resample(samples(a.racing_line), a.spacing)
    # Index order runs the lap the opposite way round from the way it is
    # driven -- the same reason the export needs --flip for the spawn,
    # which was confirmed in game. An AI line written the other way sends
    # every opponent round backwards.
    if not a.keep_order:
        rows = rows[::-1]
    rows = rotate_to_start(rows, a.start_station)
    points = [to_game(r) for r in rows]

    out = Path(a.out).resolve() / 'ai'
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'fast_lane.ai'
    length = write(path, points, a.left, a.right)

    problem = verify(path, len(points))
    if problem:
        sys.exit('fast_lane.ai did not read back: %s' % problem)
    print('AC_AI points=%d length=%.1fm spacing=%.2fm left=%.1f right=%.1f '
          'reversed=%s bytes=%d'
          % (len(points), length, length / max(len(points) - 1, 1),
             a.left, a.right, not a.keep_order, path.stat().st_size))
    print('AC_AI_OUT %s' % path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--racing-line', required=True, dest='racing_line')
    p.add_argument('--out', required=True, help='the track folder')
    p.add_argument('--spacing', type=float, default=1.5,
                   help='metres between points')
    # The line solver reports 1.649 m of clearance at its tightest point.
    # Claiming more room than that would invite the AI onto the verge
    # exactly where the road is narrowest.
    p.add_argument('--left', type=float, default=1.6,
                   help='metres of room to the left edge')
    p.add_argument('--right', type=float, default=1.6,
                   help='metres of room to the right edge')
    p.add_argument('--start-station', type=float, default=3950.0,
                   dest='start_station',
                   help='chainage the line should begin at, matching the export')
    p.add_argument('--keep-order', action='store_true', dest='keep_order',
                   help='write the axis in stored order; the default reverses '
                        'it to match the direction the lap is driven')
    main(p.parse_args())
