"""Read-only FBX export of the v17 scene for Assetto Corsa.

Assigns AC physics prefixes, builds AC_ nulls from the v15 racing axis and
writes FBX. The source .blend is opened read-only and never saved.

  blender --background --python ac_export/export_ac_v17.py -- \
      --scene /path/to/chmyrivka_visual_v17.blend --mode sector

Surfaces named 1ROAD_MAIN and 1PIT_SURFACE_V08 already follow the AC
convention in the source scene and are left untouched.
"""
from pathlib import Path
import argparse, json, math, re, sys
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parent.parent
# Ships beside the scene in the v17 release archive, so the default is
# resolved against the .blend rather than against this script.
LINE_REL = 'data/inherited_racing_v15/racing_line_native_v15.json'

# Physics surface mapping, first match wins. KEY must exist in surfaces.ini.
# A leading '1' marks the mesh as collidable; everything else is visual only.
SURFACES = [
    (r'^1(ROAD|PIT|GRASS|KERB|SAND|DIRT)', None),   # already prefixed, keep
    (r'^ROAD_SHOULDERS', 'ROAD'),
    (r'^SIDE_ROADS', 'ROAD'),
    (r'KERB|V08_Racing_kerbs', 'KERB'),
    (r'^VERGE_Gravel', 'DIRT'),
    (r'^VERGE_Grass', 'GRASS'),
    (r'^GROUND_land_industrial', 'DIRT'),
    (r'^GROUND_', 'GRASS'),
]

# Anything that carries physics after renaming spans the lap and is never
# sector-culled: the whole circuit stays drivable, only scenery is thinned.
SPINE = re.compile(r'^1(ROAD|PIT|GRASS|KERB|SAND|DIRT)_')


def line_samples(path):
    path = Path(path)
    if not path.is_file():
        sys.exit('racing axis not found: %s\n'
                 'Pass --racing-line with the path to '
                 'data/inherited_racing_v15/racing_line_native_v15.json '
                 '(it ships inside the v17 release archive).' % path)
    rows = json.loads(path.read_text(encoding='utf8'))['samples']
    assert rows, 'racing axis is empty'
    return rows


def at_station(rows, station):
    """Sample nearest the given original-map chainage, with travel direction."""
    i = min(range(len(rows)), key=lambda k: abs(rows[k]['map_s_m'] - station))
    a, b = rows[i], rows[(i + 8) % len(rows)]
    d = Vector((b['x'] - a['x'], b['y'] - a['y'], 0.0))
    if d.length < 1e-6:
        d = Vector((1.0, 0.0, 0.0))
    return Vector((a['x'], a['y'], a['z'])), d.normalized()


def surface_for(name):
    for pattern, key in SURFACES:
        if re.search(pattern, name, re.I):
            return key
    return False  # visual only


def safe(name):
    """ksEditor dislikes non-ASCII and overlong identifiers."""
    out = re.sub(r'[^A-Za-z0-9_]', '_', name)
    return out[:58] if len(out) > 58 else out


def safe_file(name):
    return re.sub(r'[^A-Za-z0-9_.-]', '_', name)


def rename(scene):
    taken, applied, visual = set(), [], 0
    for ob in scene.objects:
        if ob.type != 'MESH':
            continue
        key = surface_for(ob.name)
        base = safe(ob.name) if key is None or key is False else '1%s_%s' % (key, safe(ob.name))
        if key is False:
            visual += 1
        name, n = base, 1
        while name in taken:
            n += 1
            name = '%s_%d' % (base[:54], n)
        taken.add(name)
        if name != ob.name:
            applied.append({'from': ob.name, 'to': name, 'surface': key or 'inherited'})
        ob.name = name
    return applied, visual


def unpack_textures(out_dir, keep_hdr=False):
    """Write packed image data out and repoint the datablocks at it.

    All 25 images are packed, but each kept the path it was imported from,
    under a chmyrivka_detail_v13/textures tree that ships with nothing. The
    FBX exporter copies from that path, finds nothing and silently drops
    every texture. packed_file.data holds the original bytes, so writing
    those out costs no re-encode.
    """
    # Beside the FBX, not in a subfolder: the export strips paths to bare
    # filenames so ksEditor resolves them from the FBX's own directory.
    tex = out_dir
    tex.mkdir(parents=True, exist_ok=True)
    written, skipped = [], []
    for img in bpy.data.images:
        if img.packed_file is None:
            continue
        if not keep_hdr and img.name.lower().endswith('.hdr'):
            skipped.append(img.name)  # world lighting only; AC brings its own sky
            continue
        path = tex / safe_file(img.name)
        path.write_bytes(img.packed_file.data)
        img.filepath_raw = str(path)
        img.source = 'FILE'
        written.append(path.name)
    return written, skipped


def null(scene, name, location, direction=None, flip=False):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_type = 'ARROWS'
    ob.empty_display_size = 3.0
    ob.location = location
    if direction is not None:
        d = -direction if flip else direction
        ob.rotation_euler = d.to_track_quat('Y', 'Z').to_euler()
    scene.collection.objects.link(ob)
    return ob


def sector_points(rows, a, b, step=5.0):
    """Axis points inside the sector, thinned to `step` metres.

    The raw axis is sampled every ~0.5 m; against a 100 m+ corridor that
    detail is wasted and turns the selection into millions of comparisons.
    """
    out, last = [], None
    for r in rows:
        if not a <= r['map_s_m'] <= b:
            continue
        if last is None or r['map_s_m'] - last >= step:
            out.append((r['x'], r['y']))
            last = r['map_s_m']
    assert out, 'sector %g-%g m is outside the axis' % (a, b)
    return out


def in_sector(ob, points, corridor):
    centre = ob.matrix_world @ (sum((Vector(c) for c in ob.bound_box), Vector()) / 8.0)
    limit = corridor * corridor
    cx, cy = centre.x, centre.y
    for px, py in points:
        if (px - cx) ** 2 + (py - cy) ** 2 <= limit:
            return True
    return False


def main(a):
    scene_path = Path(a.scene).resolve()
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    scene = bpy.context.scene
    rows = line_samples(a.racing_line or scene_path.parent / LINE_REL)

    applied, visual = rename(scene)

    # Grid slot, pit box and the start/finish timing gate.
    origin, heading = at_station(rows, a.start_station)
    side = Vector((-heading.y, heading.x, 0.0))
    null(scene, 'AC_START_0', origin + Vector((0, 0, a.spawn_lift)), heading, a.flip)
    null(scene, 'AC_PIT_0', origin - heading * 12.0 + Vector((0, 0, a.spawn_lift)), heading, a.flip)
    null(scene, 'AC_TIME_0_L', origin + side * (a.gate_width / 2.0))
    null(scene, 'AC_TIME_0_R', origin - side * (a.gate_width / 2.0))

    # Selection: every physical surface always, scenery only near the sector.
    for ob in scene.objects:
        ob.select_set(False)
    points = sector_points(rows, a.sector_from, a.sector_to) if a.mode == 'sector' else None
    kept = culled = 0
    for ob in scene.objects:
        if ob.type not in ('MESH', 'EMPTY'):
            continue
        if ob.type == 'EMPTY' and not ob.name.startswith('AC_'):
            continue
        take = True
        if points is not None and ob.type == 'MESH' and not SPINE.match(ob.name):
            take = in_sector(ob, points, a.corridor)
        ob.select_set(take)
        kept, culled = kept + bool(take), culled + (not take)

    out = Path(a.output).resolve() if a.output else scene_path.parent / 'ac_export_out/chmyrivka.fbx'
    report_path = Path(a.report).resolve() if a.report else out.parent / 'export_report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    textures, skipped_textures = unpack_textures(out.parent, a.keep_hdr)
    bpy.ops.export_scene.fbx(
        filepath=str(out), use_selection=True, apply_unit_scale=True,
        global_scale=1.0, axis_forward='-Z', axis_up='Y',
        object_types={'MESH', 'EMPTY'}, use_mesh_modifiers=True,
        mesh_smooth_type='FACE', path_mode='STRIP', embed_textures=False,
        bake_space_transform=False)

    report = {
        'mode': a.mode, 'scene': str(scene_path), 'fbx': str(out),
        'renamed': len(applied), 'visual_only_meshes': visual,
        'exported_objects': kept, 'culled_objects': culled,
        'textures_written': textures, 'textures_skipped': skipped_textures,
        'start_station_map_m': a.start_station,
        'start_location': list(origin), 'flip_spawn': a.flip,
        'sector_map_m': [a.sector_from, a.sector_to] if a.mode == 'sector' else None,
        'assetto_corsa_tested': False, 'renames': applied,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('AC_EXPORT_DONE objects=%d culled=%d renamed=%d visual_only=%d textures=%d'
          % (kept, culled, len(applied), visual, len(textures)))
    print('AC_EXPORT_OUT %s' % out)


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--mode', choices=['sector', 'full'], default='sector')
    p.add_argument('--output', default=None, help='default: ac_export_out/chmyrivka.fbx beside the scene')
    p.add_argument('--report', default=None, help='default: export_report.json beside the FBX')
    p.add_argument('--sector-from', type=float, default=3800.0, dest='sector_from')
    p.add_argument('--sector-to', type=float, default=4250.0, dest='sector_to')
    p.add_argument('--corridor', type=float, default=120.0)
    p.add_argument('--start-station', type=float, default=3800.0, dest='start_station')
    p.add_argument('--gate-width', type=float, default=14.0, dest='gate_width')
    p.add_argument('--spawn-lift', type=float, default=0.15, dest='spawn_lift')
    p.add_argument('--flip', action='store_true', help='car spawns facing backwards')
    p.add_argument('--keep-hdr', action='store_true', dest='keep_hdr',
                   help='also write the 4K sky HDR, unused by AC')
    p.add_argument('--racing-line', dest='racing_line', default=None,
                   help='racing_line_native_v15.json; defaults to %s beside the scene' % LINE_REL)
    main(p.parse_args(argv))
