"""Read-only FBX export of the v17 scene for Assetto Corsa.

Assigns AC physics prefixes, builds AC_ nulls from the v15 racing axis and
writes FBX. The source .blend is opened read-only and never saved.

  blender --background --python ac_export/export_ac_v17.py -- \
      --scene /path/to/chmyrivka_visual_v17.blend --mode sector

Surfaces named 1ROAD_MAIN and 1PIT_SURFACE_V08 already follow the AC
convention in the source scene and are left untouched.
"""
from pathlib import Path
import argparse, hashlib, json, math, re, sys, time
import bpy
import bmesh
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

# Kept for the whole lap so the circuit stays drivable end to end. Only the
# surfaces a car actually runs on qualify: road, pit lane and kerbs come to
# ~120k triangles together. GRASS and DIRT are deliberately NOT here — the
# VERGE_Grass segments are modelled as volumetric tufts (one 75 m segment is
# 315k triangles, 2.16M across the lap) and would dwarf everything else.
# They are scenery, and get thinned with the rest of it.
SPINE = re.compile(r'^1(ROAD|PIT|KERB)_')


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


ALBEDO = re.compile(r'albedo|diff|color|colour|base', re.I)
NOT_ALBEDO = re.compile(r'_nor|normal|rough|spec|metal|bump|height|ao\b|mask', re.I)


def pick_albedo(material):
    """The image a material's colour mostly comes from, or None.

    Every surface material here mixes several maps through MIX_RGB or a
    Color Ramp, and the FBX exporter only follows a texture wired straight
    into Base Color. So nothing reaches ksEditor and every slot lands NULL.
    """
    images = [n for n in material.node_tree.nodes
              if n.type == 'TEX_IMAGE' and n.image and n.image.packed_file]
    if not images:
        return None
    named = [n for n in images
             if ALBEDO.search(n.image.name) and not NOT_ALBEDO.search(n.image.name)]
    return (named or images)[0]


def flat_image(material):
    """An 8x8 swatch of the material's viewport colour.

    Kerbs and the local asphalt are generated entirely from node maths and
    own no image at all. Without one AC shades them a uniform grey and the
    red/white kerb blocks stop reading as kerbs, so give them their colour
    as a tiny texture instead.
    """
    name = 'ACFLAT_%s.png' % safe(material.name)
    img = bpy.data.images.get(name)
    if img is None:
        img = bpy.data.images.new(name, 8, 8, alpha=False)
        r, g, b = material.diffuse_color[:3]
        img.pixels[:] = [r, g, b, 1.0] * 64
        img.file_format = 'PNG'
    return img


def wire_base_colour(scene, out_dir):
    """Put one image straight into Base Color so the FBX carries a texture."""
    done, flats, skipped = [], [], []
    chosen = {}
    seen = set()
    for ob in scene.objects:
        if ob.type != 'MESH' or not ob.select_get():
            continue
        for slot in ob.material_slots:
            m = slot.material
            if m is None or m.name in seen or not m.node_tree:
                continue
            seen.add(m.name)
            bsdf = next((n for n in m.node_tree.nodes
                         if n.type == 'BSDF_PRINCIPLED'), None)
            if bsdf is None:
                skipped.append(m.name)
                continue
            node = pick_albedo(m)
            if node is None:
                img = flat_image(m)
                path = out_dir / img.name
                img.filepath_raw = str(path)
                img.save()
                node = m.node_tree.nodes.new('ShaderNodeTexImage')
                node.image = img
                flats.append({'material': m.name, 'image': img.name,
                              'colour': [round(v, 3) for v in m.diffuse_color[:3]]})
                chosen[m.name] = safe_file(img.name)
            else:
                done.append({'material': m.name, 'image': node.image.name})
                chosen[m.name] = safe_file(node.image.name)
            base = bsdf.inputs['Base Color']
            for link in list(base.links):
                m.node_tree.links.remove(link)
            m.node_tree.links.new(node.outputs['Color'], base)
    return done, flats, skipped, chosen


PERSISTENCE_VARS = [('ksAmbient', 0.05), ('ksDiffuse', 0.8), ('ksSpecular', 0.8),
                    ('ksSpecularEXP', 10), ('ksEmissive', 0), ('ksAlphaRef', 0)]


def merge_materials(scene, out_dir, buckets=12):
    """Collapse every material onto a shared one per texture.

    The sector carries 284 materials and ksEditor fills each txDiffuse by
    hand, so a build is unusable without collapsing them first. Materials
    that resolve to the same image become one; the flat-colour ones are
    quantised into a few shades so a street of houses does not produce a
    material per wall.
    """
    shared, assignments = {}, {}
    step = 1.0 / buckets
    for ob in scene.objects:
        if ob.type != 'MESH' or not ob.select_get():
            continue
        for slot in ob.material_slots:
            m = slot.material
            if m is None or not m.node_tree:
                continue
            node = pick_albedo(m)
            if node is not None:
                key = safe_file(node.image.name)
                image = node.image
            else:
                shade = tuple(min(buckets - 1, int(v / step))
                              for v in m.diffuse_color[:3])
                key = 'ACSHADE_%d_%d_%d.png' % shade
                image = shared.get(key, (None, None))[1]
                if image is None:
                    image = bpy.data.images.new(key, 8, 8, alpha=False)
                    rgb = [(c + 0.5) * step for c in shade]
                    image.pixels[:] = list(rgb) + [1.0] * 1 + \
                        (list(rgb) + [1.0]) * 63
                    image.file_format = 'PNG'
                    image.filepath_raw = str(out_dir / key)
                    image.save()
            if key not in shared:
                merged = bpy.data.materials.new('AC_' + Path(key).stem)
                merged.use_nodes = True
                tree = merged.node_tree
                bsdf = next(n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED')
                tex = tree.nodes.new('ShaderNodeTexImage')
                tex.image = image
                tree.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
                shared[key] = (merged, image)
                assignments[merged.name] = key
            slot.material = shared[key][0]
    return assignments


def node_sections(fbx_name, scene):
    """ksEditor's per-node state, mirroring the tree it builds on import.

    One section for the file root, one per object, and one per mesh child
    named after its object — split into _SUB0.._SUBn when the mesh carries
    several materials, which is how the two-tone kerbs arrive.
    """
    lines = ['[model_FBX: %s]' % fbx_name, 'ACTIVE=1', 'PRIORITY=0', '']
    for ob in scene.objects:
        if not ob.select_get() or ob.type not in ('MESH', 'EMPTY'):
            continue
        node = '%s_%s' % (fbx_name, ob.name)
        lines += ['[model_FBX: %s]' % node, 'ACTIVE=1', 'PRIORITY=0', '']
        if ob.type != 'MESH':
            continue
        # Distinct materials, not slots: several slots often point at one
        # material and ksEditor splits by material, leaving such a mesh whole.
        slots = ob.material_slots
        used = {slots[p.material_index].material for p in ob.data.polygons
                if p.material_index < len(slots)} - {None}
        children = (['%s_%s' % (node, ob.name)] if len(used) < 2 else
                    ['%s_%s_SUB%d' % (node, ob.name, i) for i in range(len(used))])
        for child in children:
            lines += ['[model_FBX: %s]' % child, 'ACTIVE=1', 'PRIORITY=0',
                      'VISIBLE=1', 'TRANSPARENT=0', 'CAST_SHADOWS=1',
                      'LOD_IN=0', 'LOD_OUT=0', 'RENDERABLE=1', '']
    return lines


def write_persistence(fbx, assignments, scene=None):
    """Write ksEditor's <name>.fbx.ini so it loads the textures itself.

    ksEditor does not read Blender's material links: every txDiffuse slot
    comes in NULL and a KN5 save refuses until each is filled by hand. Its
    persistence file is plain ini keyed to the FBX's MD5, and the export
    already knows which image each material resolved to, so writing it here
    saves clicking through hundreds of materials.
    """
    digest = hashlib.md5(fbx.read_bytes()).hexdigest()
    lines = ['[HEADER]', 'VERSION=4', 'DLC_KEY=0', 'USERNAME=',
             'DATE=' + time.strftime('%m/%d/%Y %I:%M:%S %p'),
             'MD5_HASH=' + digest, '',
             '[MATERIAL_LIST]', 'COUNT=%d' % len(assignments), '']
    for index, (material, texture) in enumerate(sorted(assignments.items())):
        lines += ['[MATERIAL_%d]' % index, 'NAME=' + material,
                  'SHADER=ksPerPixel', 'ALPHABLEND=0', 'ALPHATEST=0',
                  'DEPTHMODE=0', 'VARCOUNT=%d' % len(PERSISTENCE_VARS)]
        for slot, (var, value) in enumerate(PERSISTENCE_VARS):
            lines += ['VAR_%d_NAME=%s' % (slot, var),
                      'VAR_%d_FLOAT1=%g' % (slot, value),
                      'VAR_%d_FLOAT2=0,0' % slot,
                      'VAR_%d_FLOAT3=0,0,0' % slot,
                      'VAR_%d_FLOAT4=0,0,0,0' % slot]
        lines += ['RESCOUNT=1', 'RES_0_NAME=txDiffuse', 'RES_0_SLOT=0',
                  'RES_0_TEXTURE=' + texture, '']
    if scene is not None:
        lines += node_sections(fbx.name, scene)
    path = fbx.with_name(fbx.name + '.ini')
    # newline= stops Windows translating again and giving CRCRLF
    path.write_text('\n'.join(lines), encoding='utf8', newline='\r\n')
    return path


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


def corridor_bounds(points, corridor):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs) - corridor, max(xs) + corridor,
            min(ys) - corridor, max(ys) + corridor)


def trim_to_corridor(ob, points, corridor, bounds):
    """Drop the faces that fall outside the corridor.

    Scenery is authored as a few meshes spanning the whole lap — one
    vegetation object carries 269k triangles for every garden on the
    circuit — so selecting whole objects is all-or-nothing and a 100 m
    sector still drags in 1.5M triangles. Keeping faces by their centre
    cuts those down to the part actually on screen.

    Returns True when trimmed, False when untouched, None when nothing
    survives and the caller should drop the object.
    """
    mesh = ob.data = ob.data.copy()   # may be shared; never touch the original
    world = ob.matrix_world
    limit = corridor * corridor
    x0, x1, y0, y1 = bounds
    bm = bmesh.new()
    bm.from_mesh(mesh)
    doomed = []
    for face in bm.faces:
        c = world @ face.calc_center_median()
        if not (x0 <= c.x <= x1 and y0 <= c.y <= y1):
            doomed.append(face)
            continue
        cx, cy = c.x, c.y
        if not any((px - cx) ** 2 + (py - cy) ** 2 <= limit for px, py in points):
            doomed.append(face)
    kept = len(bm.faces) - len(doomed)
    if not doomed:
        bm.free()
        return False
    if not kept:
        bm.free()
        return None
    bmesh.ops.delete(bm, geom=doomed, context='FACES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return True


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
    kept = culled = triangles = trimmed = 0
    bounds = corridor_bounds(points, a.corridor) if points else None
    for ob in scene.objects:
        if ob.type not in ('MESH', 'EMPTY'):
            continue
        if ob.type == 'EMPTY' and not ob.name.startswith('AC_'):
            continue
        take = True
        if ob.type == 'MESH':
            if a.road_only:
                take = bool(SPINE.match(ob.name))
            elif points is not None and not SPINE.match(ob.name):
                take = in_sector(ob, points, a.corridor)
                if take and a.trim:
                    result = trim_to_corridor(ob, points, a.corridor, bounds)
                    if result is None:
                        take = False
                    elif result:
                        trimmed += 1
        ob.select_set(take)
        if take and ob.type == 'MESH':
            triangles += sum(len(p.vertices) - 2 for p in ob.data.polygons)
        kept, culled = kept + bool(take), culled + (not take)

    out = Path(a.output).resolve() if a.output else scene_path.parent / 'ac_export_out/chmyrivka.fbx'
    report_path = Path(a.report).resolve() if a.report else out.parent / 'export_report.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    textures, skipped_textures = unpack_textures(out.parent, a.keep_hdr)
    if a.merge:
        chosen = merge_materials(scene, out.parent)
        wired, flats, no_bsdf = [], [], []
    else:
        wired, flats, no_bsdf, chosen = wire_base_colour(scene, out.parent)
    bpy.ops.export_scene.fbx(
        filepath=str(out), use_selection=True, apply_unit_scale=True,
        global_scale=a.scale, axis_forward=a.axis_forward, axis_up=a.axis_up,
        object_types={'MESH', 'EMPTY'}, use_mesh_modifiers=True,
        mesh_smooth_type='FACE', path_mode='STRIP', embed_textures=False,
        bake_space_transform=a.bake_transform)

    persistence = write_persistence(out, chosen, scene) if chosen else None

    report = {
        'mode': a.mode, 'scene': str(scene_path), 'fbx': str(out),
        'renamed': len(applied), 'visual_only_meshes': visual,
        'exported_objects': kept, 'culled_objects': culled,
        'exported_triangles': triangles, 'road_only': a.road_only,
        'trimmed_meshes': trimmed,
        'persistence': str(persistence) if persistence else None,
        'textures_written': textures, 'textures_skipped': skipped_textures,
        'base_colour_wired': wired, 'base_colour_flat': flats,
        'materials_without_principled': no_bsdf,
        'start_station_map_m': a.start_station,
        'start_location': list(origin), 'flip_spawn': a.flip,
        'sector_map_m': [a.sector_from, a.sector_to] if a.mode == 'sector' else None,
        'assetto_corsa_tested': False, 'renames': applied,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('AC_EXPORT_DONE objects=%d culled=%d triangles=%d renamed=%d visual_only=%d '
          'textures=%d wired=%d flat=%d trimmed=%d persistence=%d'
          % (kept, culled, triangles, len(applied), visual, len(textures),
             len(wired), len(flats), trimmed, len(chosen)))
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
    p.add_argument('--merge', action='store_true',
                   help='collapse materials onto one per texture; needed once the '
                        'sector brings hundreds that ksEditor would want by hand')
    p.add_argument('--no-trim', action='store_false', dest='trim',
                   help='keep whole scenery meshes instead of cutting them to the corridor')
    p.add_argument('--start-station', type=float, default=3800.0, dest='start_station')
    p.add_argument('--gate-width', type=float, default=14.0, dest='gate_width')
    p.add_argument('--spawn-lift', type=float, default=0.15, dest='spawn_lift')
    p.add_argument('--flip', action='store_true', help='car spawns facing backwards')
    # ksEditor reads a plain export 100x too large: bake_space_transform folds
    # Blender's metre-to-centimetre conversion into the vertex data, and
    # global_scale only takes effect alongside it. Together they land the raw
    # coordinates on their true metre values (road Y extent 2131.65 m).
    p.add_argument('--no-bake', action='store_false', dest='bake_transform',
                   help='leave the axis conversion on the nodes; ksEditor then reads it 100x large')
    p.add_argument('--axis-up', dest='axis_up', default='Y', choices=['X', 'Y', 'Z', '-X', '-Y', '-Z'],
                   help="Z leaves Blender's axes alone, for when the importer converts them itself")
    p.add_argument('--axis-forward', dest='axis_forward', default='-Z',
                   choices=['X', 'Y', 'Z', '-X', '-Y', '-Z'])
    p.add_argument('--scale', type=float, default=0.01,
                   help='FBX global scale; only applies with the baked transform')
    p.add_argument('--road-only', action='store_true', dest='road_only',
                   help='drivable surfaces alone, no scenery: smallest file that still drives')
    p.add_argument('--keep-hdr', action='store_true', dest='keep_hdr',
                   help='also write the 4K sky HDR, unused by AC')
    p.add_argument('--racing-line', dest='racing_line', default=None,
                   help='racing_line_native_v15.json; defaults to %s beside the scene' % LINE_REL)
    main(p.parse_args(argv))
