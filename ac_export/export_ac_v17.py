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
import numpy as np
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
# GROUND_ joins them: the terrain is one mesh for the whole lap and only
# 48k triangles, but culling it leaves the houses standing over a hole.
SPINE = re.compile(r'^1(ROAD|PIT|KERB)_|^1[A-Z]+_GROUND_')


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
    # ksEditor resolves a stripped texture name against a texture/ folder
    # beside the FBX, not the FBX's own directory: writing them anywhere
    # else leaves every txDiffuse NULL and blocks the KN5 save.
    tex = out_dir / 'texture'
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


def has_alpha(image):
    """True when the image carries a real alpha channel.

    Only the three leaf maps do (depth 32); bark, asphalt, brick and grass
    are opaque, and turning alpha test on for those punches holes in them.
    """
    return image.channels == 4 and image.depth in (32, 64)


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
    # Short name on purpose: the window materials produced 57-character
    # filenames and ksEditor left their txDiffuse NULL, which blocks the
    # KN5 save. A digest keeps them unique without the length.
    name = 'ACF_%s.png' % hashlib.md5(material.name.encode('utf8')).hexdigest()[:10]
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
    chosen, alpha = {}, set()
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
                # Window glass and its shadow are built without a Principled
                # node, so nothing can be wired into them — but ksEditor still
                # refuses to save a KN5 while their txDiffuse reads NULL. The
                # swatch goes into the persistence file even though the FBX
                # material cannot carry it.
                img = flat_image(m)
                path = out_dir / 'texture' / img.name
                path.parent.mkdir(parents=True, exist_ok=True)
                img.filepath_raw = str(path)
                img.save()
                skipped.append(m.name)
                chosen[m.name] = safe_file(img.name)
                continue
            node = pick_albedo(m)
            if node is None:
                img = flat_image(m)
                path = out_dir / 'texture' / img.name
                path.parent.mkdir(parents=True, exist_ok=True)
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
                if has_alpha(node.image):
                    alpha.add(safe_file(node.image.name))
            base = bsdf.inputs['Base Color']
            for link in list(base.links):
                m.node_tree.links.remove(link)
            m.node_tree.links.new(node.outputs['Color'], base)
    return done, flats, skipped, chosen, alpha


# ksEditor's own defaults (0.05 ambient, 0.8 specular) leave everything
# turned away from the sun pure black and make matte walls glossy. Kunos
# track materials sit nearer these values.
PERSISTENCE_VARS = [('ksAmbient', 0.4), ('ksDiffuse', 0.6), ('ksSpecular', 0.1),
                    ('ksSpecularEXP', 30), ('ksEmissive', 0), ('ksAlphaRef', 0)]


# --- grain on the untextured materials ---------------------------------
#
# 400 of the scene's 496 materials own no image: 318 are built from Blender
# noise nodes and 82 are a plain colour. Neither survives FBX, so each
# arrived in AC as one flat shade and a street of houses read as painted
# cardboard. Their viewport colours are deliberate, though -- only three of
# the 400 are left at the default grey -- so the colour is worth keeping and
# only the detail is missing. Each gets one of the tileable photographs the
# scene already carries, tinted to its own colour.

# First match wins, so built things are tested before the ground they
# stand on: V16_Yard_Fences_Gates is a fence, not a yard, and matching
# 'yard' first would surface a steel gate in soil.
TILEABLE = [
    (re.compile(r'brick', re.I), 'red_brick_diff_2k.jpg'),
    (re.compile(r'grass|verge|lawn|hedge|turf', re.I), 'grass001_diff_2k.jpg'),
    (re.compile(r'wall|render|plaster|facade|frame|house|roof|concrete|kerb|'
                r'plaza|slab|paving|stone|cap|crown|seam|rib|sill|chimney|'
                r'porch|shed|fence|gate|door|canopy|curb|step|plinth',
                re.I), 'rough_concrete_diff_2k.jpg'),
    (re.compile(r'soil|dirt|earth|gravel|pebble|ground|land|yard|compact',
                re.I), 'dirt_diff_2k.jpg'),
]

# Grain belongs on masonry and earth, not on what is meant to read as
# polished: concrete noise on a window pane or a galvanised post looks
# worse than the flat colour it replaces.
SMOOTH = re.compile(r'glass|glazing|metal|rubber|wire|chrome|steel|galvan|'
                    r'armco|lens|led|plaque|gasket|paint|mirror|alu', re.I)


def grain_for(name):
    if SMOOTH.search(name):
        return None
    for pattern, image in TILEABLE:
        if pattern.search(name):
            return image
    return None


# How much of the photograph's contrast to keep at most. The raw spread
# runs past 3x on a lit brick face, and a pale render would clip flat
# white across every highlight, losing both the colour and the grain.
GRAIN_STRENGTH = 0.55


def grain_multiplier(image_name, size=256, cache={}):
    """A tileable photo reduced to a brightness multiplier around 1.0.

    Dividing by the mean keeps the photograph's grain while discarding its
    own colour, so tinting afterwards lands on exactly the shade the
    material asked for instead of a blend of the two. The spread is then
    pulled back toward 1 so a light material keeps its colour instead of
    clipping to white.
    """
    if image_name in cache:
        return cache[image_name]
    img = bpy.data.images.get(image_name)
    width, height = img.size if img else (0, 0)
    if not width or not height:
        # has_data stays False until something touches the pixels, so the
        # size is what says whether there is an image to read at all.
        cache[image_name] = None
        return None
    buf = np.empty(width * height * 4, dtype=np.float32)
    try:
        img.pixels.foreach_get(buf)
    except (RuntimeError, AttributeError):
        cache[image_name] = None
        return None
    rgb = buf.reshape(height, width, 4)[:, :, :3]
    rows = np.linspace(0, height - 1, size).astype(int)
    cols = np.linspace(0, width - 1, size).astype(int)
    small = rgb[np.ix_(rows, cols)]
    # Luminance, not the three channels: normalising each channel on its
    # own still leaves the photograph's own hue in the result, and tinting
    # red brick to a cream render came out pink. One grey channel carries
    # the grain and nothing else, so the tint lands exactly on the colour
    # the material asked for.
    luma = small @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    mean = float(luma.mean())
    cache[image_name] = luma / mean if mean > 1e-4 else None
    return cache[image_name]


def tinted_grain(out_dir, image_name, colour, size=256):
    """Write (and reuse) the tileable photo tinted to one colour."""
    tag = '%s_%d_%d_%d' % (Path(image_name).stem[:6],
                           *(int(round(c * 255)) for c in colour))
    name = 'ACG_%s.png' % hashlib.md5(tag.encode('utf8')).hexdigest()[:10]
    existing = bpy.data.images.get(name)
    if existing is not None:
        return existing
    grain = grain_multiplier(image_name, size)
    if grain is None:
        return None
    # Clipping would both blow out the grain and drag the colour towards
    # white, so the contrast is scaled back to whatever this particular
    # shade can carry rather than clamped afterwards.
    brightest = max(colour)
    headroom = float(grain.max()) - 1.0
    strength = GRAIN_STRENGTH
    if brightest > 1e-4 and headroom > 1e-4:
        strength = min(strength, (1.0 / brightest - 1.0) / headroom)
    multiplier = 1.0 + (grain - 1.0) * max(strength, 0.0)
    tinted = np.clip(multiplier[:, :, None] * np.array(colour, dtype=np.float32),
                     0.0, 1.0)
    rgba = np.ones((size, size, 4), dtype=np.float32)
    rgba[:, :, :3] = tinted
    img = bpy.data.images.new(name, size, size, alpha=False)
    img.pixels.foreach_set(rgba.ravel())
    img.file_format = 'PNG'
    img.filepath_raw = str(out_dir / name)
    img.save()
    return img


def cube_project(ob, tile):
    """World-aligned box UVs, written straight into the mesh.

    1321 of the object-materials built from noise carry no UV layer at all,
    and a texture on a mesh without one samples a single texel: the grain
    would arrive as another flat colour. Projecting along whichever world
    axis each face most faces keeps brick courses level on a wall and flat
    on a roof, and tiling in metres means every building shares one scale.
    """
    mesh = ob.data
    if mesh.uv_layers or not mesh.polygons:
        return False
    matrix = np.array(ob.matrix_world.to_4x4())
    co = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
    mesh.vertices.foreach_get('co', co)
    world = co.reshape(-1, 3) @ matrix[:3, :3].T + matrix[:3, 3]
    normals = np.empty(len(mesh.polygons) * 3, dtype=np.float64)
    mesh.polygons.foreach_get('normal', normals)
    facing = np.abs(normals.reshape(-1, 3) @ matrix[:3, :3].T).argmax(axis=1)
    counts = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get('loop_total', counts)
    vertex = np.empty(len(mesh.loops), dtype=np.int32)
    mesh.loops.foreach_get('vertex_index', vertex)
    # Loops are stored polygon by polygon, so repeating each face's axis by
    # its loop count lines the two up without touching loop_start.
    axis = np.repeat(facing, counts)
    point = world[vertex]
    u = np.where(axis == 0, point[:, 1], point[:, 0])
    v = np.where(axis == 2, point[:, 1], point[:, 2])
    layer = mesh.uv_layers.new(name='ACBox')
    layer.data.foreach_set('uv', np.stack([u / tile, v / tile], 1).ravel())
    return True


def merge_materials(scene, out_dir, buckets=12, grain=True, tile=2.0):
    out_dir = out_dir / 'texture'
    out_dir.mkdir(parents=True, exist_ok=True)
    """Collapse every material onto a shared one per texture.

    The sector carries 284 materials and ksEditor fills each txDiffuse by
    hand, so a build is unusable without collapsing them first. Materials
    that resolve to the same image become one; the flat-colour ones are
    quantised into a few shades so a street of houses does not produce a
    material per wall.
    """
    shared, assignments, alpha = {}, {}, set()
    step = 1.0 / buckets
    projected = 0
    for ob in scene.objects:
        if ob.type != 'MESH' or not ob.select_get():
            continue
        if grain:
            projected += cube_project(ob, tile)
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
                rgb = [(c + 0.5) * step for c in shade]
                source = grain_for(m.name) if grain else None
                textured = (tinted_grain(out_dir, source, rgb)
                            if source else None)
                if textured is not None:
                    # Quantising the shade is what keeps this bounded: a
                    # street of houses shares a handful of tinted copies
                    # rather than one image per wall.
                    key = safe_file(textured.name)
                    image = textured
                else:
                    key = 'ACSHADE_%d_%d_%d.png' % shade
                    image = shared.get(key, (None, None))[1]
                if image is None:
                    image = bpy.data.images.new(key, 8, 8, alpha=False)
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
                if has_alpha(image):
                    alpha.add(key)
            slot.material = shared[key][0]
    if projected:
        print('AC_UV_BOX objects=%d tile=%.1fm' % (projected, tile))
    return assignments, alpha


def material_order(scene, assignments):
    """Materials as ksEditor numbers them: first seen walking the scene.

    Sorting them by name instead made Load Persistence add a second set of
    materials beside the imported ones rather than filling those in, which
    looks like it keys the sections by position, not by NAME.
    """
    ordered, seen = [], set()
    objects = sorted((o for o in scene.objects if o.select_get()),
                     key=lambda o: o.name)
    for ob in objects:
        if ob.type != 'MESH':
            continue
        for slot in ob.material_slots:
            m = slot.material
            if m is None or m.name in seen or m.name not in assignments:
                continue
            seen.add(m.name)
            ordered.append((m.name, assignments[m.name]))
    for name in sorted(set(assignments) - seen):
        ordered.append((name, assignments[name]))
    return ordered


# ksEditor refuses a mesh over 65536 vertices, and what it counts is the
# exported geometry. Modifiers run first — 492 objects carry a Bevel — and
# then the writer splits a vertex per unique normal and UV, so counting
# mesh vertices underestimates wildly: a 30000-vertex slice of terrain
# arrived as 108169. Face corners are the ceiling that expansion cannot
# exceed, so batches are measured in those.
VERTEX_LIMIT = 60000


def build_part(ob, source, faces, name, scene):
    """A new object holding just these faces of `ob`.

    Built vertex by vertex rather than by copying the mesh and deleting the
    rest: a half-million-vertex canopy needs eight parts, and copying it
    eight times costs minutes and gigabytes.
    """
    remap, coords, polygons = {}, [], []
    for index in faces:
        poly = source.polygons[index]
        for vertex in poly.vertices:
            if vertex not in remap:
                remap[vertex] = len(coords)
                coords.append(source.vertices[vertex].co[:])
        polygons.append([remap[v] for v in poly.vertices])
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(coords, [], polygons)
    for slot in ob.material_slots:
        mesh.materials.append(slot.material)
    for position, index in enumerate(faces):
        mesh.polygons[position].material_index = source.polygons[index].material_index
    for layer in source.uv_layers:
        target = mesh.uv_layers.new(name=layer.name)
        loop = 0
        for index in faces:
            poly = source.polygons[index]
            for offset in range(poly.loop_total):
                target.data[loop].uv = layer.data[poly.loop_start + offset].uv
                loop += 1
    part = bpy.data.objects.new(name, mesh)
    part.matrix_world = ob.matrix_world
    scene.collection.objects.link(part)
    return part


def split_large_meshes(scene, limit=VERTEX_LIMIT):
    """Break meshes that exceed ksEditor's per-mesh vertex ceiling.

    The limit is 65536 — a 16-bit index buffer — and a mesh over it is
    dropped on import with 'The mesh will not be loaded', silently losing
    whole categories of scenery: the garden canopies alone carry 498748
    vertices. Faces are gathered into batches that stay under the cap and
    each batch becomes its own object, so UVs, materials and the physics
    prefix in the name all survive.
    """
    split = []
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for ob in list(scene.objects):
        if ob.type != 'MESH' or not ob.select_get():
            continue
        # Measure and cut the mesh the exporter will actually write, with
        # modifiers applied; parts carry no modifiers of their own, so a
        # part built from the base mesh would silently lose its bevel.
        source = bpy.data.meshes.new_from_object(
            ob.evaluated_get(depsgraph), depsgraph=depsgraph)
        if len(source.loops) <= limit:
            bpy.data.meshes.remove(source)
            continue
        batches, batch, corners = [], [], 0
        for poly in source.polygons:
            if batch and corners + poly.loop_total > limit:
                batches.append(batch)
                batch, corners = [], 0
            batch.append(poly.index)
            corners += poly.loop_total
        if batch:
            batches.append(batch)
        if not batches:
            bpy.data.meshes.remove(source)
            continue
        # A single batch still means a rebuild: the mesh carries vertices no
        # polygon uses, and only rebuilding drops them. V12_Architecture_
        # gutter_001 reaches 85680 that way and keeps failing to load.
        for index, faces in enumerate(batches):
            part = build_part(ob, source, faces,
                              '%s_P%d' % (ob.name[:50], index), scene)
            part.data.update()
            part.select_set(True)
        ob.select_set(False)
        split.append({'object': ob.name, 'corners': len(source.loops),
                      'parts': len(batches)})
        bpy.data.meshes.remove(source)
    return split


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


def write_persistence(fbx, assignments, scene=None, alpha=()):
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
    for index, (material, texture) in enumerate(material_order(scene, assignments)):
        cut = texture in alpha
        # ksPerPixel has no alpha-test path at all, so ALPHATEST=1 on it is
        # ignored and a leaf card's transparent texels draw their own RGB --
        # black. The AT variant is the same shader with the cutout branch,
        # and it reads the threshold from ksAlphaRef below.
        lines += ['[MATERIAL_%d]' % index, 'NAME=' + material,
                  'SHADER=%s' % ('ksPerPixelAT' if cut else 'ksPerPixel'),
                  'ALPHABLEND=0',
                  'ALPHATEST=%d' % (1 if cut else 0),
                  'DEPTHMODE=0', 'VARCOUNT=%d' % len(PERSISTENCE_VARS)]
        for slot, (var, value) in enumerate(PERSISTENCE_VARS):
            if cut and var == 'ksAlphaRef':
                value = 0.35
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


# --- whole-lap builds -------------------------------------------------
#
# The lap is 5709 m and the scene carries 18.8M triangles with modifiers
# applied. ksEditor is a 32-bit program, so one KN5 cannot hold it and the
# track has to arrive as several, listed in models.ini. Cutting by station
# rather than by a corridor around a sector is what makes the pieces fit
# together: every face belongs to exactly the chunk whose station band its
# nearest axis point falls in, so neighbours share no geometry and leave no
# gap. A corridor radius would overlap at each seam and z-fight there.


# Landmarks stand back from the road by their nature, so a corridor chosen
# from how dense the housing is cuts them off. The grain elevator reaches
# 262 m from the racing line and the memorial 39 m; at a 45 m corridor the
# elevator's towers simply vanished, which is the one thing nobody driving
# the lap would fail to notice. They are cheap to keep whole -- the
# elevator is 3916 triangles and the tank 3754 -- so distance stops
# applying to them. The station band still does, or they would be written
# into every chunk at once.
LANDMARK = re.compile(r'Elevator|Industrial|MEM_|Memorial', re.I)


def is_landmark(ob, pattern):
    if pattern.search(ob.name):
        return True
    return any(s.material and pattern.search(s.material.name)
               for s in ob.material_slots)


def axis_arrays(rows, step=10.0):
    """The racing axis as numpy columns: positions and their chainage.

    Thinned first. The raw axis is sampled every half metre, which is 11475
    points, and every face centre is compared against all of them; against a
    60 m corridor that precision buys nothing and costs twenty times the
    work. A 10 m step places a face within 5 m of its true chainage, far
    inside the band it has to be sorted into.
    """
    kept, last = [], None
    for r in rows:
        if last is None or r['map_s_m'] - last >= step:
            kept.append(r)
            last = r['map_s_m']
    if kept[-1]['map_s_m'] < rows[-1]['map_s_m']:
        kept.append(rows[-1])
    xy = np.array([(r['x'], r['y']) for r in kept], dtype=np.float64)
    s = np.array([r['map_s_m'] for r in kept], dtype=np.float64)
    return xy, s


def nearest_on_axis(points, axis_xy, axis_s, batch=20000):
    """For each point, the chainage of the closest axis sample and the gap.

    Batched because the full product is 2M faces by 11475 samples; twenty
    thousand rows at a time keeps it inside a few hundred megabytes while
    staying one vectorised pass per batch instead of a Python loop.
    """
    stations = np.empty(len(points), dtype=np.float64)
    gaps = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), batch):
        block = points[start:start + batch]
        dx = block[:, None, 0] - axis_xy[None, :, 0]
        dy = block[:, None, 1] - axis_xy[None, :, 1]
        d2 = dx * dx + dy * dy
        pick = d2.argmin(axis=1)
        stations[start:start + batch] = axis_s[pick]
        gaps[start:start + batch] = np.sqrt(d2[np.arange(len(block)), pick])
    return stations, gaps


def face_centres(ob):
    """World-space centres of every polygon, read out at C speed."""
    mesh = ob.data
    flat = np.empty(len(mesh.polygons) * 3, dtype=np.float64)
    mesh.polygons.foreach_get('center', flat)
    local = flat.reshape(-1, 3)
    m = np.array(ob.matrix_world.to_4x4())
    return local @ m[:3, :3].T + m[:3, 3]


def in_band(station, a, b, lap):
    """Chainage inside [a, b), wrapping once past the finish line."""
    if b <= lap:
        return (station >= a) & (station < b)
    return (station >= a) | (station < b - lap)


def trim_to_band(ob, axis_xy, axis_s, a, b, lap, corridor):
    """Keep only the faces this chunk owns. None when nothing survives."""
    mesh = ob.data = ob.data.copy()   # may be shared; never touch the original
    if not mesh.polygons:
        return None
    station, gap = nearest_on_axis(face_centres(ob), axis_xy, axis_s)
    keep = in_band(station, a, b, lap) & (gap <= corridor)
    if keep.all():
        return False
    if not keep.any():
        return None
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    doomed = [bm.faces[i] for i in np.flatnonzero(~keep)]
    bmesh.ops.delete(bm, geom=doomed, context='FACES')
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return True


def object_touches_band(ob, axis_xy, axis_s, a, b, lap, corridor):
    """Cheap bounding-box test before paying for a per-face pass."""
    corners = np.array([(ob.matrix_world @ Vector(c))[:2] for c in ob.bound_box])
    station, gap = nearest_on_axis(corners, axis_xy, axis_s)
    span = np.hypot(np.ptp(corners[:, 0]), np.ptp(corners[:, 1]))
    if gap.min() > corridor + span:
        return False
    # A mesh spanning the whole lap has corners nowhere near its own middle,
    # so accept it whenever any corner is in the band or it is simply large.
    return bool(in_band(station, a, b, lap).any() or span > corridor)


# Tree trunks are the single heaviest thing in the scene: the two bark
# materials carry 5.7M of the 9.0M triangles the vegetation spends, about
# 2850 per trunk, where a game tree uses a couple of hundred. The leaves
# cost far less and are what the eye actually reads, so only the bark is
# reduced -- and it cannot be done per object, because 2929 of the 3042
# plants hold bark and leaves in one mesh. A Decimate modifier bound to a
# vertex group collapses the bark alone and leaves every alpha card whole.

# Not 'Wood': that also catches wooden fences and doors, and collapsing a
# plank gate to a fifth of its faces makes a mess of it.
BARK = re.compile(r'Bark|Trunk', re.I)


def thin_tree_trunks(scene, ratio):
    """Collapse bark geometry to `ratio`, leaving foliage untouched."""
    thinned = 0
    for ob in scene.objects:
        if ob.type != 'MESH' or not ob.data.polygons or not ob.select_get():
            continue
        slots = [i for i, s in enumerate(ob.material_slots)
                 if s.material and BARK.search(s.material.name)]
        if not slots:
            continue
        mesh = ob.data
        index = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get('material_index', index)
        wanted = np.isin(index, slots)
        if not wanted.any():
            continue
        starts = np.empty(len(mesh.polygons), dtype=np.int32)
        counts = np.empty(len(mesh.polygons), dtype=np.int32)
        mesh.polygons.foreach_get('loop_start', starts)
        mesh.polygons.foreach_get('loop_total', counts)
        loops = np.empty(len(mesh.loops), dtype=np.int32)
        mesh.loops.foreach_get('vertex_index', loops)
        picked = np.zeros(len(mesh.loops), dtype=bool)
        for start, total in zip(starts[wanted], counts[wanted]):
            picked[start:start + total] = True
        vertices = np.unique(loops[picked])
        group = ob.vertex_groups.new(name='ACBark')
        group.add([int(v) for v in vertices], 1.0, 'REPLACE')
        modifier = ob.modifiers.new(name='ACThin', type='DECIMATE')
        modifier.decimate_type = 'COLLAPSE'
        modifier.ratio = ratio
        modifier.vertex_group = group.name
        modifier.use_collapse_triangulate = True
        thinned += 1
    return thinned


def strip_distant_bevels(scene, axis_xy, axis_s, keep_within):
    """Turn off Bevel past `keep_within` metres of the racing line.

    Bevels cost 1.94M of the scene's 18.8M triangles, 1.42M of that on
    buildings alone. A 2 cm chamfer is worth its triangles on the kerb the
    car is about to touch and worth nothing on a roof ridge thirty metres
    away, so distance decides rather than a blanket switch.
    """
    dropped = objects = 0
    for ob in scene.objects:
        if ob.type != 'MESH' or not ob.select_get():
            continue
        bevels = [m for m in ob.modifiers if m.type == 'BEVEL' and m.show_viewport]
        if not bevels:
            continue
        centre = np.array([(ob.matrix_world @
                            (sum((Vector(c) for c in ob.bound_box), Vector()) / 8.0))[:2]])
        _, gap = nearest_on_axis(centre, axis_xy, axis_s)
        if gap[0] <= keep_within:
            continue
        for m in bevels:
            m.show_viewport = m.show_render = False
        dropped += len(bevels)
        objects += 1
    return dropped, objects


def write_models_ini(path, files):
    """The list AC reads to load several KN5 as one track.

    The first entry is the track's own name and carries the AC_ nulls:
    spawn objects held in a secondary KN5 are not picked up.
    """
    lines = []
    for index, name in enumerate(files):
        lines += ['[MODEL_%d]' % index, 'FILE=%s' % name,
                  'POSITION=0,0,0', 'ROTATION=0,0,0', '']
    path.write_text('\n'.join(lines), encoding='utf8', newline='\r\n')
    return path


def place_nulls(scene, rows, a):
    """Grid slots, pit boxes and the start/finish timing gate."""
    origin, heading = at_station(rows, a.start_station)
    side = Vector((-heading.y, heading.x, 0.0))
    # A grid rather than a lone slot: one AC_START_0 leaves no room for a
    # second car, so neither opponents nor a friend can join. Cars alternate
    # sides of the centreline and step back along it, the way a real grid is
    # laid out; the pit boxes queue up further behind.
    lift = Vector((0, 0, a.spawn_lift))
    for slot in range(a.grid):
        row, left = divmod(slot, 2)
        offset = side * (a.grid_offset if left else -a.grid_offset)
        null(scene, 'AC_START_%d' % slot,
             origin + offset - heading * (row * a.grid_spacing) + lift,
             heading, a.flip)
        null(scene, 'AC_PIT_%d' % slot,
             origin + offset - heading * (40.0 + slot * a.grid_spacing) + lift,
             heading, a.flip)
    null(scene, 'AC_TIME_0_L', origin + side * (a.gate_width / 2.0))
    null(scene, 'AC_TIME_0_R', origin - side * (a.gate_width / 2.0))
    return origin


def select_sector(scene, rows, a):
    """The original corridor-around-a-sector selection."""
    points = sector_points(rows, a.sector_from, a.sector_to) if a.mode == 'sector' else None
    bounds = corridor_bounds(points, a.corridor) if points else None
    kept = culled = triangles = trimmed = 0
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
    return kept, culled, triangles, trimmed


def select_band(scene, rows, a, band, spine, scenery, landmarks=LANDMARK):
    """Station-banded selection for a whole-lap chunk.

    `spine` takes the drivable surfaces for the entire lap regardless of
    band, because the road has to be one unbroken piece of physics; the
    scenery is what the band divides up.
    """
    axis_xy, axis_s = axis_arrays(rows)
    lap = float(axis_s.max())
    a0, a1 = band if band else (0.0, lap)
    kept = culled = triangles = trimmed = 0
    for ob in scene.objects:
        if ob.type not in ('MESH', 'EMPTY'):
            continue
        if ob.type == 'EMPTY':
            ob.select_set(ob.name.startswith('AC_'))
            kept += ob.select_get()
            continue
        is_spine = bool(SPINE.match(ob.name))
        take = False
        if is_spine:
            take = spine
        elif scenery:
            reach = (float('inf') if landmarks and is_landmark(ob, landmarks)
                     else a.corridor)
            if object_touches_band(ob, axis_xy, axis_s, a0, a1, lap, reach):
                result = trim_to_band(ob, axis_xy, axis_s, a0, a1, lap, reach)
                take = result is not None
                trimmed += bool(result)
        ob.select_set(take)
        if take:
            triangles += sum(len(p.vertices) - 2 for p in ob.data.polygons)
        kept, culled = kept + bool(take), culled + (not take)
    return kept, culled, triangles, trimmed


def evaluated_triangles(scene):
    """Triangles as the FBX writer will see them, with modifiers applied.

    The selection pass counts the stored mesh, which is the wrong number to
    plan chunks with: bevels add 1.94M triangles the count never sees and
    the trunk decimation removes several million it still counts. Only a
    dry run pays for this, where being right matters more than being quick.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    total = 0
    for ob in scene.objects:
        if ob.type != 'MESH' or not ob.select_get():
            continue
        evaluated = ob.evaluated_get(depsgraph)
        try:
            mesh = evaluated.to_mesh()
        except RuntimeError:
            continue
        if mesh:
            counts = np.empty(len(mesh.polygons), dtype=np.int32)
            mesh.polygons.foreach_get('loop_total', counts)
            total += int(np.maximum(counts - 2, 0).sum())
        evaluated.to_mesh_clear()
    return total


def build_one(a, scene_path, out, band=None, spine=True, scenery=True, nulls=True):
    """One FBX plus its persistence file.

    The scene is reopened for every chunk: trimming replaces mesh data in
    place, so a chunk built after another would inherit its neighbour's
    cuts and come out empty.
    """
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    scene = bpy.context.scene
    rows = line_samples(a.racing_line or scene_path.parent / LINE_REL)
    applied, visual = rename(scene)
    origin = place_nulls(scene, rows, a) if nulls else None

    for ob in scene.objects:
        ob.select_set(False)
    if a.mode == 'lap':
        kept, culled, triangles, trimmed = select_band(
            scene, rows, a, band, spine, scenery,
            landmarks=re.compile(a.landmarks, re.I) if a.landmarks else None)
    else:
        kept, culled, triangles, trimmed = select_sector(scene, rows, a)

    # After the selection, not before: thinning every tree in the scene for
    # a chunk that exports the road alone is four scene-wide decimations
    # that reach no FBX. Both still run ahead of the split, which measures
    # the mesh with its modifiers applied.
    trunks = thin_tree_trunks(scene, a.trunk_ratio) if a.trunk_ratio < 1.0 else 0
    bevels = (0, 0)
    if a.drop_bevel_beyond > 0:
        axis_xy, axis_s = axis_arrays(rows)
        bevels = strip_distant_bevels(scene, axis_xy, axis_s, a.drop_bevel_beyond)

    if not kept:
        return {'fbx': None, 'exported_objects': 0, 'exported_triangles': 0,
                'band': band, 'skipped': 'nothing in this band'}

    split = split_large_meshes(scene)
    out.parent.mkdir(parents=True, exist_ok=True)
    textures, skipped_textures = unpack_textures(out.parent, a.keep_hdr)
    if a.merge:
        chosen, alpha = merge_materials(scene, out.parent,
                                        grain=a.grain, tile=a.grain_tile)
        wired, flats, no_bsdf = [], [], []
    else:
        wired, flats, no_bsdf, chosen, alpha = wire_base_colour(scene, out.parent)

    if a.dry_run:
        triangles = evaluated_triangles(scene)
        print('AC_CHUNK_DRY %s objects=%d triangles=%d materials=%d '
              'bevels_off=%d trunks=%d'
              % (out.name, kept, triangles, len(chosen), bevels[0], trunks))
        return {'fbx': str(out), 'exported_objects': kept, 'band': band,
                'exported_triangles': triangles, 'materials': len(chosen),
                'bevel_modifiers_off': bevels[0], 'bevel_objects': bevels[1],
                'trunks_thinned': trunks, 'dry_run': True}

    bpy.ops.export_scene.fbx(
        filepath=str(out), use_selection=True, apply_unit_scale=True,
        global_scale=a.scale, axis_forward=a.axis_forward, axis_up=a.axis_up,
        object_types={'MESH', 'EMPTY'}, use_mesh_modifiers=True,
        mesh_smooth_type='FACE', path_mode='STRIP', embed_textures=False,
        bake_space_transform=a.bake_transform)
    persistence = write_persistence(out, chosen, scene, alpha) if chosen else None

    print('AC_CHUNK %s objects=%d culled=%d triangles=%d trimmed=%d split=%d '
          'materials=%d bevels_off=%d MB=%.1f'
          % (out.name, kept, culled, triangles, trimmed, len(split), len(chosen),
             bevels[0], out.stat().st_size / 1e6))
    return {
        'fbx': str(out), 'band': band, 'renamed': len(applied),
        'visual_only_meshes': visual, 'exported_objects': kept,
        'culled_objects': culled, 'exported_triangles': triangles,
        'trimmed_meshes': trimmed, 'split_meshes': split,
        'bevel_modifiers_off': bevels[0], 'bevel_objects': bevels[1],
        'trunks_thinned': trunks, 'trunk_ratio': a.trunk_ratio,
        'alpha_tested': sorted(alpha), 'materials': len(chosen),
        'persistence': str(persistence) if persistence else None,
        'textures_written': textures, 'textures_skipped': skipped_textures,
        'base_colour_wired': wired, 'base_colour_flat': flats,
        'materials_without_principled': no_bsdf,
        'start_location': list(origin) if origin else None,
        'megabytes': round(out.stat().st_size / 1e6, 1),
    }


def main(a):
    scene_path = Path(a.scene).resolve()
    out = Path(a.output).resolve() if a.output else scene_path.parent / 'ac_export_out/chmyrivka.fbx'
    report_path = Path(a.report).resolve() if a.report else out.parent / 'export_report.json'
    started = time.time()

    if a.mode != 'lap':
        chunks = [build_one(a, scene_path, out)]
        models = None
    else:
        rows = line_samples(a.racing_line or scene_path.parent / LINE_REL)
        lap = max(r['map_s_m'] for r in rows)
        # The track's own name goes on the drivable surfaces: AC reads the
        # AC_ nulls only from the first model, and the road has to stay one
        # piece of physics, so the bands divide the scenery alone.
        # The bands only depend on --chunks, so one chunk can be rebuilt on
        # its own and dropped in beside the others: a change to one street
        # costs a minute and one pass through ksEditor instead of seven.
        edges = [lap * i / a.chunks for i in range(a.chunks + 1)]
        wanted = range(a.chunks + 1) if a.only_chunk is None else [a.only_chunk]
        chunks = []
        for index in wanted:
            if index == 0:
                chunks.append(build_one(a, scene_path, out, band=None,
                                        spine=True, scenery=False, nulls=True))
                continue
            part = out.with_name('%s_s%d%s' % (out.stem, index, out.suffix))
            chunks.append(build_one(a, scene_path, part,
                                    band=(edges[index - 1], edges[index]),
                                    spine=False, scenery=True, nulls=False))
        models = None
        if not a.dry_run:
            # Named from --chunks rather than from what was built, so a
            # single-chunk rebuild does not shrink the list to one model.
            names = [out.with_suffix('.kn5').name] + [
                out.with_name('%s_s%d.kn5' % (out.stem, i)).name
                for i in range(1, a.chunks + 1)]
            models = str(write_models_ini(out.parent / 'models.ini', names))

    report = {
        'mode': a.mode, 'scene': str(scene_path), 'chunks': chunks,
        'models_ini': models, 'corridor': a.corridor,
        'drop_bevel_beyond': a.drop_bevel_beyond,
        'start_station_map_m': a.start_station, 'flip_spawn': a.flip,
        'grid_slots': a.grid, 'dry_run': a.dry_run,
        'sector_map_m': [a.sector_from, a.sector_to] if a.mode == 'sector' else None,
        'assetto_corsa_tested': False,
        'seconds': round(time.time() - started, 1),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    total = sum(c.get('exported_triangles', 0) for c in chunks)
    print('AC_EXPORT_DONE chunks=%d triangles=%d seconds=%.0f models_ini=%s'
          % (len(chunks), total, time.time() - started, models or '-'))
    print('AC_EXPORT_OUT %s' % out.parent)


if __name__ == '__main__':
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True)
    p.add_argument('--mode', choices=['sector', 'full', 'lap'], default='sector',
                   help="lap cuts the whole circuit into --chunks KN5 and writes models.ini")
    p.add_argument('--chunks', type=int, default=6,
                   help='scenery KN5 for mode lap; the drivable surfaces get one of their own')
    p.add_argument('--landmarks', default=LANDMARK.pattern,
                   help='objects matching this (by name or material) ignore --corridor, '
                        'because landmarks stand back from the road; empty string '
                        'subjects them to the corridor like everything else')
    p.add_argument('--only-chunk', type=int, default=None, dest='only_chunk',
                   help='rebuild just this chunk of mode lap: 0 is the road, '
                        '1..N the scenery bands. The bands come from --chunks, '
                        'so pass the same value you built the rest with')
    p.add_argument('--dry-run', action='store_true', dest='dry_run',
                   help='count what each chunk would hold without writing FBX')
    p.add_argument('--drop-bevel-beyond', type=float, default=0.0,
                   dest='drop_bevel_beyond',
                   help='metres from the racing line past which Bevel is switched off; '
                        '0 keeps every bevel')
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
    p.add_argument('--grid', type=int, default=8,
                   help='starting slots and pit boxes; one leaves nobody to race')
    p.add_argument('--grid-spacing', type=float, default=9.0, dest='grid_spacing')
    p.add_argument('--grid-offset', type=float, default=2.6, dest='grid_offset')
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
    p.add_argument('--trunk-ratio', type=float, default=1.0, dest='trunk_ratio',
                   help='collapse tree bark to this fraction of its faces; '
                        '1.0 leaves trunks alone, 0.2 is a game-weight trunk')
    p.add_argument('--no-grain', action='store_false', dest='grain',
                   help='leave colour-only materials as a flat swatch instead of '
                        'tinting a tileable photo and box-projecting UVs for it')
    p.add_argument('--grain-tile', type=float, default=2.0, dest='grain_tile',
                   help='metres per texture tile for the box projection')
    p.add_argument('--road-only', action='store_true', dest='road_only',
                   help='drivable surfaces alone, no scenery: smallest file that still drives')
    p.add_argument('--keep-hdr', action='store_true', dest='keep_hdr',
                   help='also write the 4K sky HDR, unused by AC')
    p.add_argument('--racing-line', dest='racing_line', default=None,
                   help='racing_line_native_v15.json; defaults to %s beside the scene' % LINE_REL)
    main(p.parse_args(argv))
