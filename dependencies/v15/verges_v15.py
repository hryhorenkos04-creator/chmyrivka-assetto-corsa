"""Grounded, batched verge detail for the native Sinna scene.

apply(scene, collection, source_data=None) only adds new V15 objects/materials.
The 3800--4250 m chainage is the ORIGINAL map chainage. Planting and worn soil
are regional art direction, never botanical survey or verified plot evidence.
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import random

import bpy
import numpy as np
import shapely
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from shapely.geometry import Point, Polygon, LineString, MultiPoint, shape, mapping
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT.parent / 'chmyrivka_detail_v13' / 'data'
SEED = 150641
S0, S1 = 3800.0, 4250.0
LIMIT_TRIS = 120000
UP = Vector((0, 0, 1))


def _geom(value):
    if not value:
        return Polygon()
    return shape(value) if isinstance(value, dict) else Polygon(value)


def _parts(geometry):
    return [p for p in shapely.get_parts(geometry) if p.geom_type == 'Polygon' and p.area > 1e-7]


def _load(source, filename):
    value = source.get(filename) or source.get(filename.removesuffix('.json'))
    if isinstance(value, dict):
        return value
    path = Path(source.get('source_data_dir', DEFAULT_SOURCE)) / filename
    return json.loads(path.read_text(encoding='utf8'))


class Ground:
    def __init__(self, scene, region):
        vertices, faces = [], []
        self.polys, self.planes, self.names = [], [], []
        self.ray_misses = 0
        for ob in scene.objects:
            if ob.type != 'MESH':
                continue
            if not (ob.name.startswith('GROUND_') or ob.name in ('ROAD_SHOULDERS', 'V09_Side_road_verges')):
                continue
            if any(w in ob.name.lower() for w in ('water', 'wetland')):
                continue
            box = MultiPoint([tuple(ob.matrix_world @ Vector(v))[:2] for v in ob.bound_box]).convex_hull
            if not box.intersects(region):
                continue
            ob.data.calc_loop_triangles()
            world = [ob.matrix_world @ v.co for v in ob.data.vertices]
            used = False
            for tri in ob.data.loop_triangles:
                xyz = np.array([world[i] for i in tri.vertices], dtype=float)
                p = Polygon(xyz[:, :2])
                if p.area < 1e-8 or not p.intersects(region):
                    continue
                base = len(vertices)
                vertices.extend(tuple(v) for v in xyz)
                faces.append((base, base+1, base+2))
                self.polys.append(p)
                self.planes.append(np.linalg.solve(np.c_[xyz[:, :2], np.ones(3)], xyz[:, 2]))
                used = True
            if used:
                self.names.append(ob.name)
        if not vertices:
            raise ValueError('No native dry terrain found for v15 verge roots.')
        self.bvh = BVHTree.FromPolygons(vertices, faces, all_triangles=True)
        self.tree = STRtree(self.polys)

    def hit(self, x, y):
        h = self.bvh.ray_cast(Vector((x, y, 500)), Vector((0, 0, -1)), 1000)[0]
        if h is None:
            self.ray_misses += 1
        return h


class Batch:
    def __init__(self, name, materials, role):
        self.name, self.materials, self.role = name, materials, role
        self.vertices, self.faces, self.mat_ids, self.colors = [], [], [], []

    def add(self, vertices, faces, material=0, colors=None):
        offset = len(self.vertices)
        self.vertices.extend(tuple(v) for v in vertices)
        self.faces.extend(tuple(offset+i for i in f) for f in faces)
        self.mat_ids.extend([material] * len(faces))
        self.colors.extend(colors or [(1.0, 1.0, 1.0, 1.0)] * len(vertices))

    def finish(self, collection):
        if not self.faces:
            return None
        mesh = bpy.data.meshes.new(self.name + '_mesh')
        mesh.from_pydata(self.vertices, [], self.faces)
        for m in self.materials:
            mesh.materials.append(m)
        for f, m in zip(mesh.polygons, self.mat_ids):
            f.material_index = m
            if self.role in ('verge_grass', 'yard_weeds'):
                f.use_smooth = True
        colour = mesh.color_attributes.new(name='V15_VergeColor', type='FLOAT_COLOR', domain='POINT')
        colour.data.foreach_set('color', np.asarray(self.colors, dtype=np.float32).reshape(-1))
        uv = mesh.uv_layers.new(name='V15_MetreUV')
        uv.data.foreach_set('uv', np.array([(mesh.vertices[lp.vertex_index].co.x / 2,
                                           mesh.vertices[lp.vertex_index].co.y / 2)
                                          for lp in mesh.loops], dtype=np.float32).reshape(-1))
        mesh.update(); mesh.calc_loop_triangles()
        ob = bpy.data.objects.new(self.name, mesh)
        collection.objects.link(ob)
        ob['v15_role'] = self.role
        ob['v15_visual_detail_only'] = True
        ob['placement_evidence'] = 'Inferred local verge/yard details, not exact surveyed plants or stones.'
        ob['export_lod_class'] = 'near_detail'
        ob['v15_report'] = 'data/verges_v15.json'
        return ob


def _basic(name, rgba, roughness=.9):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    mat.diffuse_color = rgba
    p = mat.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = rgba
    p.inputs['Roughness'].default_value = roughness
    p.inputs['Specular IOR Level'].default_value = .17
    return mat


def _materials():
    palette = [(.105, .155, .037, 1), (.163, .190, .055, 1),
               (.230, .221, .099, 1), (.297, .260, .153, 1)]
    grass = []
    for i, color in enumerate(palette):
        mat = _basic(f'V15_Verge_Grass_Earthy_{i}', color)
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        p = nodes.get('Principled BSDF')
        at = nodes.new('ShaderNodeVertexColor'); at.layer_name = 'V15_VergeColor'
        mix = nodes.new('ShaderNodeMixRGB'); mix.blend_type = 'MULTIPLY'
        mix.inputs[0].default_value = 1; mix.inputs[1].default_value = color
        links.new(at.outputs['Color'], mix.inputs[2]); links.new(mix.outputs['Color'], p.inputs['Base Color'])
        # Thin open leaf strips have no volume for a subsurface random walk.
        # Smooth surface normals and a restrained transmitted diffuse lobe keep
        # the bent blades from reading as opaque folded cardboard in closeups.
        p.inputs['Subsurface Weight'].default_value = 0
        translucent=nodes.new('ShaderNodeBsdfTranslucent');translucent.label='Thin leaf transmitted light'
        links.new(mix.outputs['Color'],translucent.inputs['Color'])
        combine=nodes.new('ShaderNodeMixShader');combine.label='Thin leaf 28 percent transmission';combine.inputs[0].default_value=.28
        links.new(p.outputs['BSDF'],combine.inputs[1]);links.new(translucent.outputs[0],combine.inputs[2])
        links.new(combine.outputs[0],nodes.get('Material Output').inputs['Surface'])
        mat['v15_thin_leaf_translucent_weight']=.28
        mat['source'] = 'Source-authored curved blades; four restrained dry/green regional art tones.'
        grass.append(mat)
    stones = [_basic('V15_Verge_Stone_Limestone', (.25, .23, .18, 1)),
              _basic('V15_Verge_Stone_Dark', (.13, .12, .105, 1))]
    soil = _basic('V15_Verge_Exposed_Soil', (.19, .145, .10, 1), .95)
    nodes, links = soil.node_tree.nodes, soil.node_tree.links
    p = nodes.get('Principled BSDF')
    texcoord = nodes.new('ShaderNodeTexCoord')
    for suffix, destination, color_space in [('diff_2k.jpg', 'Base Color', 'sRGB'),
                                              ('rough_2k.jpg', 'Roughness', 'Non-Color')]:
        image = next((im for im in bpy.data.images if im.name.startswith('dirt_' + suffix)), None)
        if image:
            tex = nodes.new('ShaderNodeTexImage'); tex.image = image
            links.new(texcoord.outputs['UV'], tex.inputs['Vector'])
            links.new(tex.outputs['Color'], p.inputs[destination])
    normal_img = next((im for im in bpy.data.images if im.name.startswith('dirt_nor_gl_2k.jpg')), None)
    if normal_img:
        tex = nodes.new('ShaderNodeTexImage'); tex.image = normal_img
        links.new(texcoord.outputs['UV'], tex.inputs['Vector'])
        normal = nodes.new('ShaderNodeNormalMap'); normal.inputs['Strength'].default_value = .30
        links.new(tex.outputs['Color'], normal.inputs['Color']); links.new(normal.outputs['Normal'], p.inputs['Normal'])
    at = nodes.new('ShaderNodeVertexColor'); at.layer_name = 'V15_VergeColor'
    geometry = nodes.new('ShaderNodeNewGeometry')
    noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 17.0
    noise.inputs['Detail'].default_value = 2.0; noise.inputs['Roughness'].default_value = .7
    links.new(geometry.outputs['Position'], noise.inputs['Vector'])
    ramp = nodes.new('ShaderNodeMapRange')
    ramp.inputs['From Min'].default_value = .24; ramp.inputs['From Max'].default_value = .72
    ramp.inputs['To Min'].default_value = .18; ramp.inputs['To Max'].default_value = .82
    links.new(noise.outputs['Fac'], ramp.inputs['Value'])
    multiply = nodes.new('ShaderNodeMath'); multiply.operation = 'MULTIPLY'
    links.new(at.outputs['Alpha'], multiply.inputs[0]); links.new(ramp.outputs['Result'], multiply.inputs[1])
    links.new(multiply.outputs[0], p.inputs['Alpha'])
    soil.surface_render_method = 'DITHERED'
    soil['source'] = 'Existing CC0 dirt maps at two metres per tile; generated geometry feather/alpha edge.'
    soil['future_game_requirement'] = 'Alpha-faded ground overlay requires deliberate ksEditor material translation; not game validated.'
    return grass, stones, soil


def _exclusions(scene, region, source, track, layout):
    roads = _load(source, 'side_road_plan_v09.json')
    pit = _load(source, 'pit_plan_v08.json')
    closures = _load(source, 'junction_closures_v09.json')
    env = _load(source, 'surroundings_local.json')
    road = unary_union([_geom(track['outer_polygon']), _geom(roads['outer_polygon']),
                        _geom(pit['surface_polygon']), _geom(pit['garage_polygon'])])
    reserved = [road.buffer(.38), _geom(closures.get('barrier_footprints')).buffer(.18)]
    buildings, paved, waters = [], [], []
    for f in env['features']:
        g = shape(f['geometry'])
        if not g.intersects(region.buffer(6)):
            continue
        p = f['properties']; tags = p.get('tags', {})
        if p.get('kind') == 'building':
            buildings.append(g)
        elif p.get('kind') == 'road':
            reserved.append(g.buffer(3.3))
        elif p.get('kind') == 'railway':
            reserved.append(g.buffer(3.4))
        if tags.get('natural') in ('water', 'wetland') or tags.get('waterway'):
            waters.append(g.buffer(.3))
    for yard in layout['yards']:
        # Beds also stay clear: the earlier planting remains visible there.
        for key in ('paths', 'driveways', 'pads', 'beds'):
            paved.extend(_geom(g) for g in yard.get(key, []))
    native_paved = []
    for ob in scene.objects:
        if ob.type != 'MESH' or ob.name.startswith('V15_'):
            continue
        relevant = ob.get('v12_role') in ('architecture', 'yard_detail', 'yards') and any(
            k in ob.name.lower() for k in ('step', 'porch', 'stair', 'pad', 'path', 'driveway'))
        if not relevant:
            continue
        points = [tuple(ob.matrix_world @ Vector(v))[:2] for v in ob.bound_box]
        box = MultiPoint(points).convex_hull
        if not box.intersects(region):
            continue
        # A batched object's overall rectangle can cover many plots. Individual
        # projected faces preserve the actual appendage footprints instead.
        world = [ob.matrix_world @ v.co for v in ob.data.vertices]
        faces = []
        for f in ob.data.polygons:
            p = Polygon([world[i][:2] for i in f.vertices])
            if p.is_valid and p.area > .00001 and p.intersects(region):
                faces.append(p)
        if faces:
            g = unary_union(faces)
            paved.append(g)
            native_paved.append({'object': ob.name, 'footprint': mapping(g)})
    house = unary_union(buildings)
    paths = unary_union(paved)
    reserved.extend([house.buffer(.12), paths.buffer(.18), unary_union(waters)])
    return unary_union(reserved), road, house, paths, native_paved


def apply(scene, collection, source_data=None):
    if scene.get('v15_verges_applied'):
        raise RuntimeError('v15 verge detail already applied. Rebuild from immutable source.')
    source = source_data or {}
    track = source.get('track_v08') or _load(source, 'track_plan_v08.json')
    layout = _load(source, 'yard_layout_v12.json')
    samples = track['samples']
    ss = np.array([s['s'] for s in samples]); xy = np.array([(s['x'], s['y']) for s in samples])
    selected = [(float(np.interp(S0, ss, xy[:, 0])), float(np.interp(S0, ss, xy[:, 1])))]
    selected += [tuple(p) for s, p in zip(ss, xy) if S0 < s < S1]
    selected += [(float(np.interp(S1, ss, xy[:, 0])), float(np.interp(S1, ss, xy[:, 1])))]
    axis = LineString(selected)
    region = axis.buffer(50, cap_style=2)
    obstacle, roads, buildings, paths, native_paved = _exclusions(scene, region, source, track, layout)
    safe = prep(region.difference(obstacle))
    ground = Ground(scene, region.buffer(1))
    grass_mats, stone_mats, soil_mat = _materials()
    grass = Batch('V15_Verge_Curved_Grass', grass_mats, 'verge_grass')
    seed = Batch('V15_Verge_Dry_Seed_Stems', grass_mats, 'verge_seed_stems')
    weeds = Batch('V15_Yard_Lanceleaf_Weeds', grass_mats, 'yard_weeds')
    stones = Batch('V15_Verge_Small_Stones', stone_mats, 'verge_stones')
    soil = Batch('V15_Verge_Feathered_Soil', [soil_mat], 'verge_soil_contact')
    rng = random.Random(SEED)
    records, rejected, groups = [], {}, []
    counts = {'grass_tufts': 0, 'grass_blades': 0, 'seed_stems': 0, 'weeds': 0, 'stones': 0, 'soil_patches': 0}
    max_root_delta = 0.0
    max_soil_plane_support_delta = 0.0
    soil_vertex_support_checks = 0

    def record(kind, root, vertices, house=None, extra=None):
        nonlocal max_root_delta
        footprint = MultiPoint([tuple(v)[:2] for v in vertices]).convex_hull
        if footprint.is_empty or not safe.covers(footprint):
            raise AssertionError(f'Full v15 {kind} footprint failed reservation: {root}')
        h = ground.hit(root[0], root[1])
        root_delta = abs(root[2] - h.z) if h else None
        if root_delta is not None:
            max_root_delta = max(max_root_delta, root_delta)
        r = {'kind': kind, 'root_xyz': list(root), 'footprint': mapping(footprint),
             'min_clearance_to_track_pit_side_roads_m': footprint.distance(roads),
             'min_clearance_to_mapped_buildings_m': footprint.distance(buildings),
             'min_clearance_to_yard_paving_beds_m': footprint.distance(paths),
             'root_ground_delta_m': root_delta, 'mapped_house_reference': house}
        if extra:
            r.update(extra)
        records.append(r)

    def add_tuft(x, y, house=None, tall=False):
        if counts['grass_tufts'] >= 1650:
            return False
        if not safe.covers(Point(x, y).buffer(.31)):
            return False
        h = ground.hit(x, y)
        if h is None:
            return False
        all_vertices = []
        palette = rng.choices(range(4), [.34, .34, .23, .09])[0]
        count = rng.randint(8, 11)
        for _ in range(count):
            a = rng.uniform(0, math.tau); radius = rng.uniform(0, .105)
            xx, yy = x + math.cos(a)*radius, y + math.sin(a)*radius
            base = ground.hit(xx, yy)
            if base is None:
                continue
            # Root enters native terrain by 1 mm, eliminating detached cards.
            base.z -= .001
            height = rng.uniform(.105, .32 if tall else .25)
            width = rng.uniform(.010, .024)
            turn = a + rng.uniform(-1.5, 1.5)
            side = Vector((math.cos(turn), math.sin(turn), 0)) * width/2
            lean = Vector((math.cos(a), math.sin(a), 0))*rng.uniform(.025, .14)
            v = []
            for t, width_at in [(0, 1), (.37, .81), (.73, .44)]:
                center = base + UP*height*t + lean*t*t
                v.extend([center-side*width_at, center+side*width_at])
            v.append(base+UP*height+lean)
            colors = [(.57,.57,.57,1)]*2 + [(.87,.87,.87,1)]*2 + [(1.07,1.07,1.07,1)]*3
            grass.add(v, [(0,1,3,2), (2,3,5,4), (4,5,6)], palette, colors)
            all_vertices.extend(v); counts['grass_blades'] += 1
        if not all_vertices:
            return False
        record('grass', h, all_vertices, house, {'max_height_m': max(v.z-h.z for v in all_vertices)})
        counts['grass_tufts'] += 1
        if rng.random() < .17:
            height = rng.uniform(.23, .35)
            a = rng.uniform(0, math.tau); side = Vector((math.cos(a), math.sin(a), 0))*.0022
            base = h.copy(); base.z -= .001
            tip = h + UP*height + Vector((math.cos(a), math.sin(a), 0))*.04
            middle = base.lerp(tip, .57)
            v = [base-side, base+side, middle+side, middle-side, tip]
            seed.add(v, [(0,1,2,3), (3,2,4)], 3)
            all_seed = list(v)
            for t in [.74, .84, .94]:
                c = base.lerp(tip, t); cross = side.normalized()*.012
                v = [c-cross, c+UP*.016, c+cross, c-UP*.012]
                seed.add(v, [(0,1,2,3)], 3); all_seed.extend(v)
            record('seed_stem', h, all_seed, house, {'max_height_m': height})
            counts['seed_stems'] += 1
        return True

    def add_weed(x, y, house, yard_poly):
        if counts['weeds'] >= 120 or not safe.covers(Point(x, y).buffer(.27)):
            return False
        if not yard_poly.buffer(-.10).covers(Point(x,y).buffer(.27)):
            return False
        h = ground.hit(x, y)
        if h is None:
            return False
        height = rng.uniform(.24, .46)
        verts = []
        for j in range(5):
            a = rng.uniform(0, math.tau)
            u = Vector((math.cos(a), math.sin(a), 0)); n = Vector((-u.y, u.x, 0))
            root = h + UP*(.03+j*.045)
            end = root + u*rng.uniform(.08, .17) + UP*(height-root.z+h.z)*.55
            mid = root.lerp(end, .55)+UP*.018
            width = rng.uniform(.012,.025)
            vs = [root, mid-n*width, mid+UP*.008, mid+n*width, end]
            weeds.add(vs, [(0,1,2),(0,2,3),(1,4,2),(2,4,3)], 0 if j%2 else 1)
            verts.extend(vs)
        # Broad basal leaves only; no invented flowerheads.
        record('yard_weed', h, verts, house, {'max_height_m': max(v.z-h.z for v in verts), 'inside_inferred_yard': True})
        counts['weeds'] += 1
        return True

    def add_stone(x, y, house):
        if counts['stones'] >= 400 or not safe.covers(Point(x, y).buffer(.025)):
            return False
        # Decorative granules stay behind fences, at least 3 m from all paved
        # public track/side-road reservations; never an escape-area obstruction.
        if roads.distance(Point(x, y)) < 3.0:
            return False
        h = ground.hit(x, y)
        if h is None:
            return False
        radius = rng.uniform(.006, .017)
        angles = [j*math.tau/5 + rng.uniform(-.1,.1) for j in range(5)]
        vv = [h + Vector((math.cos(a)*radius, math.sin(a)*radius, -.002)) for a in angles]
        vv += [h + Vector((math.cos(a)*radius*.61, math.sin(a)*radius*.61, radius*.56)) for a in angles]
        faces = [(j,(j+1)%5,(j+1)%5+5,j+5) for j in range(5)]+[(5,6,7,8,9)]
        stones.add(vv, faces, rng.randrange(2))
        record('small_stone', h, vv, house, {'maximum_diameter_m': 2*radius, 'visual_only': True})
        counts['stones'] += 1
        return True

    def add_patch(x, y, house=None, direction=0):
        nonlocal max_soil_plane_support_delta, soil_vertex_support_checks
        if counts['soil_patches'] >= 150:
            return False
        major, minor = rng.uniform(.34,.80), rng.uniform(.18,.40)
        ring = []
        for j in range(15):
            a = math.tau*j/15; factor = rng.uniform(.77,1.12)
            u, v = math.cos(a)*major*factor, math.sin(a)*minor*factor
            ring.append((x+u*math.cos(direction)-v*math.sin(direction), y+u*math.sin(direction)+v*math.cos(direction)))
        poly = Polygon(ring)
        if not poly.is_valid or not safe.covers(poly):
            return False
        h = ground.hit(x,y)
        if h is None:
            return False
        feather = min(.16, minor*.48)
        inner = poly.buffer(-feather, join_style=1)
        if inner.is_empty:
            return False
        areas = [inner, poly.difference(inner)]
        all_v, local_triangles = [], 0
        for area in areas:
            for idx in ground.tree.query(area, predicate='intersects'):
                cut = area.intersection(ground.polys[idx]); plane = ground.planes[idx]
                for part in _parts(cut):
                    for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
                        coords = list(tri.exterior.coords)[:-1]
                        if not tri.exterior.is_ccw:
                            coords.reverse()
                        vs = [(xx,yy,float(xx*plane[0]+yy*plane[1]+plane[2]+.004)) for xx,yy in coords]
                        # Verify the selected native plane is the upper visible
                        # dry support. Skip hidden duplicate lower terrain faces.
                        c = tri.centroid; hit = ground.hit(c.x,c.y)
                        if hit is None or abs(hit.z-(c.x*plane[0]+c.y*plane[1]+plane[2]))>.003:
                            continue
                        # Confirm each overlay vertex, including all clipped
                        # edge intersections, has the same visible native support.
                        supports = [ground.hit(xx,yy) for xx,yy in coords]
                        if any(hit is None for hit in supports):
                            continue
                        deltas = [abs(hit.z-(v[2]-.004)) for hit,v in zip(supports,vs)]
                        if max(deltas) > .003:
                            continue
                        max_soil_plane_support_delta = max(max_soil_plane_support_delta,max(deltas))
                        soil_vertex_support_checks += len(vs)
                        colors = [(1,1,1,min(1.0,poly.boundary.distance(Point(xx,yy))/feather)) for xx,yy in coords]
                        soil.add(vs, [(0,1,2)], 0, colors); all_v.extend(vs); local_triangles += 1
        if not all_v:
            return False
        record('feathered_soil_patch', h, all_v, house,
               {'source_footprint': mapping(poly), 'fade_width_m': feather,
                'surface_lift_m': .004, 'triangles': local_triangles})
        counts['soil_patches'] += 1
        return True

    # Group origins follow fences and the existing ground/shoulder edge. Avoid
    # equally spaced individual plants and leave long low-density intervals.
    candidates = []
    for yard in sorted(layout['yards'], key=lambda y: (y['station_map_m'], str(y['osm_id']))):
        if not S0 < yard['station_map_m'] < S1:
            continue
        poly = _geom(yard['polygon']); ident = yard['osm_id']
        for line_value in yard.get('fence_lines', []):
            line = LineString(line_value)
            if line.length < 2:
                continue
            for d in np.arange(rng.uniform(.7,2.0), line.length, rng.uniform(3.1,5.7)):
                p = line.interpolate(float(d))
                if axis.distance(p) > 28:
                    continue
                a = line.interpolate(max(0,float(d)-.1)); b = line.interpolate(min(line.length,float(d)+.1))
                angle = math.atan2(b.y-a.y,b.x-a.x)
                for sign in (1,-1):
                    x = p.x-math.sin(angle)*.42*sign
                    y = p.y+math.cos(angle)*.42*sign
                    if poly.buffer(-.28).contains(Point(x,y)) and safe.covers(Point(x,y).buffer(.5)):
                        candidates.append((x,y,angle,ident,poly,True)); break
    # Source track spacing is not used as a repeating spacing pattern.
    for d in np.arange(3,axis.length-3,3.1):
        if rng.random() < .28:
            continue
        p = axis.interpolate(float(d)); a = axis.interpolate(float(d)-.5); b = axis.interpolate(float(d)+.5)
        angle = math.atan2(b.y-a.y,b.x-a.x)
        for sign in (-1,1):
            # Find the first safe narrow band beyond the fully reserved road.
            for distance in np.arange(7.6,15.6,.55):
                x=p.x-math.sin(angle)*distance*sign; y=p.y+math.cos(angle)*distance*sign
                if safe.covers(Point(x,y).buffer(.50)):
                    candidates.append((x,y,angle,None,None,False)); break
    rng.shuffle(candidates)
    # Spatial group minimum prevents layering groups at the same root area.
    group_centers = []
    for x,y,angle,ident,yard_poly,in_yard in candidates:
        if len(groups) >= 195 or counts['grass_tufts'] >= 1650:
            break
        if any((x-a)**2+(y-b)**2 < 1.2**2 for a,b in group_centers):
            continue
        before=counts['grass_tufts']
        for _ in range(rng.randint(9,15)):
            along = rng.gauss(0,.52); across = rng.gauss(0,.23)
            xx=x+math.cos(angle)*along-math.sin(angle)*across
            yy=y+math.sin(angle)*along+math.cos(angle)*across
            if yard_poly is not None and not yard_poly.buffer(-.12).covers(Point(xx,yy).buffer(.31)):
                continue
            add_tuft(xx,yy,ident,tall=in_yard)
        if counts['grass_tufts'] == before:
            continue
        groups.append({'center_xy':[x,y],'mapped_house_reference':ident,'tufts':counts['grass_tufts']-before,
                       'inside_inferred_yard':in_yard})
        group_centers.append((x,y))
        if rng.random() < .78:
            add_patch(x,y,ident,angle)
        if in_yard:
            if rng.random() < .63:
                add_weed(x+math.cos(angle)*.28,y+math.sin(angle)*.28,ident,yard_poly)
            for _ in range(rng.randint(1,5)):
                xx=x+rng.uniform(-.6,.6); yy=y+rng.uniform(-.5,.5)
                if yard_poly.buffer(-.15).contains(Point(xx,yy)):
                    add_stone(xx,yy,ident)
    objects = [ob for batch in (grass,seed,weeds,stones,soil) if (ob := batch.finish(collection))]
    triangles = sum(len(ob.data.loop_triangles) for ob in objects)
    if triangles > LIMIT_TRIS:
        raise AssertionError(f'Verge budget exceeded: {triangles} > {LIMIT_TRIS}')
    # Audit true per-placement hulls, not just stems. Roots are anchored to the
    # unmodified dry native surface. Full polygons are retained for independent
    # corridor/building/paving overlay checks by the assembly validator.
    footprints = [_geom(r['footprint']) for r in records]
    violations = sum(not safe.covers(fp) for fp in footprints)
    report = {
        'version':'v15','module':'verges_v15.py','seed':SEED,
        'sector_original_map_chainage_m':[S0,S1], 'sector_half_width_m':50,
        'counts':dict(counts,groups=len(groups),objects=len(objects),triangles=triangles),
        'triangle_budget':LIMIT_TRIS,'native_ground_objects':ground.names,
        'foliage_shading':{'smooth_curved_blades_and_weeds':True,'thin_leaf_translucent_mix':.28,'volume_subsurface':False,
                           'scope':'Native Blender thin-leaf shading; explicit target game shader translation required.'},
        'ground_ray_misses':ground.ray_misses,'maximum_root_ground_delta_m':max_root_delta,
        'ground_ray_miss_policy':'Rejected placements/soil triangles; exact triangle-edge overlay misses are skipped, never floating geometry.',
        'soil_overlay_vertex_support_checks':soil_vertex_support_checks,
        'maximum_soil_plane_native_support_delta_m':max_soil_plane_support_delta,
        'full_footprint_reservation_violations':violations,
        'minimum_footprint_clearance_to_track_pit_side_roads_m':min(r['min_clearance_to_track_pit_side_roads_m'] for r in records),
        'minimum_footprint_clearance_to_mapped_buildings_m':min(r['min_clearance_to_mapped_buildings_m'] for r in records),
        'minimum_footprint_clearance_to_yard_paving_beds_m':min(r['min_clearance_to_yard_paving_beds_m'] for r in records),
        'native_appendage_pavement_footprints':native_paved,
        'objects':[{'name':ob.name,'role':ob['v15_role'],'vertices':len(ob.data.vertices),
                    'triangles':len(ob.data.loop_triangles),'materials':[m.name for m in ob.data.materials]} for ob in objects],
        'groups':groups,'placements':records,
        'reservation':mapping(obstacle.intersection(region)),
        'changes_to_source_geometry_or_materials':False,
        'limits':['Regional, inferred planting and worn earth, not exact botanical or photographic survey.',
                  'Only original-map 3800–4250 m Sinna sector; no blanket detail coverage claimed.',
                  'Near-detail prototype meshes, not measured game LOD/FPS or finalized ksEditor shaders.',
                  'Feathered soil uses alpha and original CC0 dirt maps; explicit game-material translation remains.',
                  'No new driveable physics or colliders; no obstacles placed on track or paved side streets.']}
    if violations or max_root_delta > .0001:
        raise AssertionError(f'Verge placement validation failed: {violations}; {max_root_delta}')
    out = Path(source.get('output_data_dir', ROOT/'data'))
    out.mkdir(parents=True,exist_ok=True)
    (out/'verges_v15.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    scene['v15_verges_applied'] = True
    scene['v15_verges_triangle_count'] = triangles
    bpy.context.view_layer.update()
    return report
