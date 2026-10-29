"""Independent read-only native v17 audit against immutable full v16.

Run with bpy Python under flock /tmp/chmyrivka_v16_bpy.lock. No file is saved.
Contract arrays enumerate objects by exact name, not prefix wildcards:
 allowed_visibility_changes: {object, hide_render, hide_viewport, reason}
 allowed_material_rebindings: {object, old_materials, new_materials, reason}
 new_objects: {object, role, minimum_asphalt_clearance_m=0,
               minimum_building_clearance_m=0, footprint?, footprint_tolerance_m?}
 Roles roadside/vegetation/yard_detail must avoid all paved and mapped footprints.
 Role attached_detail: associated_building, associated_osm_id, attachment_tolerance_m,
 maximum_above_building_m, minimum_below_building_m bind the sole footprint exception.
 Such geometry must stay beside its specific mapped building, within declared XY/Z
 limits, and cannot enter paved areas or another mapped building.
 Role inherited_subset proves every retained polygon against source_object with
 ordered source_polygon_indices. It inherits pre-existing clearances; this is not
 permission to create new obstacles. Attachment exceptions are explicit only.
 ground_checks: {object, vertex_indices, surfaces, minimum_offset_m,
                 maximum_offset_m}; all declared vertices raycast independently.
 material_checks: {material, required_variable_inputs} for node input evidence.

Preserves all inherited mesh topology, coordinates, normals, UV, custom attributes,
polygon material indices, modifiers, matrices, camera properties, collection state.
Only three known computed normal vectors at exact zero-area terrain faces are
canonicalized; raw custom_normal is compared exactly. This does not test AC.
Original material node graphs must remain exact; modifications use private copies.
Explicit allowed_material_retirements may name orphaned source materials omitted
on save, with every original effective {object,slot_index} user listed. Every such
user must be an explicitly allowed rebinding and no candidate user may remain.
Bitmap assets must be packed or resolve to a present file; packed payloads are hashed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import bpy
import numpy as np
import shapely
from shapely.geometry import Polygon, shape

ROOT = Path(__file__).resolve().parent.parent
SOURCE_SHA = '8e4f2c9c3e1b37de42d9d99edf115a898fa72cb76936dce19020bb02f23b0e00'
PAVED = ('1ROAD_MAIN', '1PIT_SURFACE_V08', 'SIDE_ROADS', 'SIDE_ROADS_V09')
# Explicitly inspected inherited zero-area terrain faces. Blender 4.5.7 returns
# uninitialized/NaN computed normal vectors at these isolated repeated corners.
# Their stored INT16_2D custom_normal attribute is still compared exactly.
UNDEFINED_TERRAIN_NORMALS = {
    78752: (26250, (8930, 8737, 8737)),
    143734: (47911, (32673, 32673, 32671)),
    143893: (47964, (33234, 33234, 33233)),
}


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def arr(seq, prop, width=1, dtype=np.float32):
    data = np.empty(len(seq) * width, dtype=dtype)
    seq.foreach_get(prop, data)
    return data.reshape((-1, width)) if width > 1 else data


def ahash(data):
    return hashlib.sha256(data.tobytes()).hexdigest()


def rna_properties(value):
    result = {}
    for p in value.bl_rna.properties:
        name = p.identifier
        if name == 'rna_type' or p.is_readonly or p.type == 'COLLECTION':
            continue
        try:
            v = getattr(value, name)
            if p.type == 'POINTER':
                v = None if v is None else getattr(v, 'name', None)
            elif getattr(p, 'is_array', False):
                v = list(v)
            elif p.type not in ('BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM'):
                continue
            if isinstance(v, set):
                v = sorted(v)
            result[name] = v
        except (AttributeError, TypeError):
            continue
    return result


def attributes(mesh):
    result = {}
    for a in mesh.attributes:
        if a.name in ('material_index', '.material_index'):
            continue
        values = {}
        if len(a.data):
            for p in a.data[0].bl_rna.properties:
                if p.identifier == 'rna_type' or p.type not in ('BOOLEAN', 'INT', 'FLOAT'):
                    continue
                dtype = {'FLOAT': np.float32, 'INT': np.int32, 'BOOLEAN': np.bool_}[p.type]
                values[p.identifier] = ahash(arr(a.data, p.identifier, p.array_length if p.is_array else 1, dtype))
            if not values:
                raise RuntimeError('Cannot independently hash raw attribute ' + a.name + ' / ' + a.data_type)
        result[a.name] = dict(domain=a.domain, type=a.data_type, count=len(a.data), values=values)
    return result


def mesh_state(obj):
    m = obj.data
    raw = {}
    for collection, prop, width, dtype in (
        ('vertices', 'co', 3, np.float32), ('edges', 'vertices', 2, np.int32),
        ('loops', 'vertex_index', 1, np.int32), ('polygons', 'loop_start', 1, np.int32),
        ('polygons', 'loop_total', 1, np.int32), ('polygons', 'use_smooth', 1, np.bool_),
    ):
        raw[collection + '.' + prop] = ahash(arr(getattr(m, collection), prop, width, dtype))
    normals = arr(m.corner_normals, 'vector', 3)
    original_normal_hash = ahash(normals)
    normal_exceptions = []
    if obj.name == 'GROUND_Terrain_Continuous_v11':
        for corner, (face, expected_indices) in UNDEFINED_TERRAIN_NORMALS.items():
            polygon = m.polygons[face]
            ids = tuple(polygon.vertices)
            pts = np.asarray([m.vertices[i].co[:] for i in ids], dtype=np.float64)
            area = sum(float(np.linalg.norm(np.cross(pts[j] - pts[0], pts[j+1] - pts[0])) / 2)
                       for j in range(1, len(pts)-1))
            if ids != expected_indices or area != 0.0 or corner not in polygon.loop_indices:
                raise RuntimeError('Known zero-area terrain-normal exception no longer matches exact source topology')
            normal_exceptions.append(dict(corner=corner, polygon=face, vertex_indices=list(ids), area_m2=area))
            normals[corner] = 0.0
    raw['corner_normals'] = ahash(normals)
    return {
        'geometry': raw, 'matrix': np.asarray(obj.matrix_world).tolist(),
        'uv': {u.name: ahash(arr(u.data, 'uv', 2)) for u in m.uv_layers},
        'attributes': attributes(m),
        'material_index': ahash(arr(m.polygons, 'material_index', 1, np.int32)),
        'materials': [s.material.name if s.material else None for s in obj.material_slots],
        'material_links': [s.link for s in obj.material_slots],
        'modifiers': [dict(type=v.type, **rna_properties(v)) for v in obj.modifiers],
        'hide_render': obj.hide_render, 'hide_viewport': obj.hide_viewport,
        'counts': [len(m.vertices), len(m.edges), len(m.polygons), len(m.loops)],
        'raw_computed_normal_sha256': original_normal_hash,
        'undefined_normal_canonicalization': normal_exceptions,
    }


def world_vertices(obj):
    v = arr(obj.data.vertices, 'co', 3, np.float64)
    m = np.asarray(obj.matrix_world)
    return v @ m[:3, :3].T + m[:3, 3]


def footprint(obj):
    obj.data.calc_loop_triangles()
    if not len(obj.data.loop_triangles):
        return Polygon()
    v = world_vertices(obj)
    tri = arr(obj.data.loop_triangles, 'vertices', 3, np.int32)
    xy = v[tri, :2]
    parts = shapely.polygons(xy)
    areal = shapely.area(parts) > 1e-10
    selected = list(parts[areal])
    # Vertical cladding can project to a line (or a point). Retain these sets
    # for distance/containment tests instead of accidentally treating them empty.
    flat = xy[~areal]
    if len(flat):
        length = np.ptp(flat, axis=1).max(axis=1)
        if np.any(length > 1e-8):
            selected.extend(shapely.linestrings(flat[length > 1e-8]))
        if np.any(length <= 1e-8):
            selected.extend(shapely.points(flat[length <= 1e-8, 0]))
    return shapely.union_all(selected) if selected else Polygon()


def mapped_building(name):
    return name.startswith('BLDG_') and name.endswith('_walls')


def protected(name):
    return name in PAVED or name in ('ROAD_SHOULDERS', 'V09_Side_road_verges') or name.startswith(('1PIT_', 'V08_PIT_', 'V08_Pit_', 'GROUND_', 'V11_FAR_TERRAIN', 'V09_Closure_')) or name.startswith(('V15_Kerb_', 'V15_Driving_Rubber')) or mapped_building(name)


def layer_visibility(scene):
    result = {}
    for view in scene.view_layers:
        def visit(layer, is_root=False):
            result[view.name + '/' + ('__scene_root__' if is_root else layer.collection.name)] = {
                'exclude': layer.exclude, 'hide_viewport': layer.hide_viewport,
                'holdout': layer.holdout, 'indirect_only': layer.indirect_only,
            }
            for child in layer.children:
                visit(child)
        visit(view.layer_collection, True)
    return result


def variable_sources(socket, seen=None):
    """Graph evidence, not a proof of final pixel variation or game conversion."""
    seen = set() if seen is None else seen
    out = []
    for link in socket.links:
        n = link.from_node
        if n.as_pointer() in seen:
            continue
        seen.add(n.as_pointer())
        if n.type in ('TEX_NOISE', 'TEX_VORONOI', 'TEX_WAVE', 'TEX_BRICK', 'TEX_GABOR'):
            out.append({'node': n.name, 'type': n.type})
        if n.type == 'TEX_IMAGE' and n.image:
            out.append({'node': n.name, 'type': n.type, 'image': n.image.name,
                        'size': list(n.image.size), 'colorspace': n.image.colorspace_settings.name})
        if n.type in ('ATTRIBUTE', 'VERTEX_COLOR'):
            out.append({'node': n.name, 'type': n.type,
                        'attribute': getattr(n, 'attribute_name', getattr(n, 'layer_name', ''))})
        for s in n.inputs:
            if s.is_linked:
                out += variable_sources(s, seen)
    return out


def node_tree_state(tree):
    """Stable serialized RNA and topology; ignores UI-only selection/location."""
    if tree is None:
        return None
    nodes = {}
    for node in tree.nodes:
        props = rna_properties(node)
        for key in ('location', 'width', 'height', 'select', 'show_options', 'show_preview',
                    'show_texture', 'dimensions'):
            props.pop(key, None)
        item = {'type': node.bl_idname, 'properties': props,
                'inputs': [rna_properties(s) for s in node.inputs],
                'outputs': [rna_properties(s) for s in node.outputs]}
        if getattr(node, 'color_ramp', None):
            ramp = node.color_ramp
            item['color_ramp'] = {'properties': rna_properties(ramp),
                                  'elements': [rna_properties(e) for e in ramp.elements]}
        if getattr(node, 'mapping', None) and hasattr(node.mapping, 'curves'):
            item['mapping'] = {'properties': rna_properties(node.mapping),
                'curves': [[rna_properties(p) for p in c.points] for c in node.mapping.curves]}
        for key in ('texture_mapping', 'color_mapping'):
            nested = getattr(node, key, None)
            if nested is not None:
                item[key] = rna_properties(nested)
                ramp = getattr(nested, 'color_ramp', None)
                if ramp is not None:
                    item[key + '_ramp'] = {'properties': rna_properties(ramp),
                        'elements': [rna_properties(e) for e in ramp.elements]}
        nodes[node.name] = item
    links = sorted((l.from_node.name, l.from_socket.identifier, l.to_node.name,
                    l.to_socket.identifier) for l in tree.links)
    return {'nodes': nodes, 'links': links}


def material_state(material):
    props = rna_properties(material)
    # User count / UI status are not shading, and data-block copies can alter tags.
    for key in ('use_fake_user', 'use_extra_user', 'tag'):
        props.pop(key, None)
    return {'properties': props, 'nodes': node_tree_state(material.node_tree)}


def image_state(image):
    packed = []
    if image.packed_files:
        for p in image.packed_files:
            packed.append({'size_bytes': p.packed_file.size,
                           'sha256': hashlib.sha256(bytes(p.packed_file.data)).hexdigest()})
    elif image.packed_file:
        packed.append({'size_bytes': image.packed_file.size,
                       'sha256': hashlib.sha256(bytes(image.packed_file.data)).hexdigest()})
    return {'source': image.source, 'size': list(image.size),
            'colorspace': image.colorspace_settings.name, 'alpha_mode': image.alpha_mode,
            'packed': packed}


def subset_state(obj, polygon_indices):
    """Exact local coordinates, ordered loops, shading and UV on selected faces."""
    mesh = obj.data
    polys = [mesh.polygons[int(i)] for i in polygon_indices]
    loop_ids = np.asarray([i for p in polys for i in p.loop_indices], dtype=np.int32)
    v_ids = np.asarray([i for p in polys for i in p.vertices], dtype=np.int32)
    coords = arr(mesh.vertices, 'co', 3)[v_ids]
    return {
        'matrix': np.asarray(obj.matrix_world).tolist(),
        'face_sizes': [len(p.vertices) for p in polys],
        'positions': ahash(coords),
        'uv': {u.name: ahash(arr(u.data, 'uv', 2)[loop_ids]) for u in mesh.uv_layers},
        'smooth': [p.use_smooth for p in polys],
        'material_index': [p.material_index for p in polys],
        'material_slots': [s.material.name if s.material else None for s in obj.material_slots],
        'modifiers': [dict(type=v.type, **rna_properties(v)) for v in obj.modifiers],
    }


def terrain_bvh(obj):
    from mathutils.bvhtree import BVHTree
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = arr(mesh.vertices, 'co', 3, np.float64)
        matrix = np.asarray(evaluated.matrix_world)
        vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
        triangles = arr(mesh.loop_triangles, 'vertices', 3, np.int32)
        return BVHTree.FromPolygons(vertices.tolist(), triangles.tolist(), all_triangles=True)
    finally:
        evaluated.to_mesh_clear()


def validate(args):
    from mathutils import Vector
    start = time.time()
    source, candidate = Path(args.source).resolve(), Path(args.candidate).resolve()
    contract_path = Path(args.change_report).resolve()
    contract = json.loads(contract_path.read_text())
    report = {'version': 'v17', 'independent': True, 'read_only': True,
              'source': str(source), 'candidate': str(candidate),
              'contract_sha256': filehash(contract_path),
              'assetto_corsa_tested': False, 'errors': [], 'warnings': []}

    def check(ok, message, detail=None):
        if not ok:
            report['errors'].append(dict(message=message, detail=detail))

    source_before, candidate_before = filehash(source), filehash(candidate)
    report['source_sha256'] = source_before
    check(source_before == SOURCE_SHA, 'Full v16 immutable baseline SHA differs', source_before)
    check(contract.get('source_sha256') == source_before, 'Contract must bind exact source SHA')
    check(source != candidate, 'Candidate must be separate from immutable source')
    declared_rows = contract.get('new_objects', [])
    declared = {r['object']: r for r in declared_rows}
    check(len(declared) == len(declared_rows), 'Duplicate declared new object names')
    bpy.ops.wm.open_mainfile(filepath=str(source))
    old = {o.name: mesh_state(o) for o in bpy.context.scene.objects if o.type == 'MESH'}
    old_objects = {o.name: dict(type=o.type, matrix=np.asarray(o.matrix_world).tolist(),
                   hide_render=o.hide_render, hide_viewport=o.hide_viewport)
                   for o in bpy.context.scene.objects}
    old_materials = {m.name: material_state(m) for m in bpy.data.materials}
    old_material_users = {m.name: [] for m in bpy.data.materials}
    for ob in bpy.data.objects:
        for slot_index, slot in enumerate(ob.material_slots):
            if slot.material:
                old_material_users[slot.material.name].append(dict(object=ob.name, slot_index=slot_index))
    for users in old_material_users.values():
        users.sort(key=lambda row: (row['object'], row['slot_index']))
    old_node_groups = {g.name: node_tree_state(g) for g in bpy.data.node_groups
                       if g.bl_idname == 'ShaderNodeTree'}
    old_images = {im.name: image_state(im) for im in bpy.data.images
                  if im.source not in ('VIEWER', 'RENDER_RESULT')}
    old_cameras = {o.name: dict(matrix=np.asarray(o.matrix_world).tolist(), data=rna_properties(o.data))
                   for o in bpy.context.scene.objects if o.type == 'CAMERA'}
    old_collections = {c.name: (c.hide_render, c.hide_viewport) for c in bpy.data.collections}
    old_layers = layer_visibility(bpy.context.scene)
    retained_proofs = {}
    for name, row in declared.items():
        if row.get('role') == 'inherited_subset':
            source_object = bpy.context.scene.objects.get(row.get('source_object', ''))
            ids = row.get('source_polygon_indices', [])
            valid = source_object is not None and source_object.type == 'MESH' and bool(ids)
            if valid:
                valid = all(isinstance(i, int) and 0 <= i < len(source_object.data.polygons) for i in ids)
            check(valid, 'Retained subset requires valid exact source polygon indices', name)
            check(len(set(ids)) == len(ids), 'Retained subset source polygons duplicated', name)
            if valid:
                retained_proofs[name] = subset_state(source_object, ids)
    print('V17_VALIDATOR source snapshot', len(old), 'meshes', flush=True)

    bpy.ops.wm.open_mainfile(filepath=str(candidate))
    scene = bpy.context.scene
    now = {o.name: mesh_state(o) for o in scene.objects if o.type == 'MESH'}
    missing, new = sorted(set(old) - set(now)), sorted(set(now) - set(old))
    check(not missing, 'Inherited mesh objects removed', missing)
    check(not (set(old_objects) - set(scene.objects.keys())), 'Inherited nonmesh objects removed', sorted(set(old_objects) - set(scene.objects.keys())))
    check(set(new) == set(declared), 'New mesh set differs from explicit contract', {'actual': new, 'declared': sorted(declared)})
    vis_rows = contract.get('allowed_visibility_changes', [])
    vis_allowed = {r['object']: r for r in vis_rows}
    check(len(vis_rows) == len(vis_allowed), 'Duplicate visibility declarations')
    for name, row in vis_allowed.items():
        check(name in old and not protected(name), 'Cannot change visibility of missing or protected inherited geometry', name)
        check(bool(row.get('reason')), 'Visibility allowance must explain replacement', name)
    mat_rows = contract.get('allowed_material_rebindings', [])
    mat_allowed = {r['object']: r for r in mat_rows}
    check(len(mat_rows) == len(mat_allowed), 'Duplicate material rebinding declarations')
    geometry_changes, uv_changes, attribute_changes, modifier_changes, index_changes, material_changes, visibility_changes = ([] for _ in range(7))
    for name, before in old.items():
        after = now.get(name)
        if after is None:
            continue
        if any(after[k] != before[k] for k in ('geometry', 'matrix', 'counts')):
            geometry_changes.append(name)
        if after['uv'] != before['uv']:
            uv_changes.append(name)
        if after['attributes'] != before['attributes']:
            attribute_changes.append(name)
        if after['modifiers'] != before['modifiers']:
            modifier_changes.append(name)
        if after['material_index'] != before['material_index']:
            index_changes.append(name)
        if after['materials'] != before['materials'] or after['material_links'] != before['material_links']:
            material_changes.append(name)
            allow = mat_allowed.get(name)
            check(bool(allow and allow.get('reason')), 'Inherited material bindings changed without explicit allowance', name)
            if allow:
                check(allow.get('old_materials') == before['materials'] and allow.get('new_materials') == after['materials'],
                      'Material rebinding differs from exact declared old/new slots', name)
                if after['material_links'] != before['material_links'] or 'old_links' in allow or 'new_links' in allow:
                    check(allow.get('old_links') == before['material_links'] and allow.get('new_links') == after['material_links'],
                          'Material slot linkage differs from exact declared old/new links', name)
        if any(after[k] != before[k] for k in ('hide_render', 'hide_viewport')):
            visibility_changes.append(name)
            allow = vis_allowed.get(name)
            check(bool(allow and allow.get('reason')), 'Visibility changed without explicit allowance', name)
            if allow:
                for k in ('hide_render', 'hide_viewport'):
                    check(k in allow and after[k] == allow[k], 'Visibility differs from declared value', {'object': name, 'field': k})
    check(set(material_changes) == set(mat_allowed), 'Material declaration set does not equal actual changed inherited bindings', {'actual': material_changes, 'declared': sorted(mat_allowed)})
    check(set(visibility_changes) == set(vis_allowed), 'Visibility declaration set does not equal actual changes', {'actual': visibility_changes, 'declared': sorted(vis_allowed)})
    check(not geometry_changes, 'Inherited geometry, corner normals or world matrices changed', geometry_changes)
    check(not uv_changes, 'Inherited UV layer content or names changed', uv_changes)
    check(not attribute_changes, 'Inherited mesh attributes changed', attribute_changes)
    check(not modifier_changes, 'Inherited geometry modifiers changed', modifier_changes)
    check(not index_changes, 'Inherited polygon material indices changed', index_changes)
    for name, before in old_objects.items():
        obj = scene.objects.get(name)
        if not obj or before['type'] == 'MESH':
            continue
        after = dict(type=obj.type, matrix=np.asarray(obj.matrix_world).tolist(),
                     hide_render=obj.hide_render, hide_viewport=obj.hide_viewport)
        check(after == before, 'Inherited non-mesh transform/type/visibility changed', name)
    retirement_rows = contract.get('allowed_material_retirements', [])
    retirements = {row['material']: row for row in retirement_rows}
    check(len(retirements) == len(retirement_rows), 'Duplicate material retirement declarations')
    changed_materials, retired_materials, missing_materials = [], [], []
    for name, before in old_materials.items():
        material = bpy.data.materials.get(name)
        if material is None:
            missing_materials.append(name)
            allow = retirements.get(name)
            original_users = old_material_users[name]
            declared_users = sorted(allow.get('original_users', []), key=lambda row: (row['object'], row['slot_index'])) if allow else []
            allowed = bool(allow and allow.get('reason') and original_users
                and declared_users == original_users
                and all(u['object'] in mat_allowed and u['object'] in now
                    and u['slot_index'] < len(now[u['object']]['materials'])
                    and now[u['object']]['materials'][u['slot_index']] != name for u in original_users)
                and not any(slot.material and slot.material.name == name
                            for ob in bpy.data.objects for slot in ob.material_slots))
            check(allowed, 'Missing source material requires exact explicit retired-user proof',
                  dict(material=name, original_users=original_users, declared=allow))
            retired_materials.append(dict(material=name, original_users=original_users,
                source_graph_sha256=hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest(),
                all_users_rebound=allowed, reason=allow.get('reason') if allow else None))
        elif material_state(material) != before:
            changed_materials.append(name)
    check(set(retirements) == set(missing_materials), 'Material retirement declarations must exactly equal missing original materials',
          dict(actual=missing_materials, declared=sorted(retirements)))
    check(not changed_materials, 'Original material graphs/settings changed in-place; private copies required', changed_materials)
    changed_groups = []
    for name, before in old_node_groups.items():
        group = bpy.data.node_groups.get(name)
        if group is None or node_tree_state(group) != before:
            changed_groups.append(name)
    check(not changed_groups, 'Original shader node group changed in-place', changed_groups)
    report['inherited_materials'] = dict(count=len(old_materials), changed=changed_materials,
        retained_exact=len(old_materials) - len(missing_materials) - len(changed_materials),
        retired_unused=retired_materials, shader_group_count=len(old_node_groups), changed_shader_groups=changed_groups)
    images = []
    for im in bpy.data.images:
        if im.source in ('VIEWER', 'RENDER_RESULT'):
            continue
        state = image_state(im)
        if im.name in old_images:
            check(state == old_images[im.name], 'Inherited image payload/settings changed', im.name)
        resolved = bpy.path.abspath(im.filepath, library=im.library) if im.filepath else None
        available = bool(state['packed']) or im.source == 'GENERATED' or bool(resolved and Path(resolved).is_file())
        check(available and all(int(x) > 0 for x in im.size), 'Image missing packed payload/resolved file or invalid dimensions', im.name)
        images.append(dict(name=im.name, available=available, is_packed=bool(state['packed']),
                           resolved_path=None if state['packed'] else resolved, **state))
    check(not (set(old_images) - {i['name'] for i in images}), 'Inherited images removed', sorted(set(old_images) - {i['name'] for i in images}))
    report['image_dependencies'] = images
    for name, before in old_cameras.items():
        ob = scene.objects.get(name)
        check(ob is not None and ob.type == 'CAMERA', 'Original camera missing', name)
        if ob and ob.type == 'CAMERA':
            after = dict(matrix=np.asarray(ob.matrix_world).tolist(), data=rna_properties(ob.data))
            check(after == before, 'Original camera properties or transform changed', name)
    for name, before in old_collections.items():
        c = bpy.data.collections.get(name)
        check(c is not None and (c.hide_render, c.hide_viewport) == before, 'Original collection removed or hidden', name)
    new_layers = layer_visibility(scene)
    for name, before in old_layers.items():
        check(new_layers.get(name) == before, 'Original collection exclusion/holdout changed', name)
    report['inherited_geometry'] = dict(count=len(old), unchanged=len(old) - len(geometry_changes) - len(missing),
        missing=missing, changed=geometry_changes, changed_original_uv=uv_changes,
        changed_original_attributes=attribute_changes, changed_modifiers=modifier_changes,
        changed_material_indices=index_changes, material_assignment_changes=material_changes,
        permitted_visibility_changes=visibility_changes,
        protected_mesh_count=sum(protected(n) for n in old),
        mapped_building_wall_count=sum(mapped_building(n) for n in old),
        exact_v15_kerb_meshes=sum(n.startswith('V15_Kerb_') and n not in geometry_changes for n in old),
        exact_v15_driving_rubber='V15_Driving_Rubber' in old and 'V15_Driving_Rubber' not in geometry_changes)
    terrain_name = 'GROUND_Terrain_Continuous_v11'
    if terrain_name in old and terrain_name in now:
        b, a = old[terrain_name], now[terrain_name]
        check(b['undefined_normal_canonicalization'] == a['undefined_normal_canonicalization'], 'Known undefined terrain corner normal set changed')
        stored_old, stored_new = b['attributes'].get('custom_normal'), a['attributes'].get('custom_normal')
        check(bool(stored_old and stored_old['values']) and stored_old == stored_new, 'Stored terrain custom normals differ or could not be hashed exactly')
        report['undefined_terrain_normal_exception'] = dict(object=terrain_name,
            corners=b['undefined_normal_canonicalization'],
            source_raw_computed_sha256=b['raw_computed_normal_sha256'], candidate_raw_computed_sha256=a['raw_computed_normal_sha256'],
            stored_custom_normal_attribute_exact=stored_old == stored_new,
            policy='Only these three computed outputs canonicalized in validation memory after proving exact zero-area topology; files and stored custom normals remain unchanged.')
        report['warnings'].append('Three inherited exact-zero-area terrain faces have undefined computed corner normals in bpy 4.5.7. Their explicitly listed outputs are canonicalized only in validator memory; topology and raw custom_normal data are checked exactly.')

    road_parts = {n: footprint(scene.objects[n]) for n in PAVED if n in scene.objects}
    check(set(road_parts) == set(PAVED).intersection(old) and all(n in road_parts for n in PAVED[:2]), 'Expected inherited main/pit/side road surfaces missing', sorted(road_parts))
    paved = shapely.union_all(list(road_parts.values()))
    paved_interior = paved.buffer(-.002)
    map_path = Path(args.map_data).resolve() if args.map_data else source.parent / 'source_data' / 'surroundings_local.json'
    map_data = json.loads(map_path.read_text())
    building_parts = [shape(f['geometry']) for f in map_data['features'] if f['properties'].get('kind') == 'building']
    building_by_id = {f['properties']['osm_id']: shape(f['geometry']) for f in map_data['features']
                      if f['properties'].get('kind') == 'building'}
    check(bool(building_parts), 'No authoritative mapped building footprints', str(map_path))
    buildings = shapely.union_all(building_parts)
    buildings_interior = buildings.buffer(-.002)
    report['mapped_clearance_source'] = {'file': str(map_path), 'sha256': filehash(map_path), 'buildings': len(building_parts)}
    spatial, subset_results = [], []
    for name in new:
        row = declared.get(name)
        if not row:
            continue
        obj = scene.objects[name]
        vertices = world_vertices(obj)
        check(bool(np.isfinite(vertices).all()), 'Nonfinite new geometry', name)
        role = row.get('role')
        if role == 'inherited_subset':
            actual = subset_state(obj, range(len(obj.data.polygons)))
            expected = retained_proofs.get(name)
            same = actual == expected
            check(same, 'Retained subset differs from exact inherited selected polygon geometry/UV/shading/materials', name)
            subset_results.append(dict(object=name, source_object=row.get('source_object'), polygon_count=len(obj.data.polygons), exact=same,
                source_polygon_indices_sha256=ahash(np.asarray(row.get('source_polygon_indices', []), dtype=np.int32))))
            continue
        poly = footprint(obj)
        asphalt_overlap = float(poly.intersection(paved).area)
        building_overlap = float(poly.intersection(buildings).area)
        road_dist, building_dist = float(poly.distance(paved)), float(poly.distance(buildings))
        item = dict(object=name, role=role, vertices=len(vertices), triangles=len(obj.data.loop_triangles), footprint_m2=float(poly.area),
                    asphalt_overlap_m2=asphalt_overlap, building_overlap_m2=building_overlap,
                    asphalt_clearance_m=road_dist, building_clearance_m=building_dist)
        check(role in ('roadside', 'vegetation', 'yard_detail', 'attached_detail'), 'Unknown new geometry role, independent rule required', item)
        check(asphalt_overlap < 1e-4, 'New environment intersects paved road/pit/side street', item)
        check(not poly.intersects(paved_interior), 'New projected geometry enters paved interior, including vertical planes', item)
        check(road_dist + 0.002 >= row.get('minimum_asphalt_clearance_m', 0), 'New detail lacks declared asphalt clearance', item)
        if role == 'attached_detail':
            associated_name, osm_id = row.get('associated_building'), row.get('associated_osm_id')
            associated = scene.objects.get(associated_name or '')
            associated_map = building_by_id.get(osm_id)
            valid_association = bool(associated is not None and associated.type == 'MESH'
                and associated.name in old and mapped_building(associated.name)
                and associated_map is not None and associated_name == 'BLDG_' + str(osm_id).replace('/', '_') + '_walls')
            check(valid_association, 'Attached detail requires exact inherited associated building and mapped OSM id', item)
            required_limits = ('attachment_tolerance_m', 'maximum_above_building_m', 'minimum_below_building_m')
            valid_limits = all(isinstance(row.get(k), (int, float)) and np.isfinite(row[k]) and row[k] >= 0 for k in required_limits)
            check(valid_limits, 'Attached detail requires explicit finite nonnegative XY/Z limits', item)
            if valid_association and valid_limits:
                others = shapely.union_all([p for key, p in building_by_id.items() if key != osm_id])
                other_overlap = float(poly.intersection(others).area)
                tol = float(row['attachment_tolerance_m'])
                attachment_area = associated_map.buffer(tol + .002)
                outside = float(poly.difference(attachment_area).area)
                outside_vertices = int(np.count_nonzero(~shapely.covers(attachment_area, shapely.points(vertices[:, :2]))))
                bv = world_vertices(associated)
                zlo, zhi = float(bv[:, 2].min()), float(bv[:, 2].max())
                actual_zlo, actual_zhi = float(vertices[:, 2].min()), float(vertices[:, 2].max())
                item.update(associated_building=associated_name, associated_osm_id=osm_id,
                    attachment_tolerance_m=tol, other_building_overlap_m2=other_overlap,
                    outside_attachment_area_m2=outside, outside_attachment_vertices=outside_vertices,
                    associated_z_bounds_m=[zlo, zhi], actual_z_bounds_m=[actual_zlo, actual_zhi],
                    declared_z_allowances_m=[row['minimum_below_building_m'], row['maximum_above_building_m']])
                check(other_overlap < 1e-4, 'Attached detail intersects another mapped building', item)
                check(not poly.intersects(others.buffer(-.002)), 'Attached projected geometry enters another building interior', item)
                check(outside < 1e-4 and outside_vertices == 0, 'Attached detail exceeds declared building XY vicinity', item)
                check(actual_zlo >= zlo - row['minimum_below_building_m'] - .002
                      and actual_zhi <= zhi + row['maximum_above_building_m'] + .002,
                      'Attached detail exceeds declared associated building Z bounds', item)
        else:
            check(building_overlap < 1e-4, 'New environment intersects mapped building footprint', item)
            check(not poly.intersects(buildings_interior), 'New projected geometry enters mapped building interior', item)
            check(building_dist + 0.002 >= row.get('minimum_building_clearance_m', 0), 'New detail lacks declared building clearance', item)
        if row.get('footprint'):
            tol = float(row.get('footprint_tolerance_m', 0.002))
            outside = float(poly.difference(shape(row['footprint']).buffer(tol)).area)
            item['outside_reported_footprint_m2'] = outside
            check(outside < 1e-4, 'Actual geometry outside declared footprint', item)
        spatial.append(item)
        if len(spatial) % 20 == 0:
            print('V17_VALIDATOR new spatial', len(spatial), flush=True)
    report['new_geometry'] = spatial
    report['retained_source_subsets'] = subset_results
    if subset_results:
        report['warnings'].append('Retained source subsets inherit existing source spatial conditions. Their exact selected polygons are proven preserved; their old clearances are not recertified as new content.')

    bvh_cache, ground_results = {}, []
    for row in contract.get('ground_checks', []):
        name = row['object']
        obj = scene.objects.get(name)
        ids = row.get('vertex_indices', [])
        names = row.get('surfaces', [])
        valid = obj is not None and obj.type == 'MESH' and bool(ids) and bool(names)
        if valid:
            valid = all(isinstance(i, int) and 0 <= i < len(obj.data.vertices) for i in ids)
            valid = valid and all(n in scene.objects and scene.objects[n].type == 'MESH' for n in names)
        check(valid, 'Ground check requires valid exact mesh vertices and surface names', name)
        if not valid:
            continue
        for n in names:
            if n not in bvh_cache:
                bvh_cache[n] = terrain_bvh(scene.objects[n])
        verts = world_vertices(obj)
        offsets, missing_hits, invalid_offsets = [], [], []
        lo, hi = float(row['minimum_offset_m']), float(row['maximum_offset_m'])
        check(lo <= hi, 'Ground offset bounds reversed', name)
        for i in ids:
            p = verts[i]
            hits = [bvh_cache[n].ray_cast(Vector((float(p[0]), float(p[1]), 10000.0)), Vector((0, 0, -1)), 20000.0)[0] for n in names]
            z = [v.z for v in hits if v is not None]
            if not z:
                missing_hits.append(i)
                continue
            offset = float(p[2] - max(z))
            offsets.append(offset)
            if not lo - 0.002 <= offset <= hi + 0.002:
                invalid_offsets.append({'vertex': i, 'offset_m': offset})
        result = dict(object=name, samples=len(ids), surfaces=names, min_offset_m=min(offsets) if offsets else None,
                      max_offset_m=max(offsets) if offsets else None, permitted_range_m=[lo, hi],
                      missing_hits=missing_hits, invalid_offsets=invalid_offsets)
        check(not missing_hits and not invalid_offsets, 'Declared support vertices fail independent terrain anchoring', result)
        ground_results.append(result)
    report['ground_support_checks'] = ground_results
    report['warnings'].append('Ground checks cover declared actual support vertices only; they do not establish continuous contact of every triangle with terrain. Planar footprint overlap is conservative, including transparent leaf-card geometry.')

    material_results = []
    for row in contract.get('material_checks', []):
        name = row['material']
        material = bpy.data.materials.get(name)
        check(material is not None and material.use_nodes, 'Expected material missing or has no nodes', name)
        if not material or not material.use_nodes:
            continue
        principals = [n for n in material.node_tree.nodes if n.type == 'BSDF_PRINCIPLED']
        check(len(principals) == 1, 'Expected one Principled surface for material graph checks', name)
        if len(principals) != 1:
            continue
        channels = {}
        for channel in row.get('required_variable_inputs', []):
            socket = principals[0].inputs.get(channel)
            evidence = variable_sources(socket) if socket else []
            check(bool(evidence), 'No variable input source for declared material channel', dict(material=name, input=channel))
            channels[channel] = evidence
        material_results.append(dict(material=name, inputs=channels))
    report['material_graph_checks'] = material_results
    report['warnings'].append('Material graph checks establish linked procedural/bitmap inputs, not perceptual success or AC shader equivalence. No KN5, AC physics, AI or performance claim is made.')
    report['candidate_sha256'] = filehash(candidate)
    report['source_sha256_after'] = filehash(source)
    check(report['source_sha256_after'] == source_before, 'Immutable source modified during validation')
    check(report['candidate_sha256'] == candidate_before, 'Candidate modified concurrently during read-only validation')
    report['passed'] = not report['errors']
    report['seconds'] = round(time.time() - start, 3)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'errors': len(report['errors']), 'seconds': report['seconds'], 'output': args.output}), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--change-report', required=True)
    parser.add_argument('--map-data', help='Original surroundings GeoJSON, in common scene coordinates')
    parser.add_argument('--output', default=str(ROOT / 'validation_v17.json'))
    raise SystemExit(validate(parser.parse_args()))
