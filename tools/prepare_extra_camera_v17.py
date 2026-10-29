"""Read-only additional driver camera, grounded on the actual native road."""
from pathlib import Path
import json, hashlib
import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT=Path(__file__).resolve().parent.parent
SOURCE=ROOT.parent/'chmyrivka_v16/chmyrivka_visual_v16.blend'
SHA='8e4f2c9c3e1b37de42d9d99edf115a898fa72cb76936dce19020bb02f23b0e00'
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
bpy.ops.wm.open_mainfile(filepath=str(SOURCE))
road=bpy.context.scene.objects['1ROAD_MAIN']
verts=[road.matrix_world@v.co for v in road.data.vertices]
bvh=BVHTree.FromPolygons(verts,[list(p.vertices) for p in road.data.polygons])
rows=json.loads((ROOT/'source_data/track_plan_v08.json').read_text())['samples']
s=np.array([r['s'] for r in rows])
def point(station):
    x,y=[float(np.interp(station,s,[r[k] for r in rows])) for k in ['x','y']]
    hit=bvh.ray_cast(Vector((x,y,1000)),Vector((0,0,-1)),2000)[0]
    assert hit is not None
    return Vector((x,y,hit.z+.8))
pos=point(3985);look=point(4040)
rotation=(look-pos).to_track_quat('-Z','Y').to_euler()
r={'camera':'V17_DRIVER_3985','title':'Сінна · водительский вид между дворами',
   'kind':'driver','station_map_m':3985,'location':list(pos),'rotation':list(rotation),
   'lens_mm':29.0,'eye_above_native_road_m':.8,'look_at':list(look),
   'reference_native_sha256':SHA,'scope':'Blender review camera, not a game screenshot'}
d=json.loads((ROOT/'data/extra_cameras_v15.json').read_text())
d['cameras'].append(r);d['purpose']='Four retained close/context cameras plus one road-grounded driver view.'
d.pop('native_sha256',None)
(ROOT/'data/extra_cameras_v17.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==SHA
print(json.dumps(r,ensure_ascii=False))
