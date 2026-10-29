"""Generate candidate camera-visible shrub placements; reads native, never saves it."""
import sys,json,math,random,hashlib
from pathlib import Path
import bpy,numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from bpy_extras.object_utils import world_to_camera_view
from shapely.geometry import Point,Polygon,LineString,MultiPoint,shape,mapping
from shapely.prepared import prep
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'chmyrivka_visual_v17.blend';bpy.ops.wm.open_mainfile(filepath=str(source));scene=bpy.context.scene
camera_key=sys.argv[-1] if len(sys.argv)>1 else 'V11_19_DRIVER'
if camera_key=='V17_DRIVER_3985':
 spec=json.loads((ROOT/'data/extra_cameras_v17.json').read_text())['cameras'][-1];data=bpy.data.cameras.new(camera_key);camera=bpy.data.objects.new(camera_key,data);scene.collection.objects.link(camera);camera.location=spec['location'];camera.rotation_euler=spec['rotation'];camera.data.lens=spec['lens_mm'];camera.data.sensor_width=36;bpy.context.view_layer.update()
else:camera=scene.objects['V11_19_DRIVER']
loc=camera.matrix_world.translation.copy()
front=json.loads((ROOT/'data/frontage_v17.json').read_text());layout=json.loads((ROOT/'source_data/yard_layout_v12.json').read_text());safegeom=shape(front['new_planting_region']).difference(shape(front['reservation']));safe=prep(safegeom)
sys.path.insert(0,str(ROOT/'dependencies/v15'));from verges_v15 import Ground
ground=Ground(scene,Point(loc.x,loc.y).buffer(110))
# Face-accurate opaque structural blockers only. New planting is deliberately
# excluded: this test decides whether a candidate can be read above solids.
verts=[];faces=[];faceobjects=[];objects=[]
for ob in scene.objects:
 if ob.type!='MESH' or ob.hide_render:continue
 name=ob.name.lower()
 structural=ob.get('v12_role') in ('architecture','yards','yard_detail') or 'building' in name or 'barrier' in name or 'house' in name or ob.get('v17_role')=='architecture' or ob.name=='V16_Yard_Fences_Gates'
 if not structural:continue
 box=MultiPoint([tuple(ob.matrix_world@Vector(v))[:2] for v in ob.bound_box]).convex_hull
 if not box.intersects(Point(loc.x,loc.y).buffer(110)):continue
 # Skip newage plant batches accidentally using a house prefix.
 if any(q in name for q in ['grass','plant','leaf','flower','bush','tree']):continue
 ob.data.calc_loop_triangles();world=[ob.matrix_world@v.co for v in ob.data.vertices];count=0
 for t in ob.data.loop_triangles:
  vs=[world[i] for i in t.vertices]
  if min((v-loc).length for v in vs)>130:continue
  off=len(verts);verts.extend(vs);faces.append((off,off+1,off+2));faceobjects.append(ob.name);count+=1
 if count:objects.append({'name':ob.name,'triangles':count})
bvh=BVHTree.FromPolygons(verts,faces,all_triangles=True)
def sight(p):
 d=p-loc;hit,normal,index,dist=bvh.ray_cast(loc,d.normalized(),d.length-.025)
 return {'visible':hit is None,'blocker':faceobjects[index] if index is not None else None,'distance_m':float(d.length),'point_xyz':list(p)}
rng=random.Random(17064842);candidates=[]
for y in layout['yards']:
 frontedge=LineString(y['front_edge']);frontzone=shape(y['polygon']).intersection(safegeom).intersection(frontedge.buffer(4.8,cap_style=2));free=frontzone.buffer(-.78)
 if free.is_empty:continue
 minx,miny,maxx,maxy=free.bounds
 for i in range(350):
  x=rng.uniform(minx,maxx);yy=rng.uniform(miny,maxy);p=Point(x,yy)
  if not free.covers(p):continue
  h=ground.hit(x,yy)
  if h is None:continue
  d=(h-loc).length
  if not 12<d<90:continue
  target=h+Vector((0,0,1.94));uv=world_to_camera_view(scene,camera,target)
  if uv.z<=0 or not .025<uv.x<.975 or not .20<uv.y<.70:continue
  # Top-lobe support points, with horizontal offsets along screen-right.
  screenright=camera.matrix_world.to_quaternion()@Vector((1,0,0))
  checks=[sight(h+Vector((0,0,z))+screenright*xoff) for z,xoff in [(1.52,0),(1.85,-.35),(1.85,.35),(2.08,0),(2.12,-.18),(2.12,.18)]]
  visible=sum(q['visible'] for q in checks)
  if visible<4:continue
  candidates.append({'osm_id':y['osm_id'],'station_map_m':y['station_map_m'],'root_xyz':list(h),'radius_m':.65,'design_height_m':2.3,'distance_m':d,'screen_xy':[uv.x,uv.y],'visible_samples':visible,'samples':checks,'front_edge_distance_m':frontedge.distance(p)})
# rank one candidate per parcel, prefer closer/front edge with projection spread
byyard={}
for p in candidates:
 score=p['visible_samples']*3-p['distance_m']*.075-p['front_edge_distance_m']*.7
 if p['osm_id'] not in byyard or score>byyard[p['osm_id']][0]:byyard[p['osm_id']]=(score,p)
ranked=sorted([p for score,p in byyard.values()],key=lambda p:p['distance_m'])
report={'camera':camera.name,'camera_location':list(loc),'camera_rotation':list(camera.rotation_euler),'camera_lens_mm':camera.data.lens,'scene_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'purpose':'Structural line-of-sight ranking for selected tall frontage shrub crowns. Does not prove final leaf silhouette coverage or pixel quality.','structural_meshes':objects,'candidate_count':len(candidates),'ranked_parcels':ranked}
(ROOT/('data/frontage_visibility_candidates_v17'+('_3985' if camera_key=='V17_DRIVER_3985' else '')+'.json')).write_text(json.dumps(report,indent=2))
for p in ranked:print(p['osm_id'],'root',list(round(q,3) for q in p['root_xyz']),'dist',round(p['distance_m'],1),'uv',[round(q,3) for q in p['screen_xy']],'visible',p['visible_samples'],'front',round(p['front_edge_distance_m'],2),flush=True)
