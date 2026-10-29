"""V16 native foliage: local private tint, leaf infill and grounded batched cover.

Uses immutable v15 Ground/_exclusions helpers and native authored V10 leaf atlas.
No mapped geometry changes. Only the local V12 grass batch is hidden, retained.
All new plants are regional inferred dressing, not a species/position survey.
"""
from __future__ import annotations
from pathlib import Path
import sys,json,math,random
import bpy,numpy as np
from mathutils import Vector
from shapely.geometry import Point,Polygon,LineString,MultiPoint,shape,mapping
from shapely.ops import unary_union
from shapely.prepared import prep
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'chmyrivka_v15'))
sys.path.insert(0,str(ROOT/'dependencies/v15'))
from verges_v15 import Ground,_exclusions,_geom
SEED=160641
UP=Vector((0,0,1))
MAX_TRIANGLES=550000


def _load(source,name):
    obj=source.get(name) or source.get(name.removesuffix('.json'))
    if isinstance(obj,dict):return obj
    return json.loads((Path(source.get('source_data_dir',ROOT.parent/'chmyrivka_detail_v13/data'))/name).read_text())

class Batch:
    def __init__(self,name,materials,kind):
        self.name=name;self.materials=materials;self.kind=kind
        self.v=[];self.f=[];self.mi=[];self.uv=[];self.tint=[];self.support=[]
    def add(self,vertices,faces,mi=0,uv=None,tint=(1,1,1,1),support=()):
        start=len(self.v);self.v.extend(tuple(v) for v in vertices)
        self.f.extend(tuple(start+i for i in f) for f in faces);self.mi.extend([mi]*len(faces))
        self.uv.extend(uv or [(0,0)]*len(vertices));self.tint.extend([tint]*len(vertices))
        self.support.extend(start+i for i in support)
    def finish(self,collection,ground_names):
        if not self.f:return None
        mesh=bpy.data.meshes.new(self.name+'_mesh');mesh.from_pydata(self.v,[],self.f)
        for m in self.materials:mesh.materials.append(m)
        for p,m in zip(mesh.polygons,self.mi):p.material_index=m;p.use_smooth=True
        uv=mesh.uv_layers.new(name='UVMap')
        uv.data.foreach_set('uv',np.asarray([self.uv[l.vertex_index] for l in mesh.loops],dtype=np.float32).ravel())
        vc=mesh.color_attributes.new(name='V16_FoliageTint',type='FLOAT_COLOR',domain='POINT')
        vc.data.foreach_set('color',np.asarray(self.tint,dtype=np.float32).ravel())
        mesh.update();mesh.calc_loop_triangles()
        ob=bpy.data.objects.new(self.name,mesh);collection.objects.link(ob)
        ob['v16_role']='vegetation';ob['vegetation_kind']=self.kind
        ob['v16_support_vertex_indices']=json.dumps(self.support,separators=(',',':'))
        ob['v16_reference_surface_names']=json.dumps(ground_names,separators=(',',':'))
        ob['v16_support_offset_min_m']=-.007;ob['v16_support_offset_max_m']=.007
        ob['placement_evidence']='Regional inferred planting, no individual plant survey; inherited ground and mapped objects preserved.'
        ob['export_lod_class']='near_detail';ob['v16_new_visual_only']=True
        return ob


def _private_leaf(base,variant,use_vertex=False):
    name='V16_Vegetation_'+base.name+'_'+str(variant)+('_VC' if use_vertex else '')
    if bpy.data.materials.get(name):return bpy.data.materials[name]
    mat=base.copy();mat.name=name
    n,l=mat.node_tree.nodes,mat.node_tree.links
    tx=next((q for q in n if q.type=='TEX_IMAGE' and q.image and 'diffuse' in q.image.name),None)
    if tx:
        dst=[q.to_socket for q in list(tx.outputs['Color'].links)]
        hs=n.new('ShaderNodeHueSaturation');hs.name='V16_Leaf_Palette'
        hs.inputs['Saturation'].default_value=(.73,.82,.9)[variant%3]
        hs.inputs['Value'].default_value=(1.17,1.06,.94)[variant%3]
        hs.inputs['Hue'].default_value=(.496,.501,.488)[variant%3]
        l.new(tx.outputs['Color'],hs.inputs['Color']);out=hs.outputs['Color']
        if use_vertex:
            vc=n.new('ShaderNodeVertexColor');vc.layer_name='V16_FoliageTint'
            mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1
            l.new(out,mix.inputs[1]);l.new(vc.outputs['Color'],mix.inputs[2]);out=mix.outputs[0]
        for d in dst:l.new(out,d)
    for p in n:
        if p.type=='BSDF_PRINCIPLED':
            p.inputs['Roughness'].default_value=.69;p.inputs['Specular IOR Level'].default_value=.23
        elif p.type=='NORMAL_MAP':p.inputs['Strength'].default_value=.23
    mat['v16_copy_of']=base.name;mat['v16_native_atlas_unchanged']=True
    return mat


def _grass_material(i):
    name=f'V16_Vegetation_Fine_Grass_{i}'
    m=bpy.data.materials.new(name);m.use_nodes=True;n,l=m.node_tree.nodes,m.node_tree.links
    p=n.get('Principled BSDF');out=n.get('Material Output')
    colors=[(.080,.112,.028,1),(.119,.145,.047,1),(.177,.183,.075,1),(.251,.225,.125,1)]
    p.inputs['Roughness'].default_value=.81;p.inputs['Specular IOR Level'].default_value=.18
    vc=n.new('ShaderNodeVertexColor');vc.layer_name='V16_FoliageTint'
    mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1;mix.inputs[1].default_value=colors[i]
    l.new(vc.outputs['Color'],mix.inputs[2]);l.new(mix.outputs[0],p.inputs['Base Color'])
    trans=n.new('ShaderNodeBsdfTranslucent');l.new(mix.outputs[0],trans.inputs['Color'])
    blend=n.new('ShaderNodeMixShader');blend.inputs[0].default_value=.34;l.new(p.outputs[0],blend.inputs[1]);l.new(trans.outputs[0],blend.inputs[2]);l.new(blend.outputs[0],out.inputs[0])
    m.diffuse_color=colors[i];m['source']='Authored narrow curved leaf geometry, no bitmap';return m


def apply(scene,collection,source_data=None):
    if scene.get('v16_vegetation_applied'):raise RuntimeError('Vegetation v16 already applied')
    source=dict(source_data or {});source.setdefault('source_data_dir',ROOT.parent/'chmyrivka_detail_v13/data');rng=random.Random(SEED)
    track=source.get('track_v08') or _load(source,'track_plan_v08.json');layout=_load(source,'yard_layout_v12.json')
    samples=track['samples'];ss=np.array([q['s'] for q in samples]);xy=np.array([(q['x'],q['y']) for q in samples]);full_axis=LineString(xy)
    cum=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    def station(x,y):return float(np.interp(full_axis.project(Point(x,y)),cum,ss))
    sectors=[]
    for key,s0,s1,half in [('sinna',3800,4250,52),('north',2260,2600,58)]:
        points=[(float(np.interp(s0,ss,xy[:,0])),float(np.interp(s0,ss,xy[:,1])))]
        points.extend(tuple(p) for s,p in zip(ss,xy) if s0<s<s1)
        points.append((float(np.interp(s1,ss,xy[:,0])),float(np.interp(s1,ss,xy[:,1]))))
        axis=LineString(points);sectors.append({'id':key,'range':[s0,s1],'half':half,'axis':axis,'region':axis.buffer(half,cap_style=2)})
    region=unary_union([q['region'] for q in sectors]);obstacle,roads,houses,paths,native_paved=_exclusions(scene,region,source,track,layout)
    # Explicit existing V15 kerb footprints + all protective barrier footprints.
    reserve=[]
    for ob in scene.objects:
        if ob.type!='MESH':continue
        if (ob.name.startswith('V15_Kerb_') or ob.get('v15_role')=='racing_kerb' or
            ('Barrier' in ob.name and not ob.hide_render)):
            pts=[tuple(ob.matrix_world@Vector(q))[:2] for q in ob.bound_box]
            hull=MultiPoint(pts).convex_hull
            if hull.intersects(region):
                # Whole batched barrier boxes can span kilometres. Project each face.
                faces=[];ww=[ob.matrix_world@v.co for v in ob.data.vertices]
                for f in ob.data.polygons:
                    p=Polygon([tuple(ww[i])[:2] for i in f.vertices])
                    if p.is_valid and p.area>1e-7 and p.intersects(region):faces.append(p)
                if faces:reserve.append(unary_union(faces).buffer(.20))
    yard_report=source.get('yards_v16')
    yard_path=Path(source.get('output_data_dir',ROOT/'data'))/'yards_v16.json'
    if not yard_report and yard_path.exists():yard_report=json.loads(yard_path.read_text())
    if yard_report:
        for item in yard_report.get('placements',[]):
            if item.get('footprint'):reserve.append(_geom(item['footprint']).buffer(.08))
        val=yard_report.get('reserved_detail_footprints')
        if val:
            if isinstance(val,dict):reserve.append(_geom(val).buffer(.08))
            else:reserve.extend(_geom(q).buffer(.08) for q in val)
    obstacle=unary_union([obstacle,*reserve]);safe_geom=region.difference(obstacle);safe=prep(safe_geom)
    ground=Ground(scene,region.buffer(2));ground_names=ground.names
    base=bpy.data.materials.get('V10_Game_Foliage_singleleaf')
    if not base:raise ValueError('Missing native single-leaf atlas material')
    leafm=[_private_leaf(base,i,True) for i in range(3)]
    bark=bpy.data.materials['V10_Game_Bark_Dark']
    grassm=[_grass_material(i) for i in range(4)]
    infill=Batch('V16_Vegetation_Crown_Infill',leafm,'crown_infill')
    shrubs=Batch('V16_Vegetation_Understorey',[*leafm,bark],'shrub')
    grass=Batch('V16_Vegetation_Fine_Groundcover',grassm,'fine_grass')
    garden_trees=Batch('V16_Vegetation_Garden_Canopies',[*leafm,bark],'garden_tree')
    hidden=[];rebindings=[];records=[];infill_records=[];rejected={};counts={'grass_tufts':0,'grass_blades':0,'shrubs':0,'crown_trees':0,'crown_leaf_cards':0,'shrub_leaf_cards':0,'new_garden_trees':0,'garden_tree_leaf_cards':0}
    # Old opaque folded V12 grass exists only in the Sinna detail sector.
    old=scene.objects.get('V12_Garden_ground_contact_grass')
    if old and not old.hide_render:
        hidden.append({'name':old.name,'before_hide_render':old.hide_render,'before_hide_viewport':old.hide_viewport,'after_hide_render':True,'after_hide_viewport':True,'reason':'Replaced by narrower curved, thin-leaf V16 ground cover within original Sinna sector.'})
        old.hide_render=True;old.hide_viewport=True
    # Preserve every shared mesh and all trees outside the two selected sectors.
    locals=[]
    for ob in sorted(scene.objects,key=lambda q:q.name):
        if ob.type!='MESH' or ob.hide_render:continue
        if not (ob.get('vegetation_v10') or ob.get('v12_role')=='vegetation'):continue
        x,y,z=ob.matrix_world.translation
        if not region.covers(Point(x,y)):continue
        if ob.get('vegetation_kind')=='batched_grass':continue
        leaf_slots=[]
        for i,slot in enumerate(ob.material_slots):
            mat=slot.material
            if not mat or not ('Foliage_' in mat.name or 'Garden_V10_' in mat.name):continue
            if 'pine' in mat.name:continue
            replacement=_private_leaf(mat,len(rebindings)%3)
            before_link=slot.link;slot.link='OBJECT';slot.material=replacement
            rebindings.append({'object':ob.name,'slot':i,'old_material':mat.name,'new_material':replacement.name,'old_link':before_link,'new_link':'OBJECT'})
            leaf_slots.append(i)
        if leaf_slots and ('Tree_' in ob.data.name or ob.get('vegetation_kind')=='tree'):
            locals.append((ob,leaf_slots))

    def card(batch,c,u,v,mi,tile,shade=1):
        points=[c-u-v,c+u-v,c+u+v,c-u+v]
        hull=Polygon([tuple(p)[:2] for p in points]).convex_hull
        if not safe.covers(hull):return False
        ox=(tile%2)*.5;oy=(tile//2)*.5;pad=.003
        uv=[(ox+pad,oy+pad),(ox+.5-pad,oy+pad),(ox+.5-pad,oy+.5-pad),(ox+pad,oy+.5-pad)]
        batch.add(points,[(0,1,2,3)],mi,uv,(shade,shade,shade,1));return True

    # Dense local crown groups follow inherited leaf centres, preserving root,
    # branch skeleton, original envelope and existing mapped plant silhouette.
    cameras=[Vector((1485.52,1142.37,4.93)),Vector((1430.11,1017.69,5.86)),Vector((774.18,2084.94,5.24))]
    selected=[]
    for sect in sectors:
        candidates=[q for q in locals if sect['region'].covers(Point(q[0].matrix_world.translation[:2]))]
        candidates.sort(key=lambda q:min((q[0].matrix_world.translation-c).length for c in cameras))
        selected.extend(candidates[:14])
    for ob,leaf_slots in selected:
        mesh=ob.data;world=[ob.matrix_world@v.co for v in mesh.vertices]
        bounds=np.array([tuple(p) for p in world]);lo=bounds.min(axis=0);hi=bounds.max(axis=0)
        source_faces=[f for f in mesh.polygons if f.material_index in leaf_slots and len(f.vertices)==4]
        if not source_faces:continue
        rng.shuffle(source_faces);start=len(infill.v);cards=0
        target=1300 if min((ob.matrix_world.translation-c).length for c in cameras)<75 else 700
        # At most ~3% crown span displacement: it fills gaps without making
        # oversized cardboard leaves or increasing original tree bounds.
        for j in range(target):
            f=source_faces[j%len(source_faces)];c=sum((world[i] for i in f.vertices),Vector())/len(f.vertices)
            c+=Vector((rng.gauss(0,.11),rng.gauss(0,.11),rng.gauss(0,.09)))
            a=rng.uniform(0,math.tau);tilt=rng.uniform(-.8,.8)
            length=rng.uniform(.10,.18);width=rng.uniform(.045,.085)
            u=Vector((math.cos(a),math.sin(a),0))*width/2
            v=Vector((-math.sin(a)*math.sin(tilt),math.cos(a)*math.sin(tilt),math.cos(tilt)))*length/2
            pts=np.array([tuple(c-u-v),tuple(c+u-v),tuple(c+u+v),tuple(c-u+v)])
            if (pts<lo).any() or (pts>hi).any():continue
            if card(infill,c,u,v,j%3,j%4,rng.uniform(.86,1.12)):cards+=1
        if cards:
            infill_records.append({'source_tree':ob.name,'source_root_xyz':list(ob.matrix_world.translation),'source_bounds_xyz':[lo.tolist(),hi.tolist()],'start_vertex':start,'end_vertex_exclusive':len(infill.v),'leaf_cards':cards,'station_map_m':station(ob.location.x,ob.location.y),'ground_support':'Inherited tree root, infill is attached crown geometry; no new trunk/root.'})
            counts['crown_trees']+=1;counts['crown_leaf_cards']+=cards

    def accepted(kind,root,vertices,start,end,extra=None):
        hull=MultiPoint([tuple(p)[:2] for p in vertices]).convex_hull
        if not safe.covers(hull):raise AssertionError(f'v16 unsafe {kind} footprint')
        r={'kind':kind,'root_xyz':list(root),'start_vertex':start,'end_vertex_exclusive':end,'footprint':mapping(hull),'station_map_m':station(root.x,root.y),'min_road_clearance_m':hull.distance(roads),'min_building_clearance_m':hull.distance(houses),'min_paved_clearance_m':hull.distance(paths)}
        if extra:r.update(extra)
        records.append(r)

    def tuft(x,y,yard=False):
        if counts['grass_tufts']>=3100 or not safe.covers(Point(x,y).buffer(.20)):return False
        h=ground.hit(x,y)
        if h is None:return False
        start=len(grass.v);allv=[]
        for j in range(rng.randint(6,9)):
            a=rng.uniform(0,math.tau);r=rng.uniform(0,.065)
            b=ground.hit(x+math.cos(a)*r,y+math.sin(a)*r)
            if b is None:continue
            height=rng.uniform(.045,.14 if yard else .19);width=rng.uniform(.0035,.008)
            side=Vector((math.cos(a+.7),math.sin(a+.7),0))*width/2
            lean=Vector((math.cos(a),math.sin(a),0))*rng.uniform(.02,.07)
            left=ground.hit(b.x-side.x,b.y-side.y);right=ground.hit(b.x+side.x,b.y+side.y)
            if left is None or right is None:continue
            left.z-=.001;right.z-=.001
            mid=b+UP*height*.53+lean*.30;tip=b+UP*height+lean
            vv=[left,right,mid+side*.56,mid-side*.56,tip]
            grass.add(vv,[(0,1,2,3),(3,2,4)],rng.choices(range(4),[.31,.39,.24,.06])[0],tint=(rng.uniform(.8,1.15),)*3+(1,),support=(0,1))
            allv+=vv;counts['grass_blades']+=1
        if allv:
            accepted('grass',h,allv,start,len(grass.v),{'batch':'V16_Vegetation_Fine_Groundcover','max_height_m':max(v.z-h.z for v in allv)})
            counts['grass_tufts']+=1;return True
        return False

    shrub_roots=[]
    def shrub(x,y,height=.85,house=None):
        radius=height*rng.uniform(.50,.67)
        if counts['shrubs']>=88 or not safe.covers(Point(x,y).buffer(radius+.15)):return False
        if any((x-a)**2+(y-b)**2<(.65*(radius+rr))**2 for a,b,rr in shrub_roots):return False
        h=ground.hit(x,y)
        if h is None:return False
        start=len(shrubs.v);allv=[]
        # Short branched stems terminate inside, not beneath, the leaf envelope.
        for k in range(5):
            a=k*2.399963+rng.uniform(-.2,.2);b=h+Vector((math.cos(a)*.025,math.sin(a)*.025,0));b.z-=.002
            top=h+Vector((math.cos(a)*radius*.45,math.sin(a)*radius*.45,height*.65))
            vec=top-b;u=vec.cross(UP).normalized()*.009;v=vec.normalized().cross(u)
            vv=[b+u,b+v,b-u,b-v,top+u*.25,top+v*.25,top-u*.25,top-v*.25]
            for j in range(4):
                hit=ground.hit(vv[j].x,vv[j].y)
                if hit:vv[j].z=hit.z-.002
            shrubs.add(vv,[(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)],3,support=range(4));allv+=vv
        # Low full crowns: 7 overlapping lobes, horizontal and tilted leaves.
        cards=0
        for j in range(430):
            lobe=j%7;az=lobe*2.399963
            lc=h+Vector((math.cos(az)*radius*.39,math.sin(az)*radius*.39,height*(.47+.16*math.sin(lobe*1.2))))
            z=rng.uniform(-1,1);a=rng.uniform(0,math.tau);rr=rng.random()**.25
            c=lc+Vector((math.sqrt(1-z*z)*math.cos(a)*radius*.52*rr,math.sqrt(1-z*z)*math.sin(a)*radius*.52*rr,z*height*.31*rr))
            length=rng.uniform(.11,.20);width=rng.uniform(.045,.085)
            azimuth=rng.uniform(0,math.tau);tilt=rng.uniform(-1.3,1.3)
            u=Vector((math.cos(azimuth),math.sin(azimuth),0))*width/2
            v=Vector((-math.sin(azimuth)*math.sin(tilt),math.cos(azimuth)*math.sin(tilt),math.cos(tilt)))*length/2
            if card(shrubs,c,u,v,j%3,j%4,rng.uniform(.77,1.16)):
                allv.extend([c-u-v,c+u-v,c+u+v,c-u+v]);cards+=1
        accepted('shrub',h,allv,start,len(shrubs.v),{'batch':'V16_Vegetation_Understorey','height_m':height,'mapped_house_reference':house})
        counts['shrubs']+=1;counts['shrub_leaf_cards']+=cards;shrub_roots.append((x,y,radius));return True

    # Shallow patches along race verges: transition from low grass near paving
    # to longer irregular clumps behind the edge. Significant gaps remain.
    for sector in sectors:
        axis=sector['axis'];candidates=[]
        for d in np.arange(1,axis.length-1,1.45):
            if rng.random()<.24:continue
            p=axis.interpolate(float(d));a=axis.interpolate(max(0,float(d)-.5));b=axis.interpolate(min(axis.length,float(d)+.5))
            angle=math.atan2(b.y-a.y,b.x-a.x)
            for sign in (-1,1):
                for distance in np.arange(7.1,15.2,.4):
                    x=p.x-math.sin(angle)*distance*sign;y=p.y+math.cos(angle)*distance*sign
                    if safe.covers(Point(x,y).buffer(.35)):
                        candidates.append((x,y,angle));break
        rng.shuffle(candidates)
        limit=850 if sector['id']=='north' else 1000;before=counts['grass_tufts']
        for x,y,a in candidates:
            for j in range(rng.randint(3,6)):
                u=rng.gauss(0,.6);v=rng.gauss(0,.25)
                tuft(x+math.cos(a)*u-math.sin(a)*v,y+math.sin(a)*u+math.cos(a)*v)
            if counts['grass_tufts']-before>=limit:break
        # Northern shoulder understorey in irregular short groups only beyond
        # 3m from pavement, preserving driver sight line through the corner.
        if sector['id']=='north':
            for d in np.arange(18,axis.length-8,9.5):
                if rng.random()<.24:continue
                p=axis.interpolate(float(d));a=axis.interpolate(float(d)-.5);b=axis.interpolate(float(d)+.5);angle=math.atan2(b.y-a.y,b.x-a.x)
                sign=rng.choice((-1,1));offset=rng.uniform(11.5,20)
                x=p.x-math.sin(angle)*offset*sign;y=p.y+math.cos(angle)*offset*sign
                if roads.distance(Point(x,y))<3.0:continue
                for j in range(rng.randint(1,3)):
                    shrub(x+math.cos(angle)*j*.85,y+math.sin(angle)*j*.85,rng.uniform(.65,1.25))

    # Low garden edge groups and irregular grass patches, not carpetlike lawns.
    for yard in sorted(layout['yards'],key=lambda q:(q['station_map_m'],str(q['osm_id']))):
        if not 3800<yard['station_map_m']<4250:continue
        poly=_geom(yard['polygon']);ident=yard['osm_id'];usable=poly.buffer(-.38).intersection(safe_geom)
        if usable.is_empty:continue
        minx,miny,maxx,maxy=usable.bounds
        # 30 accepted low tufts per garden provides ground contact between old
        # detail groups; deterministic clustering creates mowing variation.
        centers=[]
        for _ in range(22):
            x=rng.uniform(minx,maxx);y=rng.uniform(miny,maxy)
            if usable.covers(Point(x,y).buffer(.27)):centers.append((x,y))
            if len(centers)>=3:break
        for x,y in centers:
            for _ in range(11):
                xx=x+rng.gauss(0,.55);yy=y+rng.gauss(0,.40)
                if poly.covers(Point(xx,yy).buffer(.20)):tuft(xx,yy,True)
        # Front/side edge clusters of manageable shrubs. No doorway or path
        # moved, no new tall trees hiding the important local house facades.
        for linevalue in yard.get('fence_lines',[]):
            line=LineString(linevalue)
            if line.length<4 or counts['shrubs']>=88:continue
            d=rng.uniform(.18,.8)*line.length;p=line.interpolate(d)
            for _ in range(5):
                x=p.x+rng.uniform(-1.4,1.4);y=p.y+rng.uniform(-1.4,1.4)
                if poly.buffer(-.7).contains(Point(x,y)) and full_axis.distance(Point(x,y))<37:
                    if shrub(x,y,rng.uniform(.55,.95),ident):break
            if counts['shrubs']>=88:break

    # Second art pass: garden-scale canopy masses, not isolated saplings.
    # Keep roughly a quarter of yards without any additional trees; selected
    # larger plots get asymmetric groups, mainly at side/rear boundaries.
    tree_roots=[]
    old_tree_roots=[tuple(ob.matrix_world.translation) for ob,slots in locals]
    tree_yard_counts={}
    garden_rng=random.Random(SEED+1967)

    def tube(batch,points,radii,sides=7,root=False):
        vv=[];uv=[];supports=[]
        for k,(center,radius) in enumerate(zip(points,radii)):
            tangent=(points[min(k+1,len(points)-1)]-points[max(0,k-1)]).normalized()
            u=tangent.cross(Vector((0,1,0)))
            if u.length<.01:u=tangent.cross(Vector((1,0,0)))
            u.normalize();v=tangent.cross(u).normalized()
            for j in range(sides):
                a=j*math.tau/sides;q=center+(u*math.cos(a)+v*math.sin(a))*radius
                if root and k==0:
                    support=ground.hit(q.x,q.y)
                    if support:q.z=support.z-.002;supports.append(len(vv))
                vv.append(q);uv.append((j/sides,(q.z-points[0].z)*1.8))
        faces=[]
        for k in range(len(points)-1):
            for j in range(sides):faces.append((k*sides+j,k*sides+(j+1)%sides,(k+1)*sides+(j+1)%sides,(k+1)*sides+j))
        batch.add(vv,faces,3,uv,support=supports);return vv

    def garden_tree(x,y,radius,height,house):
        if counts['new_garden_trees']>=44 or not safe.covers(Point(x,y).buffer(radius+.22)):return False
        if any((x-a)**2+(y-b)**2<2.35**2 for a,b,z in old_tree_roots+tree_roots):return False
        if any((x-a)**2+(y-b)**2<.40**2 for a,b,r in shrub_roots):return False
        root=ground.hit(x,y)
        if root is None:return False
        begin=len(garden_trees.v);allv=[];lean=Vector((garden_rng.uniform(-.24,.24),garden_rng.uniform(-.24,.24),0))
        trunk=[root,root+UP*height*.18+lean*.20,root+UP*height*.37+lean*.55,root+UP*height*.58+lean,root+UP*height*.77+lean*.86]
        allv+=tube(garden_trees,trunk,[.16,.135,.10,.065,.022],8,True)
        lobe_centers=[]
        for j in range(6):
            a=j*2.399963+garden_rng.uniform(-.35,.35)
            radial=Vector((math.cos(a),math.sin(a),0))
            origin=trunk[1].lerp(trunk[2],garden_rng.uniform(.2,.95))
            middle=root+UP*height*garden_rng.uniform(.44,.56)+radial*radius*.27+lean*.35
            tip=root+UP*height*garden_rng.uniform(.61,.76)+radial*radius*garden_rng.uniform(.36,.46)+lean*.35
            allv+=tube(garden_trees,[origin,middle,tip],[.065,.038,.014],6)
            for k in (-1,1):
                aa=a+k*garden_rng.uniform(.27,.48)
                c=root+Vector((math.cos(aa)*radius*garden_rng.uniform(.44,.53),math.sin(aa)*radius*garden_rng.uniform(.44,.53),height*garden_rng.uniform(.62,.78)))
                twigbase=middle.lerp(tip,.52)
                allv+=tube(garden_trees,[twigbase,twigbase.lerp(c,.58)+UP*.10,c],[.024,.013,.004],5)
                lobe_centers.append(c)
        # Actual individual leaf cards, arranged into 12 uneven crown lobes.
        # Leaves partly overlap; silhouette stays porous, with branch structure
        # visible below. No opaque sphere or camera-facing whole-tree card.
        leaf_count=garden_rng.randint(2650,3050);accepted_cards=0
        for j in range(leaf_count):
            center=lobe_centers[j%len(lobe_centers)]
            z=garden_rng.uniform(-1,1);a=garden_rng.uniform(0,math.tau);rr=garden_rng.random()**.40
            c=center+Vector((math.sqrt(1-z*z)*math.cos(a)*radius*.42*rr,math.sqrt(1-z*z)*math.sin(a)*radius*.42*rr,z*height*.19*rr))
            # Physical leaf size 13–24cm with mixed tilt and fork-local tint.
            length=garden_rng.uniform(.13,.24);width=garden_rng.uniform(.065,.13)
            az=garden_rng.uniform(0,math.tau);tilt=garden_rng.uniform(-1.25,1.25)
            u=Vector((math.cos(az),math.sin(az),0))*width/2
            v=Vector((-math.sin(az)*math.sin(tilt),math.cos(az)*math.sin(tilt),math.cos(tilt)))*length/2
            if card(garden_trees,c,u,v,(j//200+counts['new_garden_trees'])%3,j%4,garden_rng.uniform(.86,1.13)):
                allv.extend([c-u-v,c+u-v,c+u+v,c-u+v]);accepted_cards+=1
        accepted('garden_tree',root,allv,begin,len(garden_trees.v),{'batch':garden_trees.name,'height_m':max(p.z-root.z for p in allv),'design_height_m':height,'crown_radius_m':radius,'mapped_house_reference':house,'leaf_cards':accepted_cards,'new_inferred_tree':True})
        counts['new_garden_trees']+=1;counts['garden_tree_leaf_cards']+=accepted_cards;tree_roots.append(tuple(root));tree_yard_counts[str(house)]=tree_yard_counts.get(str(house),0)+1
        return True

    eligible=[]
    for yard in layout['yards']:
        if not 3800<yard['station_map_m']<4250:continue
        poly=_geom(yard['polygon']);free=poly.intersection(safe_geom).buffer(-1.95)
        if free.is_empty:continue
        focus=min(abs(yard['station_map_m']-3900),abs(yard['station_map_m']-4055))
        # Stable random component prevents a systematic tree in every plot.
        priority=poly.area*.035-focus*.012+garden_rng.uniform(-3,3)
        eligible.append((priority,yard))
    eligible.sort(key=lambda q:q[0],reverse=True)
    chosen=[y for _,y in eligible[:max(1,round(len(eligible)*.76))]]
    # First give selected plots a canopy; second/third passes form groups in
    # genuinely spacious plots instead of filling the remaining open yards.
    for pass_index in range(3):
        for yard in chosen:
            if counts['new_garden_trees']>=44:break
            if pass_index==2 and tree_yard_counts.get(str(yard['osm_id']),0)<2:continue
            poly=_geom(yard['polygon']);radius=garden_rng.uniform(1.85,2.48)
            if pass_index==2:radius=garden_rng.uniform(1.75,2.05)
            free=poly.intersection(safe_geom).buffer(-radius-.23)
            if free.is_empty:continue
            minx,miny,maxx,maxy=free.bounds;candidates=[]
            for k in range(180):
                x=garden_rng.uniform(minx,maxx);y=garden_rng.uniform(miny,maxy);point=Point(x,y)
                if not free.covers(point):continue
                if any((x-a)**2+(y-b)**2<2.35**2 for a,b,z in old_tree_roots+tree_roots):continue
                # Rear and side preference; avoid a centred ornamental tree.
                distance=full_axis.distance(point)
                score=distance*.65+garden_rng.uniform(-2,2)
                candidates.append((score,x,y))
            for _,x,y in sorted(candidates,reverse=True):
                height=garden_rng.uniform(4.7,6.9)
                if garden_tree(x,y,radius,height,yard['osm_id']):break
    objects=[o for b in (infill,shrubs,grass,garden_trees) if (o:=b.finish(collection,ground_names))]
    triangles=sum(len(o.data.loop_triangles) for o in objects)
    if triangles>MAX_TRIANGLES:raise AssertionError(f'Vegetation triangle budget exceeded: {triangles}')
    # Every new polygon, not only roots, is checked against exact reservations.
    violations=[]
    for ob in objects:
        for face in ob.data.polygons:
            p=MultiPoint([tuple(ob.data.vertices[i].co)[:2] for i in face.vertices]).convex_hull
            if not safe.covers(p):
                # float32 storage drift below 0.2mm is a representation effect.
                if not safe_geom.buffer(.0002).covers(p):violations.append({'object':ob.name,'polygon':face.index})
    if violations:raise AssertionError(f'New foliage violates reservations: {violations[:6]}')
    report={'version':'v16','seed':SEED,'module':'vegetation_v16.py','native_source':'chmyrivka_visual_v15.blend','counts':dict(counts,objects=len(objects),triangles=triangles),'triangle_budget':MAX_TRIANGLES,
      'sectors':[{'id':q['id'],'original_map_chainage_m':q['range'],'half_width_m':q['half']} for q in sectors],
      'hidden_original_objects':hidden,'material_rebindings':rebindings,'new_objects':[{'name':o.name,'role':'vegetation','kind':o['vegetation_kind'],'vertices':len(o.data.vertices),'triangles':len(o.data.loop_triangles),'materials':[m.name for m in o.data.materials],'support_vertex_indices':json.loads(o['v16_support_vertex_indices']),'reference_surface_names':ground_names,'support_offset_min_m':-.007,'support_offset_max_m':.007} for o in objects],
      'garden_tree_yards':tree_yard_counts,'garden_yards_without_new_tree':[str(y['osm_id']) for y in layout['yards'] if 3800<y['station_map_m']<4250 and str(y['osm_id']) not in tree_yard_counts],'ground_ray_misses':ground.ray_misses,'native_ground_objects':ground_names,'placements':records,'crown_infill':infill_records,'full_projected_geometry_reservation_violations':violations,
      'reservation':mapping(obstacle.intersection(region)),'new_planting_region':mapping(region),
      'source_geometry_changed':False,'source_material_graphs_changed':False,'native_images_changed':False,
      'limits':['Local original-map 3800–4250m and2260–2600m only; not whole-track foliage upgrade.','Planting is artist interpretation; no individual tree/plant survey.','Existing tree roots/branches/crown bounds preserved; minor local leaf material tint copied per object.','Four batched meshes are native Blender sources; dynamic LOD, ksTree/ksGrass translation and in-game FPS are pending.']}
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'data/vegetation_v16.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    scene['v16_vegetation_applied']=True;bpy.context.view_layer.update();return report
