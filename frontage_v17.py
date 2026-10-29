"""Driver-visible front gardens: new meshes only, source v16 is immutable.

Deterministic regional planting reconstruction (not photographed parcel truth).
All ground-contact vertices use native BVH ray hits. Projected faces must remain
inside a reservation-complement including track, pit, side streets, houses,
paths, gate throats, fence solids and preserved garden props.
"""
from __future__ import annotations
import sys,json,math,random
from pathlib import Path
import bpy,numpy as np,shapely
from mathutils import Vector
from shapely.geometry import Point,Polygon,LineString,MultiPoint,shape,mapping
from shapely.ops import unary_union
from shapely.prepared import prep
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'dependencies/v15'))
from verges_v15 import Ground,_exclusions,_geom,_parts
SEED=170648
UP=Vector((0,0,1))
TRIANGLE_BUDGET=240000

def _load(source,name):
    val=source.get(name) or source.get(name.removesuffix('.json'))
    return val if isinstance(val,dict) else json.loads((Path(source['source_data_dir'])/name).read_text())

class Batch:
    def __init__(self,name,materials,kind,offset=(-.007,.007)):
        self.name=name;self.materials=materials;self.kind=kind;self.offset=offset
        self.v=[];self.f=[];self.mi=[];self.uv=[];self.tint=[];self.support=[]
    def add(self,v,f,mi=0,uv=None,tint=(1,1,1,1),support=(),colors=None):
        off=len(self.v);self.v.extend(tuple(q) for q in v);self.f.extend(tuple(off+i for i in q) for q in f)
        self.mi.extend([mi]*len(f));self.uv.extend(uv if uv is not None else [(q[0]/2,q[1]/2) for q in v]);self.tint.extend(colors or [tint]*len(v));self.support.extend(off+i for i in support)
    def finish(self,collection,ground_names):
        if not self.f:return None
        me=bpy.data.meshes.new(self.name+'_mesh');me.from_pydata(self.v,[],self.f)
        for m in self.materials:me.materials.append(m)
        for p,mi in zip(me.polygons,self.mi):p.material_index=mi;p.use_smooth=self.kind!='ground_contact'
        uv=me.uv_layers.new(name='UVMap');uv.data.foreach_set('uv',np.asarray([self.uv[l.vertex_index] for l in me.loops],np.float32).ravel())
        vc=me.color_attributes.new(name='V17_FrontageTint',type='FLOAT_COLOR',domain='POINT');vc.data.foreach_set('color',np.asarray(self.tint,np.float32).ravel())
        me.update();me.calc_loop_triangles();ob=bpy.data.objects.new(self.name,me);collection.objects.link(ob)
        ob['v17_role']='frontage';ob['frontage_kind']=self.kind;ob['v17_visual_only']=True
        ob['v17_support_vertex_indices']=json.dumps(self.support,separators=(',',':'));ob['v17_reference_surface_names']=json.dumps(ground_names,separators=(',',':'))
        ob['v17_support_offset_min_m']=self.offset[0];ob['v17_support_offset_max_m']=self.offset[1]
        ob['placement_evidence']='Artist reconstruction in mapped house frontages, not surveyed species or exact garden evidence.'
        return ob

def _plain(name,color,transmission=.18):
    m=bpy.data.materials.new('V17_Frontage_'+name);m.use_nodes=True;m.diffuse_color=(*color,1)
    n,l=m.node_tree.nodes,m.node_tree.links;p=n.get('Principled BSDF');p.inputs['Roughness'].default_value=.84;p.inputs['Specular IOR Level'].default_value=.17
    vc=n.new('ShaderNodeVertexColor');vc.layer_name='V17_FrontageTint';mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1;mix.inputs[1].default_value=(*color,1)
    l.new(vc.outputs['Color'],mix.inputs[2]);l.new(mix.outputs[0],p.inputs['Base Color'])
    if transmission:
        tr=n.new('ShaderNodeBsdfTranslucent');l.new(mix.outputs[0],tr.inputs['Color']);blend=n.new('ShaderNodeMixShader');blend.inputs[0].default_value=transmission;l.new(p.outputs[0],blend.inputs[1]);l.new(tr.outputs[0],blend.inputs[2]);l.new(blend.outputs[0],n.get('Material Output').inputs[0])
    return m

def _leaf_material(index):
    base=bpy.data.materials.get('V10_Game_Foliage_singleleaf')
    if base is None:raise RuntimeError('Required native V10 leaf atlas missing')
    m=base.copy();m.name=f'V17_Frontage_Leaf_{index}';n,l=m.node_tree.nodes,m.node_tree.links
    tx=next(q for q in n if q.type=='TEX_IMAGE' and q.image and 'diffuse' in q.image.name)
    destinations=[q.to_socket for q in list(tx.outputs['Color'].links)]
    h=n.new('ShaderNodeHueSaturation');h.inputs['Saturation'].default_value=(.72,.81,.78)[index];h.inputs['Value'].default_value=(1.06,.98,.87)[index]
    l.new(tx.outputs['Color'],h.inputs['Color']);vc=n.new('ShaderNodeVertexColor');vc.layer_name='V17_FrontageTint';mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1
    l.new(h.outputs[0],mix.inputs[1]);l.new(vc.outputs['Color'],mix.inputs[2])
    for s in destinations:l.new(mix.outputs[0],s)
    for p in n:
        if p.type=='BSDF_PRINCIPLED':p.inputs['Roughness'].default_value=.78;p.inputs['Specular IOR Level'].default_value=.19
        if p.type=='NORMAL_MAP':p.inputs['Strength'].default_value=.20
    m['source_atlas_unchanged']=True;m['v17_copy_of']=base.name
    return m

def _soil_material():
    base=bpy.data.materials.get('V16_Yard_Grass_Wear')
    if base is None:raise RuntimeError('v16 ground contact material missing')
    m=base.copy();m.name='V17_Frontage_Subtle_FenceBase_Contact'
    for n in m.node_tree.nodes:
        if n.type=='VERTEX_COLOR':n.layer_name='V17_FrontageTint'
        if n.type=='MAP_RANGE':n.inputs['To Min'].default_value=.045;n.inputs['To Max'].default_value=.29
    m['v17_copy_of']=base.name;return m

def apply(scene,collection,source_data=None):
    if scene.get('v17_frontage_applied'):raise RuntimeError('Already applied: build from immutable v16')
    source=dict(source_data or {});source.setdefault('source_data_dir',ROOT/'source_data')
    rng=random.Random(SEED);track=source.get('track_v08') or _load(source,'track_plan_v08.json');layout=_load(source,'yard_layout_v12.json')
    ss=np.array([q['s'] for q in track['samples']]);xy=np.array([(q['x'],q['y']) for q in track['samples']]);full_axis=LineString(xy)
    axis=LineString([(float(np.interp(3800,ss,xy[:,0])),float(np.interp(3800,ss,xy[:,1])))]+[tuple(q) for s,q in zip(ss,xy) if 3800<s<4250]+[(float(np.interp(4250,ss,xy[:,0])),float(np.interp(4250,ss,xy[:,1])))])
    region=axis.buffer(35,cap_style=2);obstacle,roads,houses,paths,native_paved=_exclusions(scene,region,source,track,layout)
    reserves=[];fences=[];gate_reserves=[]
    for y in layout['yards']:
        if not 3800<y['station_map_m']<4250:continue
        for line in y['fence_lines']:fences.append(LineString(line).buffer(.16))
        tangent=Vector((*y['front_tangent'],0));normal=Vector((*y['front_normal'],0))
        for key,width in [('entrance_xy',3.55),('pedestrian_gate_xy',1.30)]:
            c=Vector((*y[key],0));p=Polygon([tuple((c+tangent*u+normal*v)[:2]) for u,v in [(-width/2,-2.0),(width/2,-2.0),(width/2,1.8),(-width/2,1.8)]])
            gate_reserves.append(p)
    reserves += fences+gate_reserves
    architecture=source.get('architecture_report') or source.get('architecture_v17') or {}
    for canopy in architecture.get('canopy_reserves',[]):
        reserves.append(Polygon(canopy['polygon_xy']).buffer(.08))
    yard_report=source.get('yards_v16') or _load(source,'yards_v16.json')
    for p in yard_report.get('placements',[]):
        if p['kind']!='uneven_grass_wear':reserves.append(_geom(p['footprint']).buffer(.08))
    # Real protective barriers and racing kerbs, projected per face rather than
    # using huge batch bounding rectangles.
    for ob in scene.objects:
        if ob.type!='MESH' or ob.hide_render:continue
        if not ('Barrier' in ob.name or ob.get('v15_role')=='racing_kerb'):continue
        bb=MultiPoint([tuple(ob.matrix_world@Vector(q))[:2] for q in ob.bound_box]).convex_hull
        if not bb.intersects(region):continue
        w=[ob.matrix_world@v.co for v in ob.data.vertices];ps=[]
        for f in ob.data.polygons:
            p=MultiPoint([tuple(w[i])[:2] for i in f.vertices]).convex_hull
            if p.intersects(region):ps.append(p)
        if ps:reserves.append(unary_union(ps).buffer(.12))
    # Existing plants stay intact. Reserve roots and most shrub crown envelope,
    # while allowing intentional low grass underneath existing garden trees.
    veg_report=source.get('vegetation_v16') or _load(source,'vegetation_v16.json')
    for p in veg_report['placements']:
        if p['kind']=='shrub':reserves.append(_geom(p['footprint']).buffer(.05))
        elif p['kind']=='garden_tree':reserves.append(Point(p['root_xyz'][:2]).buffer(.30))
    obstacle=unary_union([obstacle,*reserves]);safe_geom=region.difference(obstacle);safe=prep(safe_geom);safe_fine=prep(safe_geom.buffer(.00001));safe_float=prep(safe_geom.buffer(.0002))
    ground=Ground(scene,region.buffer(2));leafm=[_leaf_material(i) for i in range(3)];bark=bpy.data.materials['V10_Game_Bark_Dark']
    greens=[_plain(f'FineGrass_{i}',c,.25) for i,c in enumerate([(.075,.106,.031),(.111,.139,.048),(.169,.178,.075),(.224,.211,.128)])]
    plantm=[_plain('PerennialLeaf',(.093,.127,.042),.25),_plain('DryStem',(.17,.16,.076),0),_plain('IvoryPetal',(.48,.43,.30),.14),_plain('DustyRosePetal',(.30,.12,.14),.14),_plain('OchreCentre',(.27,.19,.055),0)]
    shrubs=Batch('V17_Frontage_Layered_Shrubs',leafm+[bark],'front_shrubs')
    grass=Batch('V17_Frontage_Fencebase_Grass',greens,'fencebase_grass')
    flowers=Batch('V17_Frontage_Perennial_Groups',plantm,'perennial_groups')
    soil=Batch('V17_Frontage_Ground_Transition',[_soil_material()],'ground_contact',(.003,.012))
    counts={'shrubs':0,'shrub_leaf_cards':0,'grass_tufts':0,'grass_blades':0,'perennial_clumps':0,'flower_heads':0,'contact_patches':0};records=[];yard_counts={};plant_roots=[]

    def record(batch,kind,root,start,house,extra=None):
        points=batch.v[start:];hull=MultiPoint([q[:2] for q in points]).convex_hull
        if not safe_fine.covers(hull):raise AssertionError(('Placement outside safe region',kind,house))
        r={'kind':kind,'batch':batch.name,'root_xyz':list(root),'start_vertex':start,'end_vertex_exclusive':len(batch.v),'mapped_house_reference':house,'footprint':mapping(hull),'road_clearance_m':hull.distance(roads),'building_clearance_m':hull.distance(houses),'path_clearance_m':hull.distance(paths)}
        if extra:r.update(extra)
        records.append(r)

    def tube(batch,points,radii,mi,sides=5,grounded=False):
        v=[];uv=[];support=[]
        for k,(c,rad) in enumerate(zip(points,radii)):
            direction=(points[min(k+1,len(points)-1)]-points[max(k-1,0)]).normalized();u=direction.cross(Vector((0,1,0)))
            if u.length<.01:u=direction.cross(Vector((1,0,0)))
            u.normalize();w=direction.cross(u).normalized()
            for j in range(sides):
                a=j*math.tau/sides;p=c+(u*math.cos(a)+w*math.sin(a))*rad
                if k==0 and grounded:
                    h=ground.hit(p.x,p.y)
                    if h is None:raise RuntimeError('Tube root missed ground')
                    p.z=h.z-.001;support.append(len(v))
                v.append(p);uv.append((j/sides,(p.z-points[0].z)*2))
        f=[(k*sides+j,k*sides+(j+1)%sides,(k+1)*sides+(j+1)%sides,(k+1)*sides+j) for k in range(len(points)-1) for j in range(sides)]
        batch.add(v,f,mi,uv,support=support)

    def leaf(batch,c,u,v,mi=0,tile=0,tint=1):
        vv=[c-u-v,c+u-v,c+u+v,c-u+v];hull=MultiPoint([tuple(q)[:2] for q in vv]).convex_hull
        if not safe.covers(hull):return False
        ox=(tile%2)*.5;oy=(tile//2)*.5;pad=.003
        uv=[(ox+pad,oy+pad),(ox+.5-pad,oy+pad),(ox+.5-pad,oy+.5-pad),(ox+pad,oy+.5-pad)]
        batch.add(vv,[(0,1,2,3)],mi,uv,tint=(tint,tint,tint,1));return True

    def add_shrub(x,y,height,radius,house,camera_target=None):
        if counts['shrubs']>=48 or not safe.covers(Point(x,y).buffer(radius+.12)):return False
        if any((x-a)**2+(y-b)**2<(radius+r+.16)**2 for a,b,r in plant_roots):return False
        h=ground.hit(x,y)
        if h is None:return False
        start=len(shrubs.v);lobes=[]
        for j in range(7 if camera_target else 5):
            a=j*2.399963+rng.uniform(-.25,.25);radial=Vector((math.cos(a),math.sin(a),0))
            b=h+radial*.025;top=h+radial*radius*.51+UP*height*rng.uniform(*((.73,.81) if camera_target else (.58,.75)))
            tube(shrubs,[b,b.lerp(top,.42)+UP*.09,top],[.013,.009,.003],3,5,True)
            lobes.append(top-UP*.06)
        cards=0
        for j in range(1020 if camera_target else 660):
            c=lobes[j%len(lobes)];a=rng.uniform(0,math.tau);zz=rng.uniform(-1,1);r=rng.random()**.34
            p=c+Vector((math.sqrt(1-zz*zz)*math.cos(a)*radius*.43*r,math.sqrt(1-zz*zz)*math.sin(a)*radius*.43*r,zz*height*(.19 if camera_target else .24)*r))
            az=rng.uniform(0,math.tau);tilt=rng.uniform(-1.25,1.25)
            u=Vector((math.cos(az),math.sin(az),0))*rng.uniform(.029,.047);v=Vector((-math.sin(az)*math.sin(tilt),math.cos(az)*math.sin(tilt),math.cos(tilt)))*rng.uniform(.065,.10)
            if leaf(shrubs,p,u,v,j%3,j%4,rng.uniform(.88,1.14)):cards+=1
        record(shrubs,'front_shrub',h,start,house,{'height_m':height,'crown_radius_m':radius,'leaf_cards':cards,'camera_selected':bool(camera_target),'visibility_basis':camera_target});plant_roots.append((x,y,radius))
        counts['shrubs']+=1;counts['shrub_leaf_cards']+=cards;return True

    def add_tuft(x,y,house,heightrange=(.045,.21)):
        if counts['grass_tufts']>=2800 or not safe.covers(Point(x,y).buffer(.19)):return False
        h=ground.hit(x,y)
        if h is None:return False
        start=len(grass.v)
        for j in range(rng.randint(7,11)):
            a=rng.uniform(0,math.tau);b=ground.hit(x+math.cos(a)*rng.uniform(0,.07),y+math.sin(a)*rng.uniform(0,.07))
            if b is None:continue
            height=rng.uniform(*heightrange);side=Vector((math.cos(a+.4),math.sin(a+.4),0))*rng.uniform(.003,.006);lean=Vector((math.cos(a),math.sin(a),0))*rng.uniform(.025,.10)
            left=ground.hit(b.x-side.x,b.y-side.y);right=ground.hit(b.x+side.x,b.y+side.y)
            if left is None or right is None:continue
            left.z-=.001;right.z-=.001;mid=b+UP*height*.52+lean*.25;tip=b+UP*height+lean
            shade=rng.uniform(.82,1.08);grass.add([left,right,mid+side*.58,mid-side*.58,tip],[(0,1,2,3),(3,2,4)],rng.choices(range(4),[.29,.41,.24,.06])[0],support=(0,1),colors=[(.54*shade,)*3+(1,)]*2+[(shade,)*3+(1,)]*3)
            counts['grass_blades']+=1
        if len(grass.v)==start:return False
        record(grass,'fencebase_tuft',h,start,house);counts['grass_tufts']+=1;return True

    def add_perennial(x,y,height,house,flower=True):
        radius=.30
        if counts['perennial_clumps']>=145 or not safe.covers(Point(x,y).buffer(.47)):return False
        h=ground.hit(x,y)
        if h is None:return False
        start=len(flowers.v)
        for j in range(rng.randint(4,7)):
            a=rng.uniform(0,math.tau);radial=Vector((math.cos(a),math.sin(a),0));b=h+radial*.05;hh=height*rng.uniform(.68,1.06);tip=h+radial*rng.uniform(.06,.20)+UP*hh
            tube(flowers,[b,b.lerp(tip,.45),tip],[.0035,.0023,.001],1,4,True)
            for k in range(3):
                c=b.lerp(tip,.18+k*.19);direction=Vector((math.cos(a+k*2.2),math.sin(a+k*2.2),.28)).normalized();s=direction.cross(UP).normalized()*rng.uniform(.025,.041);end=c+direction*rng.uniform(.14,.23)
                vv=[c,c.lerp(end,.42)+s,c.lerp(end,.70)+UP*.018,end,c.lerp(end,.42)-s]
                flowers.add(vv,[(0,1,2,3,4)],0,tint=(rng.uniform(.82,1.08),)*3+(1,))
            if flower and j%2==0:
                # Small open compound heads, grouped perennials rather than a
                # saturated flower carpet. Petals fold upward toward the tip.
                center=tip+UP*.007;petalmat=2 if int(house.split('/')[-1])%4 else 3
                for k in range(7):
                    aa=k*math.tau/7+rng.uniform(-.09,.09);d=Vector((math.cos(aa),math.sin(aa),.15));s=Vector((-math.sin(aa),math.cos(aa),0))*.016
                    end=center+d*rng.uniform(.035,.050);flowers.add([center,center.lerp(end,.6)+s,end,center.lerp(end,.6)-s],[(0,1,2,3)],petalmat)
                vv=[center+Vector((math.cos(k*math.tau/7)*.012,math.sin(k*math.tau/7)*.012,.006)) for k in range(7)];flowers.add(vv,[tuple(range(7))],4);counts['flower_heads']+=1
        record(flowers,'perennial_clump',h,start,house,{'design_height_m':height,'flowering':flower});counts['perennial_clumps']+=1;return True

    def contact_patch(poly,house):
        inside=poly.buffer(-.16);made=0;start=len(soil.v)
        for band in [inside,poly.difference(inside)]:
            if band.is_empty:continue
            for idx in ground.tree.query(band,predicate='intersects'):
                plane=ground.planes[idx]
                for part in _parts(band.intersection(ground.polys[idx])):
                    for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
                        if tri.area<1e-6:continue
                        co=list(tri.exterior.coords)[:-1]
                        if not tri.exterior.is_ccw:co.reverse()
                        c=tri.centroid;hit=ground.hit(c.x,c.y)
                        if hit is None or abs(hit.z-(c.x*plane[0]+c.y*plane[1]+plane[2]))>.003:continue
                        vv=[(x,y,float(x*plane[0]+y*plane[1]+plane[2]+.006)) for x,y in co]
                        if any((hit:=ground.hit(x,y)) is None or abs(hit.z-(z-.006))>.003 for x,y,z in vv):continue
                        alpha=[min(1,poly.boundary.distance(Point(x,y))/.16) for x,y,z in vv]
                        soil.add(vv,[(0,1,2)],support=(0,1,2),colors=[(1,1,1,a) for a in alpha]);made+=1
        if made:
            # Multiple clipped patch pieces need per-face containment rather
            # than a convex hull across a protected path or fence.
            records.append({'kind':'ground_contact','batch':soil.name,'start_vertex':start,'end_vertex_exclusive':len(soil.v),'mapped_house_reference':house,'footprint':mapping(poly),'vertical_offset_m':.006})
            counts['contact_patches']+=1

    # Six bounded art-review corrections, selected from camera projection and
    # per-face structural ray visibility. The old review scene SHA is retained
    # as evidence; final render remains the actual visual acceptance test.
    visibility=json.loads((ROOT/'data/frontage_visibility_targets_v17.json').read_text())
    camera_targets=visibility['targets'];camera_house_ids={q['osm_id'] for q in camera_targets}
    saved_rng=rng;rng=random.Random(SEED+2041)
    accepted_targets=[]
    for target in camera_targets:
        x,y,z=target['root_xyz']
        if not add_shrub(x,y,target['design_height_m'],target['radius_m'],target['osm_id'],target):
            raise AssertionError(('Camera-selected front shrub rejected',target['osm_id']))
        accepted_targets.append(target)
    rng=saved_rng

    yards=sorted([y for y in layout['yards'] if 3800<y['station_map_m']<4250],key=lambda y:(min(abs(y['station_map_m']-3900),abs(y['station_map_m']-4045)),y['osm_id']))
    for y in yards:
        ident=y['osm_id'];poly=_geom(y['polygon']);front=LineString(y['front_edge']);n=Vector((*y['front_normal'],0));u=Vector((*y['front_tangent'],0));house=_geom(y['house_polygon'])
        usable=poly.intersection(safe_geom);before=counts.copy()
        # Only the front 4.8 m and first side-gap metres are eligible. Every
        # fourth plot is kept largely open to preserve variety and sightlines.
        frontzone=front.buffer(4.8,cap_style=2).intersection(usable);free=frontzone.buffer(-.62)
        if not free.is_empty and int(ident.split('/')[-1])%4!=2 and ident not in camera_house_ids:
            candidates=[]
            for _ in range(95):
                x=rng.uniform(free.bounds[0],free.bounds[2]);yy=rng.uniform(free.bounds[1],free.bounds[3]);p=Point(x,yy)
                if not free.covers(p):continue
                # Front corners and side gaps, not a continuous window hedge.
                enddistance=min(p.distance(Point(front.coords[0])),p.distance(Point(front.coords[-1])))
                score=front.distance(p)*.65+enddistance*.10+rng.uniform(0,.8)
                candidates.append((score,x,yy))
            made=0
            for _,x,yy in sorted(candidates):
                if add_shrub(x,yy,rng.uniform(1.25,1.85),rng.uniform(.52,.68),ident):
                    made+=1
                    if made>=(2 if front.length>17 else 1):break
        # Irregular grouped perennials beside open fence portions, allowed to
        # rise above some low fence panels without blocking the house window.
        if not frontzone.is_empty:
            for d in np.arange(.8,front.length-.5,1.0):
                if rng.random()<.43:continue
                p=front.interpolate(float(d));offset=rng.uniform(.55,1.25);q=Vector((p.x,p.y,0))-n*offset
                if frontzone.covers(Point(q.x,q.y).buffer(.4)) and not any((q.x-a)**2+(q.y-b)**2<(r+.32)**2 for a,b,r in plant_roots):
                    add_perennial(q.x,q.y,rng.uniform(.65,1.42),ident,flower=int(ident.split('/')[-1])%3!=1)
        # Front-of-fence strip stays low. Groups are separated by bare/mown
        # gaps; gate throats were removed above and therefore remain clear.
        for d in np.arange(.4,front.length-.3,.35):
            if math.sin(d*.7+int(ident.split('/')[-1])%11)<-.25 or rng.random()<.14:continue
            p=front.interpolate(float(d))
            for k in range(rng.randint(2,4)):
                off=rng.uniform(.24,.75)*(1 if k%3 else -1);q=Vector((p.x,p.y,0))+n*off+u*rng.uniform(-.16,.16)
                add_tuft(q.x,q.y,ident,(.045,.22 if off>0 else .30))
        # Sparse leaf/soil contact under sections of fence; never a continuous
        # brown band and never an opaque patch covering native detail.
        for d in np.arange(1,front.length-1,4.2):
            if rng.random()<.38:continue
            p=front.interpolate(float(d));c=Vector((p.x,p.y,0))+n*.34
            coords=[]
            for k in range(19):
                a=k*math.tau/19;r=rng.uniform(.83,1.1);q=c+u*math.cos(a)*rng.uniform(.7,1.2)*r+n*math.sin(a)*.38*r;coords.append(q[:2])
            raw=Polygon(coords)
            if not raw.is_valid:raw=raw.buffer(0)
            for part in _parts(raw.intersection(safe_geom)):
                if part.area>.12:contact_patch(part,ident)
        yard_counts[ident]={k:counts[k]-before[k] for k in counts}
        selected_here=[r for r in records if r.get('camera_selected') and r['mapped_house_reference']==ident]
        yard_counts[ident]['camera_selected_shrubs']=len(selected_here)
        yard_counts[ident]['shrubs']+=len(selected_here)
        yard_counts[ident]['shrub_leaf_cards']+=sum(r['leaf_cards'] for r in selected_here)
    objects=[ob for b in (shrubs,grass,flowers,soil) if (ob:=b.finish(collection,ground.names))]
    # Validate every polygon including projected degenerate vertical faces;
    # float32 geometry storage tolerance is at most 0.2 mm at these coordinates.
    violations=[];checks=0
    for ob in objects:
        for f in ob.data.polygons:
            hull=MultiPoint([tuple(ob.data.vertices[i].co)[:2] for i in f.vertices]).convex_hull;checks+=1
            if not safe_float.covers(hull):violations.append({'object':ob.name,'polygon':f.index})
    if violations:raise AssertionError(('Frontage projected collision',violations[:8]))
    triangles=sum(len(ob.data.loop_triangles) for ob in objects)
    if triangles>TRIANGLE_BUDGET:raise AssertionError(('Triangle budget',triangles))
    supports=[];maxdelta=0;support_checks=0
    for ob in objects:
        ids=json.loads(ob['v17_support_vertex_indices']);lo=ob['v17_support_offset_min_m'];hi=ob['v17_support_offset_max_m']
        for i in ids:
            p=ob.data.vertices[i].co;hit=ground.hit(p.x,p.y)
            if hit is None:raise AssertionError(('Support missed ground',ob.name,i))
            delta=p.z-hit.z;support_checks+=1;maxdelta=max(maxdelta,abs(delta))
            if not lo-.0002<=delta<=hi+.0002:raise AssertionError(('Support offset',ob.name,i,delta,lo,hi))
        supports.append({'name':ob.name,'role':'frontage','kind':ob['frontage_kind'],'vertices':len(ob.data.vertices),'triangles':len(ob.data.loop_triangles),'support_vertex_indices':ids,'reference_surface_names':ground.names,'support_offset_min_m':lo,'support_offset_max_m':hi,'materials':[m.name for m in ob.data.materials]})
    report={'version':'v17','module':'frontage_v17.py','seed':SEED,'source':'immutable chmyrivka_visual_v16.blend','original_map_chainage_m':[3800,4250],'counts':dict(counts,objects=len(objects),triangles=triangles,yards=len(yards)),'triangle_budget':TRIANGLE_BUDGET,'new_objects':supports,'placements':records,'yards':yard_counts,'reservation':mapping(obstacle.intersection(region)),'new_planting_region':mapping(region),'native_ground_objects':ground.names,'camera_selected_groups':accepted_targets,'support_checks':support_checks,'maximum_absolute_support_offset_m':maxdelta,'ground_ray_misses':ground.ray_misses,'projected_polygon_checks':checks,'full_projected_geometry_reservation_violations':violations,'material_rebindings':[],'hidden_original_objects':[],'source_geometry_changed':False,'source_material_graphs_changed':False,'native_images_changed':False,'limits':['Regional artist reconstruction, no exact species or parcel photographs claimed.','Frontage art only, no new background trees or full-track upgrade.','New alpha and foliage materials are native Blender source; AC shader translation, LOD and FPS unverified.']}
    outdir=Path(source.get('output_data_dir',ROOT/'data'));outdir.mkdir(exist_ok=True);(outdir/'frontage_v17.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    scene['v17_frontage_applied']=True;bpy.context.view_layer.update();return report
