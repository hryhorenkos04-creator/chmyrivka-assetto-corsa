"""Native residential art pass. Original mapped geometry is immutable.

apply(scene, collection, source_data=None) -> JSON report. The 33 existing
inferred plots are NOT cadastral evidence. No exact address/facade claim.
Ground / terrain clipping follows verges_v15.py, copied here for independence.
"""
from pathlib import Path
import json, math, random
import bpy, bmesh, shapely
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from shapely.geometry import Polygon, Point, LineString, MultiPoint, shape, mapping
from shapely.ops import unary_union
from shapely.strtree import STRtree

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'chmyrivka_v15'/'baseline_data'
UP=Vector((0,0,1)); SEED=160642; LIMIT_TRIS=200000

def _parts(g): return [p for p in shapely.get_parts(g) if p.geom_type=='Polygon' and p.area>1e-5]
def _load(source,name):
    v=source.get(name) or source.get(name.removesuffix('.json'))
    if isinstance(v,dict):return v
    return json.loads((Path(source.get('source_data_dir',SOURCE))/name).read_text())

class Ground:
    def __init__(self,scene,region):
        verts=[];faces=[];self.polys=[];self.planes=[];self.names=[];self.ray_misses=0
        for ob in scene.objects:
            if ob.type!='MESH' or not ob.name.startswith('GROUND_') or any(w in ob.name.lower() for w in ('water','wetland')):continue
            if not MultiPoint([tuple(ob.matrix_world@Vector(v))[:2] for v in ob.bound_box]).convex_hull.intersects(region):continue
            ob.data.calc_loop_triangles();world=[ob.matrix_world@v.co for v in ob.data.vertices];used=False
            for tri in ob.data.loop_triangles:
                xyz=np.asarray([world[i] for i in tri.vertices]);p=Polygon(xyz[:,:2])
                if p.area<1e-8 or not p.intersects(region):continue
                off=len(verts);verts.extend(xyz);faces.append((off,off+1,off+2));self.polys.append(p)
                self.planes.append(np.linalg.solve(np.c_[xyz[:,:2],np.ones(3)],xyz[:,2]));used=True
            if used:self.names.append(ob.name)
        self.bvh=BVHTree.FromPolygons(verts,faces,all_triangles=True);self.tree=STRtree(self.polys)
    def hit(self,xy):
        h=self.bvh.ray_cast(Vector((xy[0],xy[1],500)),Vector((0,0,-1)),1000)[0]
        if h is None:self.ray_misses+=1;raise ValueError(('No native yard ground',xy))
        return h

class Batch:
    def __init__(self,name,mats,role):
        self.name=name;self.mats=mats;self.role=role;self.v=[];self.f=[];self.mi=[];self.alpha=[];self.supports=[]
    def poly(self,mat,vs,fs,alpha=None,supports=None):
        off=len(self.v);self.v.extend(tuple(v) for v in vs);self.f.extend(tuple(off+i for i in f) for f in fs)
        self.mi.extend([mat]*len(fs));self.alpha.extend(alpha or [1.]*len(vs));self.supports.extend(off+i for i in (supports or []))
    def box(self,mat,c,axes,size,support=False):
        a,b,d=axes;c=Vector(c)
        vs=[c+a*x*size[0]/2+b*y*size[1]/2+d*z*size[2]/2 for x,y,z in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
        self.poly(mat,vs,[(0,3,2,1),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7)])
    def beam(self,mat,a,b,w,h=None):
        a,b=Vector(a),Vector(b);u=(b-a).normalized();n=u.cross(UP)
        if n.length<.01:n=Vector((1,0,0))
        n.normalize();z=u.cross(n).normalized();self.box(mat,(a+b)*.5,(u,n,z),((b-a).length,w,h or w))
    def cylinder(self,mat,c,r,h,n=16,cap=True):
        c=Vector(c);vs=[c+Vector((r*math.cos(j*math.tau/n),r*math.sin(j*math.tau/n),z)) for z in (0,h) for j in range(n)]
        fs=[(j,(j+1)%n,(j+1)%n+n,j+n) for j in range(n)]
        if cap:fs.extend([tuple(reversed(range(n))),tuple(range(n,2*n))])
        self.poly(mat,vs,fs)
    def finish(self,collection):
        if not self.f:return None
        me=bpy.data.meshes.new(self.name+'_mesh');me.from_pydata(self.v,[],self.f)
        for m in self.mats:me.materials.append(m)
        for p,mi in zip(me.polygons,self.mi):p.material_index=mi
        uv=me.uv_layers.new(name='V16_MetreUV');uv.data.foreach_set('uv',np.array([(me.vertices[l.vertex_index].co.x/2,me.vertices[l.vertex_index].co.y/2) for l in me.loops],np.float32).reshape(-1))
        col=me.color_attributes.new(name='V16_YardFade',type='FLOAT_COLOR',domain='POINT');col.data.foreach_set('color',np.asarray([(1,1,1,a) for a in self.alpha],np.float32).reshape(-1))
        me.update();me.calc_loop_triangles();ob=bpy.data.objects.new(self.name,me);collection.objects.link(ob)
        ob['v16_role']=self.role;ob['v16_visual_only']=True;ob['position_evidence']='Regional inferred domestic dressing on unchanged v12 plot layout; not exact addresses.'
        return ob

def _mat(name,color,rough=.8,metal=0,grain=False):
    m=bpy.data.materials.new('V16_Yard_'+name);m.use_nodes=True;m.diffuse_color=(*color,1)
    nn,ll=m.node_tree.nodes,m.node_tree.links;p=nn.get('Principled BSDF');p.inputs['Roughness'].default_value=rough;p.inputs['Metallic'].default_value=metal
    geo=nn.new('ShaderNodeNewGeometry');noise=nn.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=3.7;noise.inputs['Detail'].default_value=3
    if grain:
        stretch=nn.new('ShaderNodeVectorMath');stretch.operation='MULTIPLY';stretch.inputs[1].default_value=(18,18,.35);ll.new(geo.outputs['Position'],stretch.inputs[0]);ll.new(stretch.outputs['Vector'],noise.inputs['Vector'])
    else:ll.new(geo.outputs['Position'],noise.inputs['Vector'])
    ramp=nn.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].position=.22;ramp.color_ramp.elements[0].color=(*[v*.63 for v in color],1);ramp.color_ramp.elements[1].position=.78;ramp.color_ramp.elements[1].color=(*[v*1.12 for v in color],1)
    ll.new(noise.outputs['Fac'],ramp.inputs['Fac']);ll.new(ramp.outputs['Color'],p.inputs['Base Color'])
    fine=nn.new('ShaderNodeTexNoise');fine.inputs['Scale'].default_value=95;fine.inputs['Detail'].default_value=2;ll.new(geo.outputs['Position'],fine.inputs['Vector'])
    bump=nn.new('ShaderNodeBump');bump.inputs['Distance'].default_value=.0018 if grain else .0006;bump.inputs['Strength'].default_value=.28;ll.new(fine.outputs['Fac'],bump.inputs['Height']);ll.new(bump.outputs['Normal'],p.inputs['Normal'])
    return m

def _materials():
    specs=[('Dull_Galvanized',(.22,.24,.22),.66,.55),('Faded_Sage',(.19,.25,.19),.82,.22),('Faded_Blue',(.20,.29,.32),.79,.25),('Old_Iron',(.115,.135,.12),.83,.36),('Wood_Silver',(.24,.23,.19),.92,0),('Wood_Ochre',(.30,.25,.17),.90,0),('Concrete_Post',(.32,.32,.29),.91,0),('Rust_Seams',(.21,.105,.055),.98,.05),('Pale_Pickets',(.44,.45,.36),.87,0),('Dark_Barrel',(.105,.16,.135),.81,.20)]
    mats=[_mat(n,c,r,m,'Wood' in n) for n,c,r,m in specs]
    for name,color in [('Garden_Soil',(.115,.083,.045)),('Worn_Earth',(.22,.18,.12)),('Grass_Wear',(.17,.20,.08))]:
        m=_mat(name,color,.96);nn,ll=m.node_tree.nodes,m.node_tree.links;p=nn.get('Principled BSDF')
        attr=nn.new('ShaderNodeVertexColor');attr.layer_name='V16_YardFade';ll.new(attr.outputs['Alpha'],p.inputs['Alpha']);m.surface_render_method='DITHERED'
        # Existing native CC0 dirt normal adds photographed grain at a 2 m tile.
        im=next((i for i in bpy.data.images if i.name.startswith('dirt_nor_gl_2k.jpg')),None)
        if im:
            tc=nn.new('ShaderNodeTexCoord');tx=nn.new('ShaderNodeTexImage');tx.image=im;nm=nn.new('ShaderNodeNormalMap');nm.inputs['Strength'].default_value=.32
            ll.new(tc.outputs['UV'],tx.inputs['Vector']);ll.new(tx.outputs['Color'],nm.inputs['Color']);ll.new(nm.outputs['Normal'],p.inputs['Normal'])
        if name=='Grass_Wear':
            # Art review: an opaque olive procedural patch became a pale sticker
            # against the native grass. Use restrained photographic earth grain,
            # irregular sparse coverage and at most 38% opacity instead.
            dirt=next((i for i in bpy.data.images if i.name.startswith('dirt_diff_2k.jpg')),None)
            if dirt:
                coord=nn.new('ShaderNodeTexCoord');albedo=nn.new('ShaderNodeTexImage');albedo.image=dirt;albedo.label='Existing packed CC0 dirt, 2 m tile'
                ll.new(coord.outputs['UV'],albedo.inputs['Vector'])
                tint=nn.new('ShaderNodeMixRGB');tint.blend_type='MULTIPLY';tint.inputs[0].default_value=1;tint.inputs[2].default_value=(.35,.38,.22,1)
                ll.new(albedo.outputs['Color'],tint.inputs[1]);ll.new(tint.outputs['Color'],p.inputs['Base Color'])
            else:
                for link in list(p.inputs['Base Color'].links):ll.remove(link)
                p.inputs['Base Color'].default_value=(.055,.068,.027,1)
            geo=nn.new('ShaderNodeNewGeometry');coverage=nn.new('ShaderNodeTexNoise');coverage.inputs['Scale'].default_value=3.2;coverage.inputs['Detail'].default_value=3;coverage.inputs['Roughness'].default_value=.72
            ll.new(geo.outputs['Position'],coverage.inputs['Vector'])
            remap=nn.new('ShaderNodeMapRange');remap.clamp=True;remap.inputs['From Min'].default_value=.28;remap.inputs['From Max'].default_value=.72;remap.inputs['To Min'].default_value=.035;remap.inputs['To Max'].default_value=.38
            ll.new(coverage.outputs['Fac'],remap.inputs['Value'])
            alpha=nn.new('ShaderNodeMath');alpha.operation='MULTIPLY';ll.new(attr.outputs['Alpha'],alpha.inputs[0]);ll.new(remap.outputs['Result'],alpha.inputs[1]);ll.new(alpha.outputs[0],p.inputs['Alpha'])
            p.inputs['Specular IOR Level'].default_value=.10
            m['v16_visual_revision']='Review 2: dark packed dirt tint with irregular 3.5–38% maximum coverage; no pale solid spots.'
            m['v16_maximum_alpha']=.38
        mats.append(m)
    return mats

def _keep_nonfence(scene,collection,layout):
    """Hide complete originals; copied mesh retains ONLY unchanged non-fence islands."""
    band=unary_union([LineString(line).buffer(.145,cap_style=2) for y in layout['yards'] for line in y['fence_lines']])
    # Round corner caps cover fence post corners and gate endposts.
    band=unary_union([band]+[Point(p).buffer(.15) for y in layout['yards'] for l in y['fence_lines'] for p in l])
    mailbox_centres=[]
    for y in layout['yards']:
        a,b=np.asarray(y['front_edge'][0]),np.asarray(y['front_edge'][-1]);t=(b-a)/np.linalg.norm(b-a)
        mailbox_centres.append(np.asarray(y['pedestrian_gate_xy'])+t*.57)
    reports=[];retained=[]
    for suffix in ('hardware','grey','green','concrete','cream'):
        name='V12_Yard_Houses_'+suffix;ob=scene.objects.get(name)
        if ob is None:continue
        me=ob.data;parents=list(range(len(me.vertices)))
        def find(i):
            while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
            return i
        for edge in me.edges:
            a,b=map(int,edge.vertices);ra,rb=find(a),find(b)
            if ra!=rb:parents[rb]=ra
        groups={}
        for i in range(len(me.vertices)):groups.setdefault(find(i),[]).append(i)
        remove=[];kept=[]
        world=[ob.matrix_world@v.co for v in me.vertices]
        for ids in groups.values():
            hull=MultiPoint([tuple(world[i])[:2] for i in ids]).convex_hull
            xyz=np.asarray([world[i] for i in ids]);span=xyz[:,2].max()-xyz[:,2].min();xymean=xyz[:,:2].mean(axis=0)
            # Mailboxes live ON the fence corridor too. Preserve body and lid
            # using their documented v12 centre and dimensions; black slots are
            # a separate original object which is never hidden by this module.
            mailbox=any(np.linalg.norm(xymean-c)<.018 for c in mailbox_centres) and (
                (suffix in ('green','grey') and abs(span-.34)<.0002) or
                (suffix=='hardware' and abs(span-.025)<.0002))
            if band.covers(hull) and not mailbox:remove.extend(ids)
            else:kept.extend(ids)
        if not remove:continue
        original_visible={'hide_render':bool(ob.hide_render),'hide_viewport':bool(ob.hide_viewport)}
        ob.hide_render=True;ob.hide_viewport=True
        clone=None
        if kept:
            clone=ob.copy();clone.data=ob.data.copy();clone.name='V16_YardRetained_'+suffix;collection.objects.link(clone);clone.hide_render=False;clone.hide_viewport=False
            # The vertex index attribute survives deletion and certifies provenance.
            a=clone.data.attributes.new('v16_source_vertex',type='INT',domain='POINT');a.data.foreach_set('value',np.arange(len(me.vertices),dtype=np.int32))
            a=clone.data.attributes.new('v16_source_polygon',type='INT',domain='FACE');a.data.foreach_set('value',np.arange(len(me.polygons),dtype=np.int32))
            bm=bmesh.new();bm.from_mesh(clone.data);bm.verts.ensure_lookup_table();bmesh.ops.delete(bm,geom=[bm.verts[i] for i in remove],context='VERTS');bm.to_mesh(clone.data);bm.free();clone.data.update()
            clone['v16_role']='inherited_subset';clone['v16_source_object']=name;clone['v16_source_mesh']=me.name;clone['v16_retained_policy']='Identical non-fence connected islands, source vertex and polygon indices retained in named mesh attributes.'
            retained.append(clone)
        reports.append({'source_object':name,'source_mesh':me.name,'source_visibility':original_visible,'hidden_render':True,'hidden_viewport':True,'source_mesh_unchanged':True,'source_vertices':len(me.vertices),'removed_fence_vertices':len(remove),'retained_vertices':len(kept),'retained_copy':clone.name if clone else None,'retained_source_vertex_attribute':'v16_source_vertex' if clone else None})
    return reports,retained

def _fence(batch,ground,aa,bb,style,height,rng,gate=False):
    a,b=Vector((*aa,0)),Vector((*bb,0));u=(b-a).normalized();n=Vector((-u.y,u.x,0));L=(b-a).length
    if L<.12:return
    count=max(1,math.ceil(L/2.3));base=[ground.hit((a+u*L*j/count)[:2]) for j in range(count+1)]
    for p in base:
        # Centre roots are explicit support markers inside solid posts.
        batch.box(6 if style==1 else 0,p+UP*(height/2-.012),(u,n,UP),(.085,.085,height+.024))
    color=[1,4,3,2,8][style]
    for k,(p,q) in enumerate(zip(base,base[1:])):
        for z in (.24,height-.13):batch.beam(0 if style!=1 else 5,p+UP*z,q+UP*z,.033,.045)
        length=(q-p).length
        if style in (0,3):
            subdivisions=max(4,math.ceil(length/.075));vv=[]
            for j in range(subdivisions+1):
                c=p.lerp(q,j/subdivisions)+n*(.006 if j%2 else -.006)
                vv.extend([c+UP*.045,c+UP*(height-.02)])
            batch.poly(color,vv,[(2*j,2*j+2,2*j+3,2*j+1) for j in range(subdivisions)])
            # Panel overlap seam and bolts: purposeful details, no random rubbish.
            batch.beam(7,p+UP*.06+n*.013,p+UP*(height-.03)+n*.013,.005,.008)
            for z in (.24,height-.13):batch.cylinder(0,p+n*.018+UP*z,.011,.012,6)
        elif style in (1,4):
            boards=max(1,int(length/.18));w=min(.12,length/boards*.72)
            for j in range(boards):
                p1=p.lerp(q,(j+.5)/boards);hh=height-rng.uniform(.015,.055)
                # Rectangular old timber with softened irregular tops; open gaps.
                batch.box(color,p1+UP*(hh/2+.035),(u,n,UP),(w,.024,hh))
        else:
            # A wire mesh panel, open diamond network instead of repeated iron bars.
            rect=Polygon([(0,.10),(length,.10),(length,height-.06),(0,height-.06)])
            for sign in (-1,1):
                for offset in np.arange(-height,length+height,.28):
                    cut=LineString([(offset,-.1),(offset+sign*(height+.3),height+.2)]).intersection(rect)
                    if cut.geom_type!='LineString' or cut.length<.04:continue
                    (x0,z0),(x1,z1)=list(cut.coords)[0],list(cut.coords)[-1]
                    c0=p.lerp(q,x0/length)+UP*z0;c1=p.lerp(q,x1/length)+UP*z1
                    axis=(c1-c0).normalized();side=axis.cross(n).normalized()*.0022
                    batch.poly(3,[c0-side,c0+side,c1+side,c1-side],[(0,1,2,3)])
        if gate:
            batch.beam(0,p+UP*.18,q+UP*(height-.16),.032,.034)
            batch.beam(0,q+UP*.72+n*.045,q-u*.15+UP*.72+n*.045,.024,.025)

def _overlay(batch,ground,poly,material,lift=.007,fade=.18):
    """Clip to upper native triangle planes, with denser fade ring geometry."""
    made=0
    inside=poly.buffer(-fade) if fade else poly
    if inside.is_empty:inside=poly.buffer(-fade*.3)
    bands=[inside,poly.difference(inside)] if not inside.is_empty and fade else [poly]
    for band in bands:
        for idx in ground.tree.query(band,predicate='intersects'):
            plane=ground.planes[idx]
            for part in _parts(band.intersection(ground.polys[idx])):
                for tri in shapely.get_parts(shapely.constrained_delaunay_triangles(part)):
                    coords=list(tri.exterior.coords)[:-1]
                    if not tri.exterior.is_ccw:coords.reverse()
                    c=tri.centroid
                    if abs(ground.hit((c.x,c.y)).z-(c.x*plane[0]+c.y*plane[1]+plane[2]))>.003:continue
                    vv=[(x,y,float(x*plane[0]+y*plane[1]+plane[2]+lift)) for x,y in coords]
                    if max(abs(ground.hit((x,y)).z-(v[2]-lift)) for (x,y),v in zip(coords,vv))>.003:continue
                    alpha=[min(1.,poly.boundary.distance(Point(x,y))/fade) if fade else 1 for x,y in coords]
                    batch.poly(material,vv,[(0,1,2)],alpha=alpha,supports=[0,1,2]);made+=1
    return made

def _rect(c,u,n,w,l):return Polygon([tuple((c+u*x+n*y)[:2]) for x,y in [(-w/2,-l/2),(w/2,-l/2),(w/2,l/2),(-w/2,l/2)]])

def apply(scene,collection,source_data=None):
    if scene.get('v16_yards_applied'):raise RuntimeError('Rebuild yards from immutable v15 scene.')
    source=source_data or {};layout=_load(source,'yard_layout_v12.json');track=_load(source,'track_plan_v08.json');side=_load(source,'side_road_plan_v09.json');pit=_load(source,'pit_plan_v08.json')
    roads=unary_union([shape(track['outer_polygon']),shape(side['outer_polygon']),shape(pit['surface_polygon']),shape(pit['garage_polygon'])])
    houses=unary_union([shape(f['geometry']) for f in _load(source,'surroundings_local.json')['features'] if f['properties'].get('kind')=='building'])
    plots=unary_union([shape(y['polygon']) for y in layout['yards']]);ground=Ground(scene,plots.buffer(1));mats=_materials()
    fences=Batch('V16_Yard_Fences_Gates',mats,'yard_detail');surfaces=Batch('V16_Yard_Ground_Wear_Beds',mats,'yard_detail');props=Batch('V16_Yard_Domestic_Props',mats,'yard_detail')
    hidden,retained=_keep_nonfence(scene,collection,layout)
    placements=[];yard_records=[];counts={'fence_sections':0,'gate_leaves':0,'garden_beds':0,'service_paths':0,'ground_wear_patches':0,'wood_stacks':0,'water_barrels':0,'garden_supports':0}
    # Exclude native trunks and existing small decorative props in addition to map constraints.
    prop_reserved=[]
    for ob in scene.objects:
        if ob.type!='MESH' or ob.hide_render:continue
        if 'tree' in ob.name.lower() and ob.location.length>0:
            p=Point(ob.matrix_world.translation.x,ob.matrix_world.translation.y)
            if p.intersects(plots):prop_reserved.append(p.buffer(.75))
    trees=unary_union(prop_reserved)
    def record(kind,poly,ident,extra=None):
        if poly.intersection(roads).area>1e-7 or poly.intersection(houses).area>1e-7:raise AssertionError(('New yard detail intersects mapped surface',kind,ident))
        rec={'kind':kind,'mapped_house_reference':ident,'footprint':mapping(poly),'minimum_asphalt_clearance_m':float(poly.distance(roads)),'minimum_building_clearance_m':float(poly.distance(houses))}
        if extra:rec.update(extra)
        placements.append(rec)
    for index,y in enumerate(layout['yards']):
        rng=random.Random(SEED+int(y['osm_id'].split('/')[-1]));ident=y['osm_id'];plot=shape(y['polygon']);house=shape(y['house_polygon']);front=LineString(y['front_edge'])
        u=Vector((*y['front_tangent'],0));n=Vector((*y['front_normal'],0));center=Vector((house.centroid.x,house.centroid.y,0));style=rng.choices([0,1,2,3,4],[.27,.23,.12,.17,.21])[0];height=rng.uniform(1.16,1.43)
        yr={'osm_id':ident,'station_map_m':y['station_map_m'],'front_fence_style':['sage_corrugated','silver_wood','wire_mesh','blue_corrugated','painted_pickets'][style],'front_height_m':height,'new_details':[]}
        for aa,bb in y['fence_lines']:
            ln=LineString([aa,bb]);isfront=ln.distance(front)<.02 and ln.intersection(front.buffer(.03)).length>ln.length*.85
            parts=[ln]
            if isfront:
                cuts=Point(y['entrance_xy']).buffer(1.66)
                if not y['combined_gate']:cuts=unary_union([cuts,Point(y['pedestrian_gate_xy']).buffer(.51)])
                parts=[p for p in shapely.get_parts(ln.difference(cuts)) if p.geom_type=='LineString']
            rear_style=2 if rng.random()<.57 else (1 if style in (1,4) else 0)
            for part in parts:
                if part.length<.13:continue
                _fence(fences,ground,part.coords[0],part.coords[-1],style if isfront else rear_style,height if isfront else rng.uniform(.99,1.20),rng)
                counts['fence_sections']+=1
        aa,bb=np.asarray(front.coords[0]),np.asarray(front.coords[-1]);tu=(bb-aa)/np.linalg.norm(bb-aa)
        specs=[(np.asarray(y['entrance_xy']),3.24)]
        if not y['combined_gate']:specs.append((np.asarray(y['pedestrian_gate_xy']),.94))
        for c,w in specs:
            leaves=[(c-tu*w/2,c-tu*.025),(c+tu*.025,c+tu*w/2)] if w>1 else [(c-tu*w/2,c+tu*w/2)]
            for a,b in leaves:_fence(fences,ground,a,b,style,height-.03,rng,True);counts['gate_leaves']+=1
        paving=unary_union([shape(p) for key in ('paths','driveways','pads','beds') for p in y[key]])
        free=plot.buffer(-.42).difference(houses.buffer(.83)).difference(roads.buffer(.18)).difference(paving.buffer(.12)).difference(trees)
        used=[]
        # A worn service strip follows the long side of the house, never cuts through it.
        ring=[Vector((*q[:2],0)) for q in list(house.exterior.coords)[:-1]];us=[(p-center).dot(u) for p in ring];ns=[(p-center).dot(n) for p in ring]
        side_sign=-1 if index%2 else 1
        sidepos=(min(us)-1.25) if side_sign<0 else (max(us)+1.25)
        pa=center+u*sidepos+n*(min(ns)-2.0);pb=center+u*sidepos+n*(max(ns)+4)
        service=LineString([pa[:2],pb[:2]]).buffer(.56,cap_style=1).intersection(free)
        for p in _parts(service):
            if p.area<1.2:continue
            if _overlay(surfaces,ground,p,11,fade=.22):record('worn_service_path',p,ident);used.append(p.buffer(.18));counts['service_paths']+=1;yr['new_details'].append('service_path')
        # Rear vegetables and modest bare earth occupy actual remaining spaces.
        candidates=[]
        for uu in (min(us)-2.8,max(us)+2.8,0):
            for nn in (min(ns)-3.8,max(ns)+3.4):candidates.append(center+u*uu+n*nn)
        rng.shuffle(candidates)
        usable=free.difference(unary_union(used));made=0
        for c in candidates:
            bed=_rect(c,u,n,rng.uniform(2.0,3.1),rng.uniform(2.5,4.0))
            if made>=1+(index%4==0) or not usable.covers(bed.buffer(.14)):continue
            if not _overlay(surfaces,ground,bed,10,fade=.13):continue
            record('vegetable_bed',bed,ident);used.append(bed.buffer(.30));counts['garden_beds']+=1;made+=1;yr['new_details'].append('vegetable_bed')
            # Low physical furrows break flat dark rectangles, local soil not foliage.
            bounds=[(Vector((*p[:2],0))-c).dot(u) for p in list(bed.exterior.coords)[:-1]]
            for x in np.arange(min(bounds)+.20,max(bounds)-.1,.40):
                ln=LineString([(c+u*x-n*5)[:2],(c+u*x+n*5)[:2]]).intersection(bed.buffer(-.18))
                if ln.geom_type!='LineString':continue
                for part in _parts(ln.buffer(.045,cap_style=2)):_overlay(surfaces,ground,part,11,lift=.018,fade=.035)
            usable=usable.difference(bed.buffer(.30))
        # A few larger irregular low-contrast grass wear patches, not a tiled pattern.
        for j in range(3):
            c=usable.representative_point() if j==0 else Point(rng.uniform(*plot.bounds[::2]),rng.uniform(*plot.bounds[1::2]))
            major=rng.uniform(1.4,2.7);minor=rng.uniform(.75,1.5)
            xy=[]
            for k in range(17):
                ang=k*math.tau/17;rr=rng.uniform(.83,1.12);q=Vector((c.x,c.y,0))+u*(math.cos(ang)*major*rr)+n*(math.sin(ang)*minor*rr);xy.append(q[:2])
            patch=Polygon(xy).intersection(usable)
            for p in _parts(patch):
                if p.area<1.4:continue
                if _overlay(surfaces,ground,p,12,fade=.30):record('uneven_grass_wear',p,ident);counts['ground_wear_patches']+=1
        # Daily-use objects: one selected utility cluster in some yards.
        if index%3!=0:
            for c in candidates:
                reserve=_rect(c,u,n,1.7,1.25)
                if not usable.covers(reserve.buffer(.20)):continue
                p=ground.hit(c[:2]);kind='water_barrel' if index%2 else 'wood_stack'
                if kind=='water_barrel':
                    props.cylinder(9,p+UP*.005,.285,.84,20)
                    for h in (.15,.66,.84):props.cylinder(0,p+UP*h,.294,.018,20)
                    props.cylinder(3,p+UP*.862,.08,.008,12)
                    counts['water_barrels']+=1
                else:
                    # Eight sawn timber lengths with visible gap/shadow structure.
                    for row in range(3):
                        for j in range(4-row%2):
                            q=p+u*((j-1.5)*.26)+n*(rng.uniform(-.05,.05))+UP*(.10+row*.18)
                            props.box(5 if (j+row)%3 else 4,q,(u,n,UP),(.22,rng.uniform(.85,1.05),.15))
                    props.box(3,p+UP*.62,(u,n,UP),(1.18,1.16,.035));counts['wood_stacks']+=1
                record(kind,reserve,ident,{'root_xyz':list(p),'root_surface':'native terrain'});used.append(reserve);yr['new_details'].append(kind);break
        yard_records.append(yr)
    objects=[o for b in (fences,surfaces,props) if (o:=b.finish(collection))]
    # Object-wide footprints derived from actual faces, not huge convex hulls.
    new=[]
    for ob,batch in zip(objects,[b for b in (fences,surfaces,props) if b.f]):
        polys=[]
        for f in ob.data.polygons:
            p=Polygon([tuple(ob.data.vertices[i].co)[:2] for i in f.vertices])
            if p.is_valid and p.area>1e-8:polys.append(p)
        fp=unary_union(polys);road_overlap=fp.intersection(roads).area;house_overlap=fp.intersection(houses).area
        if road_overlap>1e-6 or house_overlap>1e-6:raise AssertionError((ob.name,'Footprint collision',road_overlap,house_overlap))
        rec={'object':ob.name,'role':'yard_detail','minimum_asphalt_clearance_m':float(fp.distance(roads)),'minimum_building_clearance_m':float(fp.distance(houses)),'footprint':mapping(fp),'triangles':len(ob.data.loop_triangles)}
        if batch.supports:rec.update(support_vertex_indices=batch.supports,reference_surface_names=ground.names,min_height_offset_m=.003,max_height_offset_m=.023)
        new.append(rec)
    triangles=sum(len(o.data.loop_triangles) for o in objects)
    if triangles>LIMIT_TRIS:raise AssertionError(('Yards triangle budget exceeded',triangles))
    inherited=[{'object':o.name,'source_object':o['v16_source_object'],'source_vertex_attribute':'v16_source_vertex','source_polygon_indices':[int(v.value) for v in o.data.attributes['v16_source_polygon'].data],'role':'inherited_subset','source_modifiers_and_materials_unchanged':True} for o in retained]
    report={'version':'v16','module':'yards_v16.py','seed':SEED,'sector_original_map_chainage_m':[3800,4250],'counts':dict(counts,yards=len(layout['yards']),objects=len(objects),retained_objects=len(retained),triangles=triangles),'triangle_budget':LIMIT_TRIS,'new_objects':new+inherited,'replacement_sources':hidden,'retained_copies':inherited,'native_ground_objects':ground.names,'ground_ray_misses':ground.ray_misses,'placements':placements,'reserved_detail_footprints':mapping(unary_union([shape(p['footprint']) for p in placements if p['kind']!='uneven_grass_wear'])),'yards':yard_records,'limits':['Plot boundaries and all new objects are plausible regional art, not measured parcel/address evidence.','Native procedural/alpha materials still require explicit target AC shader translation.','No modification to mapped building vertices, native terrain, racing route, road/pit surfaces or kerbs.']}
    scene['v16_yards_applied']=True
    (ROOT/'data').mkdir(exist_ok=True);(ROOT/'data/yards_v16.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
