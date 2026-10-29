"""Driver-visible house individuality on twelve unchanged map envelopes.

Regional reconstruction, not address photographs. All original mesh data,
coordinates, UVs, normals and polygon material indices remain immutable.
Only private material copies/rebindings and new per-house attachments are used.
"""
from pathlib import Path
import json, math, random
import bpy
import numpy as np
from mathutils import Vector
import shapely
from shapely.geometry import Polygon, MultiPoint, box
from shapely.ops import unary_union

ROOT=Path(__file__).resolve().parent
UP=Vector((0,0,1))
TARGETS={
 '403554197':dict(wall=(1.00,.92,.80),trim='cream',gable=None,canopy=False,roof=(1.02,.95,.86)),
 '403555594':dict(wall=(.81,.91,.94),trim='sage',gable='sage',canopy=True,roof=(1.28,.84,.69)),
 '403555593':dict(wall=(.97,.93,.82),trim='cream',gable='timber',canopy=True,roof=(.91,.93,.92)),
 '406501364':dict(wall=(.95,.96,.92),trim='blue',gable=None,canopy=False,roof=(.93,1.02,1.04)),
 '403555592':dict(wall=(.90,.91,.81),trim='cream',gable='cream',canopy=False,roof=(.90,.94,.92)),
 '403555596':dict(wall=(.90,.94,.97),trim='blue',gable=None,canopy=True,roof=(.99,.95,.88)),
 '403555595':dict(wall=(1.04,.92,.79),trim='timber',gable=None,canopy=False,roof=(1.06,.93,.86)),
 '403555589':dict(wall=(.86,.97,.92),trim='cream',gable=None,canopy=False,roof=(.94,.99,.96)),
 '403555591':dict(wall=(.92,.94,.85),trim='sage',gable=None,canopy=True,roof=(1.02,.92,.81)),
 '403555588':dict(wall=(.99,.97,.88),trim='cream',gable='sage',canopy=True,roof=(.90,.99,.96)),
 '403555590':dict(wall=(.93,.95,1.02),trim='blue',gable='blue',canopy=False,roof=(.88,.95,1.02)),
 '403555609':dict(wall=(1.00,.93,.84),trim='cream',gable=None,canopy=False,roof=(1.02,.98,.94)),
}
COLORS={'cream':(.52,.50,.41),'sage':(.18,.29,.24),'blue':(.16,.24,.27),'timber':(.23,.16,.105),
 'metal':(.16,.20,.20),'plaster':(.43,.42,.36),'linen':(.70,.66,.55),'lace':(.76,.74,.66),'slate':(.16,.20,.19)}

def world(ob):
    a=np.empty((len(ob.data.vertices),3),np.float32);ob.data.vertices.foreach_get('co',a.ravel());m=np.asarray(ob.matrix_world,dtype=np.float64)
    return a.astype(np.float64)@m[:3,:3].T+m[:3,3]

def masks(scene,records):
    region=unary_union([Polygon(r['ring']).buffer(1.5) for r in records]);roads=[];names=[];houses={}
    for ob in scene.objects:
        if ob.type!='MESH':continue
        road=ob.name in ('1ROAD_MAIN','1PIT_SURFACE_V08','ROAD_SHOULDERS','SIDE_ROADS','SIDE_ROADS_V09','V09_Side_road_verges')
        building=ob.name.endswith('_walls') and ob.name.startswith(('BLDG_','V10_Industrial_Mapped_'))
        if not (road or building):continue
        vv=world(ob)
        if not len(vv):continue
        if not region.intersects(box(*vv[:,:2].min(0),*vv[:,:2].max(0))):continue
        if building:houses[ob.name]=MultiPoint(vv[:,:2]).convex_hull;continue
        ob.data.calc_loop_triangles();ix=np.asarray([tuple(t.vertices) for t in ob.data.loop_triangles],dtype=np.int32)
        if not len(ix):continue
        pp=shapely.polygons(vv[ix,:2]);keep=(shapely.area(pp)>1e-10)&shapely.intersects(region,pp)
        if np.any(keep):roads.append(shapely.union_all(pp[keep]));names.append(ob.name)
    return unary_union(roads),houses,names

class Batch:
    def __init__(self,r,mats,road,houses):
        self.r=r;self.mats=mats;self.v=[];self.f=[];self.mi=[];self.uv=[];self.omissions=[];self.tested=0
        self.allowed=Polygon(r['ring']).buffer(1.2)
        self.forbidden=unary_union([road]+[h for name,h in houses.items() if name!=r['wall_name']]);self.forbidden_inner=self.forbidden.buffer(-.002)
    def poly(self,key,vs,fs,role):
        self.tested+=1;vv=np.asarray([tuple(v) for v in vs],dtype=np.float32).astype(np.float64);h=MultiPoint(vv[:,:2]).convex_hull
        overlap=h.intersection(self.forbidden).area
        linear_intrusion=h.area<1e-10 and self.forbidden_inner.intersects(h)
        if overlap>1e-9 or linear_intrusion or not self.allowed.buffer(.0002).covers(h):
            self.omissions.append({'role':role,'overlap_m2':overlap,'vertices':len(vv)});return False
        off=len(self.v);self.v.extend(map(tuple,vv));self.f.extend([tuple(off+i for i in f) for f in fs]);self.mi.extend([list(self.mats).index(key)]*len(fs));return True
    def box(self,key,c,axes,size,role):
        a,b,d=axes;c=Vector(c);vs=[c+a*x*size[0]/2+b*y*size[1]/2+d*z*size[2]/2 for x,y,z in [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1)]]
        return self.poly(key,vs,[(0,3,2,1),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7),(4,5,6,7)],role)
    def beam(self,key,a,b,w,h,role):
        a,b=Vector(a),Vector(b);u=(b-a).normalized();n=u.cross(UP)
        if n.length<.01:n=Vector((1,0,0))
        n.normalize();z=u.cross(n).normalized();return self.box(key,(a+b)*.5,(u,n,z),((b-a).length,w,h),role)
    def finish(self,col):
        if not self.f:return None
        name='V17_Architecture_'+self.r['osm_id'].replace('/','_');me=bpy.data.meshes.new(name+'_mesh');me.from_pydata(self.v,[],self.f)
        for m in self.mats.values():me.materials.append(m)
        for p,i in zip(me.polygons,self.mi):p.material_index=i
        uv=me.uv_layers.new(name='V17_MetricDetail')
        for p in me.polygons:
            n=p.normal;u=Vector((-n.y,n.x,0));u=Vector((1,0,0)) if u.length<.001 else u.normalized()
            for li in p.loop_indices:
                v=me.vertices[me.loops[li].vertex_index].co;uv.data[li].uv=(v.dot(u),v.z)
        me.update();me.calc_loop_triangles();ob=bpy.data.objects.new(name,me);col.objects.link(ob)
        ob['v17_role']='attached_detail';ob['associated_osm_id']=self.r['osm_id'];ob['accuracy']='Inferred regional house detail; unchanged original mapped envelope.'
        return ob

def material(key):
    c=COLORS[key];m=bpy.data.materials.new('V17_Architecture_'+key);m.use_nodes=True;m.diffuse_color=(*c,1)
    nn,ll=m.node_tree.nodes,m.node_tree.links;p=nn.get('Principled BSDF');p.inputs['Roughness'].default_value=.82;p.inputs['Specular IOR Level'].default_value=.25
    geo=nn.new('ShaderNodeNewGeometry');vec=geo.outputs['Position']
    stretch=nn.new('ShaderNodeVectorMath');stretch.operation='MULTIPLY';stretch.inputs[1].default_value=(1.2,1.2,32) if key not in ('linen','lace') else (35,35,1.4);ll.new(vec,stretch.inputs[0])
    noise=nn.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=1.6;noise.inputs['Detail'].default_value=2;ll.new(stretch.outputs[0],noise.inputs[0])
    ramp=nn.new('ShaderNodeValToRGB');ramp.color_ramp.elements[0].color=(*[v*.73 for v in c],1);ramp.color_ramp.elements[1].color=(*[v*1.12 for v in c],1);ll.new(noise.outputs['Fac'],ramp.inputs[0]);ll.new(ramp.outputs[0],p.inputs['Base Color'])
    bump=nn.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.25;bump.inputs['Distance'].default_value=.0008;ll.new(noise.outputs['Fac'],bump.inputs['Height']);ll.new(bump.outputs[0],p.inputs['Normal'])
    if key in ('metal','slate'):p.inputs['Metallic'].default_value=.32;p.inputs['Roughness'].default_value=.61
    if key in ('linen','lace'):p.inputs['Sheen Weight'].default_value=.20;p.inputs['Roughness'].default_value=.95
    m['accuracy']='Original authored regional reconstruction material, not a photograph of an address';return m

def private_tint(old,ident,tint,base=None):
    m=old.copy();m.name='V17_'+ident+'_'+old.name;n,l=m.node_tree.nodes,m.node_tree.links;p=next((x for x in n if x.type=='BSDF_PRINCIPLED'),None)
    if p is None:return m
    source=p.inputs['Base Color'].links[0].from_socket if p.inputs['Base Color'].links else None
    mix=n.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1;mix.inputs[2].default_value=(*tint,1);mix.label='V17 scoped house palette'
    if source:l.new(source,mix.inputs[1])
    else:mix.inputs[1].default_value=p.inputs['Base Color'].default_value
    color=mix.outputs[0]
    if base is not None:
        # Water/weathering anchored to physical ground height, strongest on the
        # plinth and tapering over first 65cm rather than evenly painted noise.
        geo=n.new('ShaderNodeNewGeometry');sep=n.new('ShaderNodeSeparateXYZ');l.new(geo.outputs['Position'],sep.inputs[0]);r=n.new('ShaderNodeMapRange');r.clamp=True;r.inputs['From Min'].default_value=base+.22;r.inputs['From Max'].default_value=base+.90;r.inputs['To Min'].default_value=.23;r.inputs['To Max'].default_value=0;l.new(sep.outputs['Z'],r.inputs['Value'])
        fade=n.new('ShaderNodeMixRGB');fade.blend_type='MULTIPLY';fade.inputs[2].default_value=(.40,.43,.32,1);l.new(r.outputs['Result'],fade.inputs[0]);l.new(color,fade.inputs[1]);color=fade.outputs[0]
    l.new(color,p.inputs['Base Color']);m['v17_source_material']=old.name;m['v17_policy']='Private copy; original shader graph retained with scoped finish multiplier and low-wall dampness.';return m

def rebind(ob,index,new,report):
    old=ob.material_slots[index].material;oldlink=ob.material_slots[index].link
    ob.material_slots[index].material=new
    report.append({'object':ob.name,'slot':index,'old_material':old.name,'new_material':new.name,'old_link':oldlink,'new_link':ob.material_slots[index].link})

def apply(scene,collection,source_data=None):
    source_data=source_data or {};sd=Path(source_data.get('source_data_dir',ROOT.parent/'chmyrivka_v16/source_data'));od=Path(source_data.get('output_data_dir',ROOT/'data'));od.mkdir(parents=True,exist_ok=True)
    records=[r for r in json.loads((sd/'architecture_placement_v12.json').read_text()) if r['osm_id'].split('/')[-1] in TARGETS]
    assert len(records)==12
    roads,houses,roadnames=masks(scene,records);mats={k:material(k) for k in COLORS};bindings=[];rows=[];objects=[];canopy_reserves=[];totals=dict(houses=12,canopies=0,gable_clad_frontages=0,window_surrounds=0,curtain_windows=0,roof_palette_copies=0)
    for r in records:
        ident=r['osm_id'].split('/')[-1];spec=TARGETS[ident];rng=random.Random(int(ident)+1700);b=Batch(r,mats,roads,houses);base,eave,peak=r['base_z'],r['eave_z'],r['peak_z'];counts=dict(canopies=0,gable_clad_frontages=0,window_surrounds=0,curtain_windows=0)
        wall=scene.objects[r['wall_name']]
        for i,s in enumerate(wall.material_slots):
            if s.material:rebind(wall,i,private_tint(s.material,ident,spec['wall'],base),bindings)
        roofcopies={}
        for name in (r['roof_name'],'V12_Roof_way_'+ident):
            ob=scene.objects.get(name)
            if not ob:continue
            for i,s in enumerate(ob.material_slots):
                if not s.material:continue
                old=s.material
                if old.name not in roofcopies:roofcopies[old.name]=private_tint(old,ident,spec['roof'])
                rebind(ob,i,roofcopies[old.name],bindings)
        totals['roof_palette_copies']+=len(roofcopies)
        # Existing v13/v15 per-house material slots already isolate original sash
        # geometry. Rebind those slots only; no polygon remapping or shared edits.
        for ob in scene.objects:
            if ob.type!='MESH' or not ob.name.startswith('V12_Architecture_frame_'):continue
            for i,s in enumerate(ob.material_slots):
                if s.material and ('way_'+ident+'_') in s.material.name:
                    old=s.material;mc=old.copy();mc.name='V17_Frame_'+ident+'_'+old.name
                    pp=next((x for x in mc.node_tree.nodes if x.type=='BSDF_PRINCIPLED'),None)
                    if pp:
                        for lk in list(pp.inputs['Base Color'].links):mc.node_tree.links.remove(lk)
                        pp.inputs['Base Color'].default_value=(*COLORS[spec['trim']],1);pp.inputs['Roughness'].default_value=.78
                    rebind(ob,i,mc,bindings)
        for ei,e in enumerate(r['edges']):
            if ei!=r['front_edge_index'] and e['visibility']<-.35:continue
            a,u,n=Vector(e['a']),Vector(e['u']),Vector(e['n']);L=e['length'];front=ei==r['front_edge_index']
            def point(x,z,dep=0):return a+u*x+n*dep+UP*z
            for oi,o in enumerate(e['openings']):
                x0,x1,z0,z1=o['x0'],o['x1'],o['z0'],o['z1'];cx=(x0+x1)/2;cz=(z0+z1)/2;w=x1-x0;h=z1-z0
                if o['isdoor']:
                    if front and spec['canopy']:
                        # Pitched metal canopy over existing shallow flat cover;
                        # original remains as soffit/structure. No support posts.
                        wid=w+.78;dep=.97;back=z1+.50;tip=z1+.30
                        q=[point(cx-wid/2,back,-.015),point(cx+wid/2,back,-.015),point(cx+wid/2,tip,dep),point(cx-wid/2,tip,dep)]
                        before=len(b.f);b.poly('slate',q+[(v-UP*.035) for v in q],[(0,1,2,3),(7,6,5,4),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)],'entry_canopy')
                        if len(b.f)>before:
                            counts['canopies']+=1
                            canopy_reserves.append({'osm_id':r['osm_id'],'polygon_xy':[[float(v.x),float(v.y)] for v in q],'ground_posts':False})
                            for sign in (-1,1):
                                xx=cx+sign*(wid/2-.10);b.beam('metal',point(xx,z1-.18,.035),point(xx,tip-.06,dep-.08),.035,.04,'canopy_bracket')
                            b.beam(spec['trim'],q[3]-UP*.05,q[2]-UP*.05,.07,.10,'canopy_fascia')
                            for j in range(1,6):
                                t=j/6;b.beam('metal',q[0].lerp(q[1],t)+UP*.022,q[3].lerp(q[2],t)+UP*.022,.015,.014,'canopy_sheet_seam')
                    continue
                # Wide painted architraves create readable facade rhythm; three
                # variants, applied around the original opening rather than over it.
                trim=spec['trim'];wide=.105 if ident in ('403555588','403555594','403555595') else .075
                before=len(b.f)
                for xx in (x0-wide/2-.015,x1+wide/2+.015):b.box(trim,point(xx,cz,.028),(u,n,UP),(wide,.062,h+.12),'window_architrave')
                b.box(trim,point(cx,z1+.055,.032),(u,n,UP),(w+wide*2+.03,.070,.11),'window_lintel')
                if front and ident in ('403555588','403555593','403555595'):
                    b.box('cream',point(cx,z1+.122,.042),(u,n,UP),(w+wide*2+.13,.11,.045),'window_lintel_cap')
                if len(b.f)>before:counts['window_surrounds']+=1
                # Place cloth behind the original -.145m glass and in front of
                # the old -.29m curtain/dark backing. Native glass untouched.
                # Asymmetric partially drawn curtains, no emissive component.
                if front or rng.random()<.65:
                    widths=(w*rng.uniform(.29,.42),w*rng.uniform(.23,.35));key='lace' if (int(ident)+oi)%3 else 'linen'
                    for lo,hi in ((x0+.08,x0+widths[0]),(x1-widths[1],x1-.08)):
                        steps=max(8,round((hi-lo)/.025));vs=[]
                        for j in range(steps+1):
                            x=lo+(hi-lo)*j/steps;dep=-.205+.014*math.sin(j*math.tau/4.5);sag=.025*math.sin(j/steps*math.pi)
                            vs.extend((point(x,z0+.09+sag,dep),point(x,z1-.07,dep)))
                        b.poly(key,vs,[(j*2,j*2+2,j*2+3,j*2+1) for j in range(steps)],'recessed_curtain')
                    if ident=='403555588' or rng.random()<.35:
                        b.box(key,point(cx,z1-.17,-.205),(u,n,UP),(w-.16,.022,.16),'curtain_valance')
                    counts['curtain_windows']+=1
            if front and spec['gable'] and peak-eave>.6:
                rv=[Vector(v) for v in r['roof_source']['vertices']];mid=point(L/2,eave);is_eave=any((mid-(rv[f[0]]+rv[f[1]])/2).length<.25 for f in r['roof_source']['faces'] if len(f)==4)
                if not is_eave:
                    # Preserve roof silhouette and central pre-existing attic vent.
                    g=Polygon([(.16,eave+.13),(L-.16,eave+.13),(L/2,peak-.10)])
                    vent=box(L/2-.255,eave+(peak-eave)*.39-.19,L/2+.255,eave+(peak-eave)*.39+.19)
                    g=g.difference(vent);before=len(b.f)
                    for z in np.arange(eave+.13,peak-.10,.14):
                        patch=g.intersection(box(0,z,L,z+.134))
                        for pp in shapely.get_parts(patch):
                            if pp.geom_type!='Polygon' or pp.area<.001:continue
                            coords=list(pp.exterior.coords)[:-1];vs=[point(x,zz,.052) for x,zz in coords];b.poly(spec['gable'],vs,[tuple(range(len(vs)))],'gable_board')
                    if len(b.f)>before:counts['gable_clad_frontages']+=1
        ob=b.finish(collection)
        if ob:
            row={'object':ob.name,'role':'attached_detail','associated_building':r['wall_name'],'associated_osm_id':r['osm_id'],'attachment_tolerance_m':1.2,'maximum_above_building_m':1.5,'minimum_below_building_m':.05,'minimum_asphalt_clearance_m':0.,'minimum_building_clearance_m':0.}
            objects.append(row)
        for key,val in counts.items():totals[key]+=val
        rows.append({'osm_id':r['osm_id'],'station_map_m':r['station_map_m'],'wall_object':r['wall_name'],'roof_object':r['roof_name'],'detail_object':ob.name if ob else None,'style':spec,'counts':counts,'tested_primitives':b.tested,'omitted_primitives':b.omissions,'polygons':len(ob.data.polygons) if ob else 0,'triangles':len(ob.data.loop_triangles) if ob else 0})
    report={'version':'v17','scope':'Twelve near-road houses around Sinna map3900–4070m, selected from established review cameras01–03','geometry_policy':'Original objects/meshes/transforms/UVs/normals and polygon material indices unchanged; explicit private material rebindings only','accuracy':'OSM envelopes retained; house treatments/canopies/cladding are regional reconstruction, not address-matched photographs','counts':totals,'canopy_reserves':canopy_reserves,'new_objects':objects,'buildings':rows,'material_rebindings':bindings,'guard':{'paved_objects':roadnames,'full_projected_primitives_checked':True,'other_buildings_excluded':True,'maximum_attachment_distance_m':1.2,'omitted_primitives':sum(len(x['omitted_primitives']) for x in rows)},'limitations':['Native Blender materials require deliberate baking/conversion for AC','No KN5 or game validation','Original openings and house silhouettes remain; gable board finish is an overlay inside existing roof envelope','Canopy/architrave additions are inferred, not actual surveyed facade details']}
    (od/'architecture_v17.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));return report
