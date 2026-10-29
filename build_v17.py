"""Assemble the v17 environment pass from immutable v16. Blender 4.5.7.

Run under flock /tmp/chmyrivka_v16_bpy.lock. All local art is explicitly
inferred; map footprints, route, terrain, pit entry and kerbs remain fixed.
"""
from pathlib import Path
import argparse, hashlib, json, sys, time
import bpy

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'chmyrivka_v16/chmyrivka_visual_v16.blend'
SOURCE_SHA='8e4f2c9c3e1b37de42d9d99edf115a898fa72cb76936dce19020bb02f23b0e00'
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'dependencies/v16'));sys.path.insert(2,str(ROOT/'dependencies/v15'))

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(path,value):
    path=ROOT/path;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def snapshot(scene):
    return {o.name:{'visibility':(o.hide_render,o.hide_viewport),
                   'materials':[s.material.name if s.material else None for s in o.material_slots],
                   'material_links':[s.link for s in o.material_slots]}
            for o in scene.objects}

def main(a):
    source=Path(a.source).resolve();assert sha(source)==SOURCE_SHA
    bpy.ops.wm.open_mainfile(filepath=str(source));scene=bpy.context.scene
    original=snapshot(scene);start=time.time()
    collection=bpy.data.collections.new('67_V17_Driver_Visible_Architecture')
    scene.collection.children.link(collection)
    source_data={'source_data_dir':str(ROOT/'source_data'),'output_data_dir':str(ROOT/'data')}
    report={'version':'v17','source_sha256':SOURCE_SHA,'assetto_corsa_tested':False,
            'source_native':str(source),'components':{}}
    modules=a.modules.split(',')
    for name in modules:
        module=__import__(name+'_v17');print('V17_BEGIN',name,flush=True)
        report['components'][name]=module.apply(scene,collection,source_data)
        source_data[name+'_report']=report['components'][name]
        print('V17_DONE',name,flush=True)
    after=snapshot(scene);visibility=[];rebindings=[];new=[]
    for name,old in original.items():
        assert name in after,'Original object deleted: '+name
        if old['visibility']!=after[name]['visibility']:
            visibility.append({'object':name,'hide_render':after[name]['visibility'][0],
                               'hide_viewport':after[name]['visibility'][1],
                               'reason':'Explicit replacement of inferred residential decoration; original mesh retained.'})
        if old['materials']!=after[name]['materials'] or old['material_links']!=after[name]['material_links']:
            rebindings.append({'object':name,'old_materials':old['materials'],
                               'new_materials':after[name]['materials'],
                               'old_links':old['material_links'],'new_links':after[name]['material_links'],
                               'reason':'Scoped private material copies for v17 environment/glazing.'})
    for ob in scene.objects:
        if ob.name in original or ob.type!='MESH':continue
        role=ob.get('v17_role','yard_detail')
        if role not in ('roadside','vegetation','yard_detail','attached_detail','inherited_subset'):
            role='vegetation' if 'Vegetation' in ob.name else 'yard_detail'
        new.append({'object':ob.name,'role':role,'minimum_asphalt_clearance_m':0.,'minimum_building_clearance_m':0.})
    contract={'source_sha256':SOURCE_SHA,'allowed_visibility_changes':visibility,
              'allowed_material_rebindings':rebindings,'new_objects':new,'ground_checks':[]}
    before_users={}
    for object_name,old in original.items():
        for index,material_name in enumerate(old['materials']):
            if material_name:
                before_users.setdefault(material_name,[]).append({'object':object_name,'slot_index':index})
    after_used={m for state in after.values() for m in state['materials'] if m}
    contract['allowed_material_retirements']=[]
    for material_name,users in sorted(before_users.items()):
        material=bpy.data.materials.get(material_name)
        if material_name not in after_used and material and material.users==0 and not material.use_fake_user:
            contract['allowed_material_retirements'].append({'material':material_name,
                'original_users':users,
                'reason':'All original slot users were explicitly rebound to private copies; unreferenced original datablock is omitted by Blender save.'})
    # Explicit metadata from component authors augments, never silently removes,
    # the complete automatic inventory. Final contract is inspected by root.
    for component in report['components'].values():
        for record in component.get('new_objects',[]):
            record=dict(record)
            record['object']=record.get('object',record.get('name'))
            entry=next((x for x in new if x['object']==record['object']),None)
            if entry is not None:entry.update(record)
            supports=record.get('support_vertex_indices',[])
            if supports:
                contract['ground_checks'].append({'object':record['object'],'vertex_indices':supports,
                    'surfaces':record['reference_surface_names'],
                    'minimum_offset_m':record.get('support_offset_min_m',record.get('min_height_offset_m',0)),
                    'maximum_offset_m':record.get('support_offset_max_m',record.get('max_height_offset_m',0))})
        contract['ground_checks'].extend(component.get('ground_checks',[]))
    # Component semantic category is mapped to the auditor's explicit roles.
    for record in contract['new_objects']:
        if record['role']=='frontage':
            record['role']='yard_detail' if record.get('kind')=='ground_contact' else 'vegetation'
    dump('data/change_contract_v17.json',contract)
    packed=[]
    for im in bpy.data.images:
        if im.source!='FILE':continue
        if not im.packed_file:
            if not im.has_data:im.reload()
            im.pack()
        assert im.packed_file,im.name
        packed.append({'name':im.name,'size':list(im.size)})
    dump('data/packed_images_v17.json',packed)
    scene.name='Chmyrivka_Visual_v17';scene['version']='v17'
    scene['assetto_corsa_tested']=False;scene['v17_source_sha256']=SOURCE_SHA
    about=bpy.data.texts.get('ABOUT_THIS_SCENE_RU.txt') or bpy.data.texts.new('ABOUT_THIS_SCENE_RU.txt')
    about.clear();about.write('ЧМИРІВКА / СТАРОБІЛЬСЬК — v17\nБлизкие фасады, входы и видимые с дороги палисадники.\nИсходник v16 сохранён. Картографические контуры, дорога, рельеф, пит-лейн и поребрики сохраняются.\nОтделка и дворовые детали — региональная реконструкция, не адресная фотограмметрия.\nЭто исходная Blender-сцена, не готовый мод Assetto Corsa. См. документацию выпуска.\n')
    bpy.context.view_layer.update()
    target=Path(a.output).resolve() if a.output else ROOT/'chmyrivka_visual_v17.blend'
    bpy.ops.wm.save_as_mainfile(filepath=str(target),compress=True)
    report.update({'native_sha256':sha(target),'new_meshes':len(new),'hidden_superseded':visibility,
                   'material_rebindings':rebindings,'seconds':round(time.time()-start,2),
                   'source_unchanged':sha(source)==SOURCE_SHA})
    dump('build_info_v17.json',report);print('V17_SAVED',target,report['native_sha256'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default=str(SOURCE));p.add_argument('--output')
    p.add_argument('--modules',default='architecture,frontage');main(p.parse_args())
