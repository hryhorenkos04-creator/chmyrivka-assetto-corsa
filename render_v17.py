"""Read-only v17 rendering with the original v15 review cameras."""
from pathlib import Path
import argparse, hashlib, json, time
import bpy

ROOT=Path(__file__).resolve().parent

def main(a):
    path=Path(a.scene).resolve();sha=hashlib.sha256(path.read_bytes()).hexdigest()
    bpy.ops.wm.open_mainfile(filepath=str(path));scene=bpy.context.scene
    # Camera-bound timeline markers otherwise override scene.camera at render.
    for marker in scene.timeline_markers:marker.camera=None
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.render.threads_mode='FIXED';scene.render.threads=8
    scene.cycles.samples=a.samples;scene.cycles.seed=0;scene.cycles.use_denoising=True
    scene.render.resolution_x=a.width;scene.render.resolution_y=round(a.width*9/16);scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    loaded=json.loads((ROOT/a.cameras).read_text());rows=loaded['cameras'] if isinstance(loaded,dict) else loaded
    selection=set(a.views.split(',')) if a.views else None
    out=ROOT/a.output;out.mkdir(parents=True,exist_ok=True)
    manifest=out/'render_manifest.json';done=json.loads(manifest.read_text()) if manifest.exists() else []
    for i,r in enumerate(rows,1):
        if selection and str(i) not in selection:continue
        camera=scene.objects.get(r['camera'])
        if camera is None:
            assert a.cameras!='data/review_cameras_v15.json','Original review camera missing'
            data=bpy.data.cameras.new(r['camera']);camera=bpy.data.objects.new(r['camera'],data)
            scene.collection.objects.link(camera);camera.location=r['location'];camera.rotation_euler=r['rotation'];data.lens=r['lens_mm']
            data.clip_start=.05;data.clip_end=10000
            r=dict(r,location=list(camera.location),rotation=list(camera.rotation_euler),lens_mm=data.lens)
        assert max(abs(x-y) for x,y in zip(camera.location,r['location']))<1e-6
        assert max(abs(x-y) for x,y in zip(camera.rotation_euler,r['rotation']))<1e-6
        assert abs(camera.data.lens-r['lens_mm'])<1e-6
        scene.camera=camera;filename=f'{i:02d}_{r["kind"]}.png';scene.render.filepath=str(out/filename)
        start=time.time();bpy.ops.render.render(write_still=True,scene=scene.name)
        assert scene.camera==camera
        row={**r,'file':filename,'native_sha256':sha,'native_scene':str(path),'resolution':[scene.render.resolution_x,scene.render.resolution_y],
             'samples':a.samples,'seed':0,'engine':'Blender Cycles CPU','view_transform':scene.view_settings.view_transform,
             'exposure':scene.view_settings.exposure,'seconds':round(time.time()-start,2)}
        done=[x for x in done if x['camera']!=r['camera']]+[row]
        manifest.write_text(json.dumps(done,ensure_ascii=False,indent=2))
        print('V17_RENDERED',i,filename,flush=True)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,'Read-only source changed'

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',default='screenshots_v17')
    p.add_argument('--width',type=int,default=1600);p.add_argument('--samples',type=int,default=48);p.add_argument('--views')
    p.add_argument('--cameras',default='data/review_cameras_v15.json')
    main(p.parse_args())
