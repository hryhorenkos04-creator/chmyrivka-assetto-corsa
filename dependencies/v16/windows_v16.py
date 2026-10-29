"""Thin residential glazing and cloth shading; native geometry stays immutable.

The existing panes have no thickness. A single-surface Principled refraction
lobe was obscuring the already modelled curtains. A Fresnel-weighted glossy /
straight-through transparent sheet represents that geometry explicitly.
"""
from pathlib import Path
import json
import bpy

ROOT = Path(__file__).resolve().parent

def thin_glass(old):
    mat = old.copy(); mat.name = 'V16_Window_Thin_' + old.name
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    out = nodes.new('ShaderNodeOutputMaterial')
    clear = nodes.new('ShaderNodeBsdfTransparent')
    clear.inputs['Color'].default_value = (.975, .984, .982, 1)
    reflection = nodes.new('ShaderNodeBsdfGlossy')
    reflection.inputs['Color'].default_value = (.97, .98, .98, 1)
    fresnel = nodes.new('ShaderNodeFresnel'); fresnel.inputs['IOR'].default_value = 1.50
    position = nodes.new('ShaderNodeNewGeometry')
    noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 2.2
    noise.inputs['Detail'].default_value = 2
    rough = nodes.new('ShaderNodeMapRange')
    rough.inputs['From Min'].default_value = .2; rough.inputs['From Max'].default_value = .8
    rough.inputs['To Min'].default_value = .035; rough.inputs['To Max'].default_value = .085
    links.new(position.outputs['Position'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], rough.inputs['Value'])
    links.new(rough.outputs['Result'], reflection.inputs['Roughness'])
    mix = nodes.new('ShaderNodeMixShader')
    links.new(fresnel.outputs['Fac'], mix.inputs[0])
    links.new(clear.outputs[0], mix.inputs[1]); links.new(reflection.outputs[0], mix.inputs[2])
    links.new(mix.outputs[0], out.inputs['Surface'])
    mat.diffuse_color = (.18, .22, .23, 1)
    mat['v16_method'] = 'Thin sheet: IOR1.50 Fresnel reflection and tinted straight-through transmission. Existing recessed curtains/interior retained.'
    mat['ac_conversion'] = 'Native glass response must be mapped deliberately to ksEditor glass shader; this is not an AC material.'
    return mat

def cloth(old):
    mat = old.copy(); mat.name = 'V16_Window_Cloth_' + old.name
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    p = next(n for n in nodes if n.type == 'BSDF_PRINCIPLED')
    p.inputs['Roughness'].default_value = .88
    p.inputs['Specular IOR Level'].default_value = .20
    p.inputs['Sheen Weight'].default_value = .12
    # Retain the existing woven colour and bump rather than introduce another
    # large-scale pattern to tiny pieces of folded fabric.
    fabric = nodes.new('ShaderNodeBsdfTranslucent')
    color_links = list(p.inputs['Base Color'].links)
    if color_links: links.new(color_links[0].from_socket, fabric.inputs['Color'])
    else: fabric.inputs['Color'].default_value = p.inputs['Base Color'].default_value
    mix = nodes.new('ShaderNodeMixShader'); mix.inputs[0].default_value = .14
    links.new(p.outputs[0], mix.inputs[1]); links.new(fabric.outputs[0], mix.inputs[2])
    out = next(n for n in nodes if n.type == 'OUTPUT_MATERIAL')
    links.new(mix.outputs[0], out.inputs['Surface'])
    mat['v16_method'] = 'Existing folded fabric geometry and colour; restrained thin-fabric transmission, no emitted light.'
    return mat

def apply(scene, collection, source_data=None):
    copies, bindings = {}, []
    for ob in scene.objects:
        if ob.type != 'MESH' or not ob.name.startswith('V12_Architecture_'): continue
        for i, slot in enumerate(ob.material_slots):
            old = slot.material
            if old is None: continue
            if old.name in ('V12_Architecture_Window_Glass', 'V12_Architecture_Window_Shadow'):
                kind = 'glass'
            elif old.name in ('V12_Architecture_curtain_white', 'V12_Architecture_curtain_linen'):
                kind = 'cloth'
            else: continue
            if old.name not in copies:
                copies[old.name] = thin_glass(old) if kind == 'glass' else cloth(old)
            slot.material = copies[old.name]
            bindings.append({'object':ob.name, 'slot':i, 'old_material':old.name,
                             'new_material':slot.material.name, 'kind':kind})
    assert any(r['kind']=='glass' for r in bindings), 'Expected local glazing objects were not found'
    report = {'version':'v16', 'geometry_changed':False, 'new_materials':[m.name for m in copies.values()],
              'bindings':bindings, 'count':len(bindings),
              'accuracy':'Existing inferred regional windows; no photographed address claim',
              'limitation':'Thin-glass Blender shading must be translated for AC; no emissive cheat or baked fake reflection.'}
    (ROOT/'data/windows_v16.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
