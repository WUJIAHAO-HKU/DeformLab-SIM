"""Procedural textile PBR and modeled test-fixture details; no backdrop images.

Geometry details do not replace the explicit attachment grip or change mass.
Normal maps represent submillimeter fibers, not extra simulated deformation.
"""
from pathlib import Path
import json,hashlib
import numpy as np


def make_textures(directory):
    from PIL import Image
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    n=1024;y,x=np.mgrid[:n,:n]/n
    # 0.5 mm yarn spacing on the 140 x 280 mm rest specimen.
    u=x*280;v=y*560
    warp=np.cos(2*np.pi*u);weft=np.cos(2*np.pi*v)
    over=np.cos(np.pi*u)*np.cos(np.pi*v)
    height=.000022*(warp+weft)+.000009*over
    gy,gx=np.gradient(height,.28/n,.14/n)
    normal=np.stack([-gx,-gy,np.ones_like(gx)],axis=-1)
    normal/=np.linalg.norm(normal,axis=-1,keepdims=True)
    rng=np.random.default_rng(42)
    grain=rng.normal(0,1,(n,n))
    variation=1+.06*warp+.05*weft+.025*over+.012*grain
    # Muted blue-green elastic textile, not a calibrated cotton claim.
    base=np.array([37.,108.,119.])[None,None,:]*variation[:,:,None]
    # Fine pale selvedge, 1.1 mm each side, advected by material UVs.
    edge=(x<.0011/.14)|(x>1-.0011/.14)
    base[edge]=np.array([125.,166.,166.])*variation[edge,None]
    for name,data in [('textile-base.png',base),
                      ('textile-normal.png',(normal*.5+.5)*255),
                      ('textile-roughness.png',np.clip(.86+.035*warp+.025*weft,.72,.97)*255)]:
        Image.fromarray(np.clip(data,0,255).astype(np.uint8)).save(directory/name)
    return dict(yarn_spacing_m=.0005,normal_height_amplitude_m=.000053,
                texture_resolution=[n,n],scope='Procedural appearance maps; no material calibration or force change')


def cloth_material(stage,visual,nx,nz,root):
    from pxr import UsdGeom,UsdShade,Sdf,Vt,Gf
    folder=Path(root)/'assets/task-platform/textile-v2';make_textures(folder)
    mat=UsdShade.Material.Define(stage,'/World/WovenFabric')
    shader=UsdShade.Shader.Define(stage,'/World/WovenFabric/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
    shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.46)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    reader=UsdShade.Shader.Define(stage,'/World/WovenFabric/UV');reader.CreateIdAttr('UsdPrimvarReader_float2')
    reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    for label,file,input_name,dtype,output,space in [
        ('Color','textile-base.png','diffuseColor',Sdf.ValueTypeNames.Color3f,'rgb','sRGB'),
        ('Normal','textile-normal.png','normal',Sdf.ValueTypeNames.Normal3f,'rgb','raw'),
        ('Roughness','textile-roughness.png','roughness',Sdf.ValueTypeNames.Float,'r','raw')]:
        tex=UsdShade.Shader.Define(stage,'/World/WovenFabric/'+label);tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(str(folder/file))
        tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(space)
        tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
        if label=='Normal':
            tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
            tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
        tex.CreateOutput(output,Sdf.ValueTypeNames.Float3 if output=='rgb' else Sdf.ValueTypeNames.Float)
        shader.CreateInput(input_name,dtype).ConnectToSource(tex.ConnectableAPI(),output)
    uv=np.array([[i/nx,j/nz] for j in range(nz+1) for i in range(nx+1)],np.float32)
    UsdGeom.PrimvarsAPI(visual).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex').Set(Vt.Vec2fArray.FromNumpy(uv))
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(mat)


def fixture(stage,fixed,width,out):
    from pxr import UsdGeom,UsdShade,Sdf,Vt,Gf,Usd
    fixed=np.asarray(fixed);root='/World/FixtureDetails'
    UsdGeom.Xform.Define(stage,root)
    def material(name,color,roughness,metallic):
        path=root+'/Materials/'+name;mat=UsdShade.Material.Define(stage,path)
        s=UsdShade.Shader.Define(stage,path+'/Surface');s.CreateIdAttr('UsdPreviewSurface')
        for key,value,typ in [('diffuseColor',Gf.Vec3f(*color),Sdf.ValueTypeNames.Color3f),('roughness',roughness,Sdf.ValueTypeNames.Float),('metallic',metallic,Sdf.ValueTypeNames.Float)]:s.CreateInput(key,typ).Set(value)
        mat.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(),'surface');return mat
    steel=material('SatinSteel',[.48,.52,.56],.27,1.)
    alloy=material('BeadBlastedAluminum',[.29,.34,.39],.38,1.)
    dark=material('AnodizedAluminum',[.055,.065,.075],.31,.8)
    rubber=material('Rubber',[.017,.019,.021],.91,0.)
    def mesh(path,points,faces,mat,normals=None):
        m=UsdGeom.Mesh.Define(stage,path);p=np.asarray(points,np.float32)
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(p));m.CreateFaceVertexCountsAttr([len(f) for f in faces]);m.CreateFaceVertexIndicesAttr(np.concatenate(faces).tolist())
        m.CreateSubdivisionSchemeAttr('none');m.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([p.min(0),p.max(0)],np.float32)))
        if normals is not None:
            # Explicit corner normals also avoid Fabric's transient vertex /
            # face-varying size mismatch when importing freshly authored meshes.
            corner_normals=np.asarray(normals,np.float32)[np.concatenate(faces)]
            m.SetNormalsInterpolation('faceVarying');m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(corner_normals))
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat);return m
    def rounded(path,center,size,r,mat):
        half=np.array(size)/2;inner=half-r;vertices=[];normals=[];faces=[]
        for axis in range(3):
            b=(axis+1)%3;c=(axis+2)%3
            for sign in [-1,1]:
                vals=[np.unique(np.r_[-half[k],-inner[k]-.707*r,-inner[k],0,inner[k],inner[k]+.707*r,half[k]]) for k in (b,c)]
                start=len(vertices);nv=len(vals[1])
                for u in vals[0]:
                    for v in vals[1]:
                        p=np.zeros(3);p[axis]=sign*half[axis];p[b]=u;p[c]=v
                        nearest=np.clip(p,-inner,inner);n=p-nearest;n/=np.linalg.norm(n)
                        vertices.append(nearest+r*n+center);normals.append(n)
                for i in range(len(vals[0])-1):
                    for j in range(nv-1):
                        ids=[start+i*nv+j,start+(i+1)*nv+j,start+(i+1)*nv+j+1,start+i*nv+j+1]
                        faces.append(ids if sign>0 else ids[::-1])
        return mesh(path,vertices,faces,mat,normals)
    def cylinder(path,center,r,h,mat):
        m=UsdGeom.Cylinder.Define(stage,path);m.CreateRadiusAttr(r);m.CreateHeightAttr(h);m.CreateAxisAttr('Z')
        m.AddTranslateOp().Set(Gf.Vec3d(*center));UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(mat)
    def bolt(path,center,r=.0045,axis='Z'):
        # Actual recessed hex socket: rim, cavity walls and cavity floor.
        n=36;theta=np.arange(n)*2*np.pi/n
        outer=np.c_[r*np.cos(theta),r*np.sin(theta)]
        hexv=np.c_[.52*r*np.cos(np.arange(6)*np.pi/3),.52*r*np.sin(np.arange(6)*np.pi/3)]
        inner=np.array([(1-(i%6)/6)*hexv[i//6]+((i%6)/6)*hexv[(i//6+1)%6] for i in range(n)])
        points=np.vstack([np.c_[outer,np.full(n,z)] for z in (-.0015,.0025)]+[np.c_[inner,np.full(n,z)] for z in (.0025,.0004)])
        faces=[]
        for j in range(n):
            k=(j+1)%n;faces.extend([[j,k,n+k,n+j],[n+j,n+k,2*n+k,2*n+j],[2*n+j,2*n+k,3*n+k,3*n+j]])
        faces.extend([list(range(3*n,4*n)),list(range(n-1,-1,-1))])
        if axis=='Y':points=points[:,[0,2,1]];points[:,1]*=-1
        mesh(path,points+center,faces,steel)
    # Preserve existing collider and attachment frames. The visible bevel lies
    # within the original base envelope; only its corners differ by <= 2 mm.
    for path in ['/World/FixtureBase','/World/FixturePost','/World/FixedClamp/Jaw0','/World/FixedClamp/Jaw2','/World/Franka/panda_hand/PullPad0','/World/Franka/panda_hand/PullPad2']:
        UsdGeom.Imageable(stage.GetPrimAtPath(path)).MakeInvisible()
    rounded(root+'/Base',np.array([fixed[0],fixed[1],.022]),[.25,.18,.044],.002,dark)
    for i,x in enumerate([-.095,.095]):
        for j,y in enumerate([-.065,.065]):
            cylinder(root+f'/Washer{i}{j}',[fixed[0]+x,fixed[1]+y,.0445],.007,.001,steel)
            bolt(root+f'/BaseBolt{i}{j}',[fixed[0]+x,fixed[1]+y,.046])
    for i,x in enumerate([-.057,.057]):
        cylinder(root+f'/Column{i}',[fixed[0]+x,fixed[1]+.047,(fixed[2]+.044)/2],.009,fixed[2]-.044,steel)
        cylinder(root+f'/ColumnFoot{i}',[fixed[0]+x,fixed[1]+.047,.051],.014,.014,alloy)
    rounded(root+'/Crosshead',fixed+[0,.036,0],[width+.045,.055,.032],.0015,alloy)
    # Fixed jaws and moving adapter share the same specimen gap, with a
    # 0.6 mm visual clearance around the 1 mm nominal cloth surface.
    for moving,parent,origin in [(False,root+'/Fixed',fixed),(True,'/World/Franka/panda_hand/TensileAdapter',np.array([0.,0.,.10]))]:
        UsdGeom.Xform.Define(stage,parent)
        for side in [-1,1]:
            label='Front' if side<0 else 'Back'
            rounded(parent+'/'+label,origin+[0,side*.012,0],[width+.035,.018,.027],.0012,alloy)
            rounded(parent+'/'+label+'Rubber',origin+[0,side*.0019,0],[width+.002,.0022,.019],.0003,rubber)
            for i,x in enumerate([-.06,0,.06]):bolt(parent+'/'+label+f'Screw{i}',origin+[x,side*.022,0],.0038,axis='Y')
        if moving:
            for i,x in enumerate([-.025,.025]):rounded(parent+f'/Yoke{i}',origin+[x,.017,-.029],[.012,.014,.038],.001,steel)
    prims=[p for p in stage.Traverse() if str(p.GetPath()).startswith((root,'/World/Franka/panda_hand/TensileAdapter')) and p.IsA(UsdGeom.Gprim)]
    report=dict(render_geometry_count=len(prims),geometry_paths=[str(p.GetPath()) for p in prims],
        details='Beveled metal base and jaws, cylindrical columns, hex socket recesses, rubber pads',
        dynamics='Original base collider, explicit attachment grip and mass retained. Details are visual rigid geometry; no friction-grip validation.',
        base_collider_max_bevel_difference_m=.002,cloth_visual='Measured surface plus fiber normal and roughness maps; no generated shape animation')
    textures=Path(__file__).resolve().parents[1]/'assets/task-platform/textile-v2'
    report['texture_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(textures.glob('*.png'))}
    (Path(out)/'appearance-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
