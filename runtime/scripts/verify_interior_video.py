"""Recompute video/trajectory and modeled-room evidence without simulator imports."""
import argparse,json,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--physics-only',action='store_true');a=p.parse_args();root=a.directory.resolve()
assert root.is_relative_to(ROOT/'results')
subprocess.run([sys.executable,str(ROOT/'scripts/verify_cloth_pull.py'),str(root)],check=True)
r=json.loads((root/'report.json').read_text());d=np.load(root/'trajectory.npz')
assert r['rendering']['mode']==('physics_only' if a.physics_only else 'native_path_tracing')
audit=json.loads((root/'scene-geometry-audit.json').read_text())
assert audit['mesh_count']>100 and audit['vertices']>10000
assert audit['physical_tabletop']['collision']=='actual visible triangle mesh'
from pxr import Usd,UsdGeom,UsdPhysics
stage=Usd.Stage.Open(str(root/'initial.usda'))
assert abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)<1e-12
meshes=[UsdGeom.Mesh(p) for p in Usd.PrimRange(stage.GetPrimAtPath('/World/Brownstone01'),Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
assert len(meshes)==audit['mesh_count']
assert sum(len(m.GetPointsAttr().Get()) for m in meshes)==audit['vertices']
if (root/'appearance-audit.json').exists():
    appearance=json.loads((root/'appearance-audit.json').read_text())
    assert len(appearance['geometry_paths'])==appearance['render_geometry_count']
    for path in appearance['geometry_paths']:
        prim=stage.GetPrimAtPath(path);assert prim.IsA(UsdGeom.Gprim)
        if prim.IsA(UsdGeom.Mesh):
            mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get());assert np.isfinite(points).all()
            normals=mesh.GetNormalsAttr().Get()
            if normals:
                expected=len(mesh.GetFaceVertexIndicesAttr().Get()) if mesh.GetNormalsInterpolation()=='faceVarying' else len(points)
                assert len(normals)==expected
    for name,sha in appearance.get('texture_sha256',{}).items():
        path=ROOT/'assets/task-platform/textile-v2'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
table=stage.GetPrimAtPath(audit['physical_tabletop']['path'])
assert table.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(table).GetCollisionEnabledAttr().Get()
bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render']).ComputeWorldBound(table).ComputeAlignedRange()
assert abs(bounds.GetMax()[2])<1e-5 and bounds.GetSize()[2]>.03
usd_camera=UsdGeom.Camera(stage.GetPrimAtPath('/World/Camera'))
fx=1280*usd_camera.GetFocalLengthAttr().Get()/usd_camera.GetHorizontalApertureAttr().Get()
fy=720*usd_camera.GetFocalLengthAttr().Get()/usd_camera.GetVerticalApertureAttr().Get()
q=r['all_step_quality'];assert q['physics_steps_checked']==round(r['duration_s']*r['physics_hz'])
assert q['min_triangle_area_m2']>1e-10 and q['max_hand_binding_error_m']<.005 and q['max_node_speed_m_s']<2
x=d['cloth_nodes'];t=d['time_s'];poses=d['camera_poses']
assert len(poses)==len(t) and np.isfinite(poses).all()
if 'simulation_time_s' in d:
    assert np.allclose(np.diff(d['simulation_time_s']),1/r['fps'],rtol=0,atol=1e-8)
camera_travel=float(np.linalg.norm(poses[-1,:3]-poses[0,:3]));assert camera_travel>.5
hold=x[t>=8.];hold_range=float(np.linalg.norm(np.ptp(hold,axis=0),axis=1).max())
assert hold_range<.005,('Residual oscillation',hold_range)
# Sampled surface intersections complement all-step finite/area/binding checks.
import importlib.util
spec=importlib.util.spec_from_file_location('crossings',ROOT/'scripts/analyze_tasks.py')
crossings=importlib.util.module_from_spec(spec);spec.loader.exec_module(crossings)
indices=np.unique(np.linspace(0,len(t)-1,12,dtype=int))
counts=[crossings.crossing_pairs(x[i],d['faces']) for i in indices]
assert max(counts)==0,counts
if a.physics_only:
    result=dict(passed=True,scope='Dynamics preflight; no video or rendered-depth claim',
                all_step_quality=q,sampled_self_crossings=counts,last_two_seconds_max_node_range_m=hold_range)
    (root/'physics-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0)
depths=[]
for path in sorted(root.glob('depth-*.npz')):
    values=np.load(path)['depth_m'].squeeze();valid=np.isfinite(values)&(values>0)&(values<100)
    z=values[valid];assert len(z)>values.size*.5
    lo,hi=np.percentile(z,[5,95]);assert hi-lo>.3
    depths.append(dict(file=path.name,finite_fraction=float(valid.mean()),p05_m=float(lo),p95_m=float(hi)))
    # Fixed metric range across time: the same color always means the same Z.
    v=np.clip((values-.3)/(5.-.3),0,1);v[~valid]=1
    # A direct depth visualization; it is not substituted for the RGB scene.
    rgb=np.stack([.15+.8*v,.15+.7*(1-np.abs(2*v-1)),.95-.8*v],axis=-1)
    rgb[~valid]=0
    Image.fromarray((rgb*255).astype(np.uint8)).save(root/(path.stem+'.png'))
    if path.stem in (f'depth-{i:05d}' for i in (0,len(t)//2,len(t)-1)):
        # Interoperable 16-bit depth still: integer millimeters, zero is invalid.
        mm=np.where(valid&(values<65.535),np.rint(values*1000),0).astype(np.uint16)
        Image.fromarray(mm).save(root/(path.stem+'-mm.png'))
assert len(depths) in (3,len(t))
if len(depths)==len(t):
    for i in range(len(t)):
        sample=np.load(root/f'depth-{i:05d}.npz')
        assert abs(float(sample['time_s'])-t[i])<1e-9
        assert abs(float(sample['simulation_time_s'])-d['simulation_time_s'][i])<1e-9
        assert np.allclose(sample['camera_pose'],poses[i],atol=1e-9)
frames=sorted((root/'frames').glob('*.png'));assert len(frames)==r['frame_count']
assert [f.name for f in frames]==[f'{i:05d}.png' for i in range(len(frames))]
for f in frames:
    with Image.open(f) as im:assert im.size==(1280,720)
silhouettes=[]
for index in (0,len(t)//2,len(t)-1):
    eye=poses[index,:3];w,qx,qy,qz=poses[index,3:]
    rotation=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*w),2*(qx*qz+qy*w)],
                       [2*(qx*qy+qz*w),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*w)],
                       [2*(qx*qz-qy*w),2*(qy*qz+qx*w),1-2*(qx*qx+qy*qy)]])
    local=(x[index]-eye)@rotation;assert (local[:,2]<0).all()
    pixels=np.c_[640+fx*local[:,0]/(-local[:,2]),360-fy*local[:,1]/(-local[:,2])]
    projected=Image.new('1',(1280,720));draw=ImageDraw.Draw(projected)
    for triangle in d['faces']:draw.polygon([tuple(point) for point in pixels[triangle]],fill=1)
    expected=np.asarray(projected,dtype=bool)
    rgb=np.asarray(Image.open(frames[index]).convert('RGB'),dtype=float)
    observed=(rgb[:,:,1]>1.25*rgb[:,:,0])&(rgb[:,:,2]>1.2*rgb[:,:,0])&(rgb[:,:,1]>40)
    # Color is a diagnostic for this known teal specimen, not general segmentation.
    intersection=np.count_nonzero(expected&observed);union=np.count_nonzero(expected|observed)
    iou=intersection/max(union,1)
    projected.save(root/f'projected-cloth-{index:05d}.png')
    sample=np.load(root/f'depth-{index:05d}.npz')['depth_m'].squeeze()
    centers=x[index][d['faces']].mean(1)
    interior=(centers[:,2]>x[index,:,2].min()+.025)&(centers[:,2]<x[index,:,2].max()-.025)
    local_centers=(centers[interior]-eye)@rotation
    uv=np.rint(np.c_[640+fx*local_centers[:,0]/(-local_centers[:,2]),360-fy*local_centers[:,1]/(-local_centers[:,2])]).astype(int)
    inside=(uv[:,0]>=0)&(uv[:,0]<1280)&(uv[:,1]>=0)&(uv[:,1]<720)
    uv=uv[inside];expected_z=-local_centers[inside,2]
    residual=np.abs(sample[uv[:,1],uv[:,0]]-expected_z)
    # Camera rays hit rendered physical cloth, not a background billboard.
    assert np.median(residual)<.003 and np.percentile(residual,95)<.010,(index,np.percentile(residual,[50,95]))
    silhouettes.append(dict(frame=index,render_vs_gpu_silhouette_iou=float(iou),
        cloth_depth_median_error_m=float(np.median(residual)),cloth_depth_p95_error_m=float(np.percentile(residual,95))))
    normal_path=root/f'normals-{index:05d}.npz'
    if normal_path.exists():
        observed_normals=np.load(normal_path)['normals'][uv[:,1],uv[:,0],:3]
        triangles=x[index][d['faces']][interior][inside]
        geometric_normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
        geometric_normals/=np.linalg.norm(geometric_normals,axis=-1,keepdims=True)
        agreement=float(np.median(np.sum(observed_normals*geometric_normals,axis=-1)))
        assert agreement>.97,(index,agreement)
        silhouettes[-1]['normal_world_frame_median_cosine']=agreement
assert min(v['render_vs_gpu_silhouette_iou'] for v in silhouettes)>.75,silhouettes
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();video=root/'interior-cloth.mp4'
assert not video.exists()
subprocess.run([ffmpeg,'-v','error','-framerate',str(r['fps']),'-i',str(root/'frames/%05d.png'),'-an','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
decoded=subprocess.run([ffmpeg,'-v','error','-i',str(video),'-progress','pipe:1','-nostats','-f','null','-'],check=True,capture_output=True,text=True)
progress=dict(line.split('=',1) for line in decoded.stdout.splitlines() if '=' in line)
assert int(progress['frame'])==len(frames),progress
if len(depths)==len(t):
    depth_video=root/'interior-depth.mp4'
    assert not depth_video.exists()
    subprocess.run([ffmpeg,'-v','error','-framerate',str(r['fps']),'-i',str(root/'depth-%05d.png'),'-an','-c:v','libx264','-crf','17','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(depth_video)],check=True)
    decoded_depth=subprocess.run([ffmpeg,'-v','error','-i',str(depth_video),'-progress','pipe:1','-nostats','-f','null','-'],check=True,capture_output=True,text=True)
    assert int(dict(line.split('=',1) for line in decoded_depth.stdout.splitlines() if '=' in line)['frame'])==len(t)
    npz_files=sorted(root.glob('normals-*.npz'));assert len(npz_files)==3
    for path in npz_files:
        normals=np.load(path)['normals'][:,:,:3];length=np.linalg.norm(normals,axis=-1)
        assert normals.shape==(720,1280,3) and np.isfinite(normals).all()
        assert np.mean(np.abs(length-1)<.02)>.5
    # World-space point cloud from the final actual Z image, not inferred RGB depth.
    index=len(t)-1;values=np.load(root/f'depth-{index:05d}.npz')['depth_m'].squeeze()
    yy,xx=np.mgrid[0:720:4,0:1280:4];z=values[yy,xx]
    valid=np.isfinite(z)&(z>0)&(z<100)
    points_camera=np.stack([(xx-640)*z/fx,-(yy-360)*z/fy,-z],axis=-1)[valid]
    points_world=points_camera@rotation.T+eye
    colors=np.asarray(Image.open(frames[index]).convert('RGB'))[yy,xx][valid]
    np.savez_compressed(root/'pointcloud-final.npz',points_world_m=points_world,colors_rgb=colors,camera_pose=poses[index])
    with (root/'pointcloud-final.ply').open('w') as stream:
        stream.write('ply\nformat ascii 1.0\nelement vertex '+str(len(points_world))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for point,color in zip(points_world,colors):stream.write(' '.join(f'{v:.6f}' for v in point)+' '+' '.join(str(int(v)) for v in color)+'\n')
    calibration=json.loads((root/'camera-calibration.json').read_text())
    calibration['normals']='World-space XYZ unit normals from native RTX; frame verified against measured cloth triangle normals.'
    (root/'camera-calibration.json').write_text(json.dumps(calibration,indent=2)+'\n')
    dataset=dict(schema='deformable-rgbd-v1',calibration=calibration,
        depth_storage='NPZ float32 meters, Z distance along camera optical axis; infinity/zero may mean invalid',
        preview_storage='Depth PNG/MP4 are false color with fixed 0.3 to 5.0 meter range; not metric pixels',
        millimeter_stills='*-mm.png are uint16 millimeters; zero invalid',
        pose_convention='xyz world meters + quaternion wxyz; USD axes (+X right,+Y up,-Z forward). OpenCV xyz to USD xyz is diag(1,-1,-1).',
        frames=[dict(index=i,time_s=float(t[i]),simulation_time_s=float(d['simulation_time_s'][i]),
            rgb=f'frames/{i:05d}.png',depth=f'depth-{i:05d}.npz',camera_pose=poses[i].tolist()) for i in range(len(t))])
    (root/'rgbd-dataset.json').write_text(json.dumps(dataset,indent=2)+'\n')
detail_check=None
if (root/'detail-sensors.npz').exists():
    detail=np.load(root/'detail-sensors.npz');c=json.loads((root/'detail-camera.json').read_text())
    assert Image.open(root/'cloth-detail.png').size==(1920,1920)
    depth=detail['depth_m'].squeeze();assert depth.shape==(1920,1920)
    assert abs(float(detail['simulation_time_s'])-d['simulation_time_s'][-1])<1e-8
    eye_detail=np.array(c['camera_pose'][:3]);w,qx,qy,qz=c['camera_pose'][3:]
    rot=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*w),2*(qx*qz+qy*w)],
                  [2*(qx*qy+qz*w),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*w)],
                  [2*(qx*qz-qy*w),2*(qy*qz+qx*w),1-2*(qx*qx+qy*qy)]])
    centers=x[-1][d['faces']].mean(1)
    centers=centers[(centers[:,2]>x[-1,:,2].min()+.025)&(centers[:,2]<x[-1,:,2].max()-.025)]
    local=(centers-eye_detail)@rot
    uv=np.rint(np.c_[c['cx']+c['fx']*local[:,0]/(-local[:,2]),c['cy']-c['fy']*local[:,1]/(-local[:,2])]).astype(int)
    valid=(uv[:,0]>=0)&(uv[:,0]<1920)&(uv[:,1]>=0)&(uv[:,1]<1920)
    uv=uv[valid];err=np.abs(depth[uv[:,1],uv[:,0]]+local[valid,2])
    assert valid.mean()>.9 and np.percentile(err,95)<.005
    detail_check=dict(resolution=[1920,1920],cloth_depth_median_error_m=float(np.median(err)),cloth_depth_p95_error_m=float(np.percentile(err,95)))
    valid_depth=np.isfinite(depth)&(depth>0)&(depth<100)
    v=np.clip((depth-.3)/4.7,0,1);v[~valid_depth]=1
    rgb=np.stack([.15+.8*v,.15+.7*(1-np.abs(2*v-1)),.95-.8*v],axis=-1);rgb[~valid_depth]=0
    Image.fromarray((rgb*255).astype(np.uint8)).save(root/'detail-depth.png')
verification=dict(passed=True,simulation_duration_s=r['duration_s'],frame_count=len(frames),fps=r['fps'],
 camera_travel_m=camera_travel,last_two_seconds_max_node_range_m=hold_range,
 sampled_self_crossings=counts,sampled_self_crossing_frame_indices=indices.tolist(),depth=depths,render_physics_alignment=silhouettes,
 geometry_meshes=audit['mesh_count'],geometry_vertices=audit['vertices'],
 video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),video_bytes=video.stat().st_size,
 source='Uncomposited native RTX frames of live GPU dynamics within modeled USD architecture',
 limitations='One attached-edge elastic cloth task; no frictional grasp, folding, multi-body or calibrated-force validation. Self-intersection checks are sampled. Offline path-traced recording, not a real-time frame-rate claim.')
verification['paired_depth_frames']=len(depths)
verification['depth_visualization_range_m']=[.3,5.]
verification['appearance']='Fine textile PBR maps and modeled clamp details; surface dynamics are unchanged and uncalibrated.'
verification['detail_capture']=detail_check
verification['saved_usd_geometry_independently_verified']=True
verification['decoded_video_frames']=int(progress['frame'])
verification['artifacts']={}
for name in ('initial.usda','trajectory.npz','scene-geometry-audit.json','report.json'):
    path=root/name
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    verification['artifacts'][name]=dict(bytes=path.stat().st_size,sha256=sha)
for name in ('camera-calibration.json','rgbd-dataset.json','detail-camera.json','detail-sensors.npz','cloth-detail.png','appearance-audit.json','interior-depth.mp4','pointcloud-final.ply'):
    path=root/name
    if path.exists():
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        verification['artifacts'][name]=dict(bytes=path.stat().st_size,sha256=sha)
(root/'interior-verification.json').write_text(json.dumps(verification,indent=2)+'\n')
print(json.dumps(verification,indent=2))
