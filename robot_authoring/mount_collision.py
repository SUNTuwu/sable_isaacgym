"""Triangle-surface intersection and closed-component containment checks."""
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

def components(obj, transform=None):
    vv=np.array([v.co for v in obj.data.vertices],dtype=np.float64)
    if transform is not None:vv=vv@transform[:3,:3].T+transform[:3,3]
    adjacency={i:set() for i in range(len(vv))}
    for e in obj.data.edges:
        a,b=e.vertices;adjacency[a].add(b);adjacency[b].add(a)
    unseen=set(adjacency);result=[]
    obj.data.calc_loop_triangles()
    while unseen:
        todo=[next(iter(unseen))];ids=set()
        while todo:
            i=todo.pop()
            if i in ids:continue
            ids.add(i);todo.extend(adjacency[i]-ids)
        unseen-=ids
        faces=[list(t.vertices) for t in obj.data.loop_triangles if t.vertices[0] in ids]
        edges={}
        for face in faces:
            for a,b in zip(face,face[1:]+face[:1]):
                edge=tuple(sorted((a,b)));edges[edge]=edges.get(edge,0)+1
        closed=all(n==2 for n in edges.values())
        result.append({'vertices':vv[sorted(ids)],'triangles':vv[np.array(faces)],
                       'tree':BVHTree.FromPolygons([Vector(p) for p in vv],faces,all_triangles=True,epsilon=1e-9),
                       'vertex_count':len(ids),'closed':closed})
    return result

def inside(point,triangles):
    # Non-axis-aligned rays avoid cylinder grid degeneracies. Deduplicate hits
    # on shared triangle edges and vote across three rays.
    votes=[]
    for direction in [[1,.371,.219],[.173,1,.413],[.279,.153,1]]:
        d=np.array(direction,dtype=float);d/=np.linalg.norm(d)
        a=triangles[:,0];e1=triangles[:,1]-a;e2=triangles[:,2]-a
        h=np.cross(d,e2);det=np.einsum('ij,ij->i',e1,h)
        mask=abs(det)>1e-14;inv=np.zeros(len(det));inv[mask]=1/det[mask]
        s=point-a;u=np.einsum('ij,ij->i',s,h)*inv
        q=np.cross(s,e1);v=q@d*inv;t=np.einsum('ij,ij->i',e2,q)*inv
        hits=sorted(t[mask&(u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(t>1e-8)])
        unique=[]
        for x in hits:
            if not unique or x-unique[-1]>1e-8:unique.append(x)
        votes.append(len(unique)%2==1)
    return sum(votes)>=2

def interference(a,b):
    pairs=[]
    for i,ca in enumerate(a):
        for j,cb in enumerate(b):
            overlap=ca['tree'].overlap(cb['tree'])
            containment=False if overlap else ((cb['closed'] and inside(ca['vertices'][0],cb['triangles'])) or
                                                (ca['closed'] and inside(cb['vertices'][0],ca['triangles'])))
            pairs.append({'roll_component':i,'pitch_component':j,'roll_vertices':ca['vertex_count'],
                          'pitch_vertices':cb['vertex_count'],'intersecting_triangle_pairs':len(overlap),
                          'roll_closed':ca['closed'],'pitch_closed':cb['closed'],
                          'containment':bool(containment),'penetrating':bool(overlap or containment)})
    return {'penetrating':any(p['penetrating'] for p in pairs),'pairs':pairs,
            'method':'Actual triangle surfaces (BVH) plus closed-component ray-parity containment; authoring zero only',
            'surface_tolerance_m':1e-9,'touching_surfaces_count_as_interference':True,
            'scope':'Mesh intersection; containment only for closed components; not a full motion-envelope or hardware-clearance certificate'}
