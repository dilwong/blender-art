import bpy
import bmesh
import numpy as np
import math

twist_angle = 10
a = 1.42

radians = np.radians(twist_angle)
R = np.array(((np.cos(radians),-np.sin(radians),0), (np.sin(radians), np.cos(radians),0), (0, 0, 1)))

def bravais_lattice(n1, n2, height):
    a1 = a * np.array([-np.sqrt(3)/2, 3/2])
    a2 = a * np.array([np.sqrt(3)/2, 3/2])
    return (np.append(x * a1 + y * a2, height) for x in range(-n1,n1) for y in range(-n2,n2))

# def rotated_lattice(n1, n2, height, theta):
#     radians = np.radians(theta)
#     R = np.array(((np.cos(radians),-np.sin(radians),0), (np.sin(radians), np.cos(radians),0), (0, 0, 1)))
#     return (R.dot(positions) for positions in bravais_lattice(n1, n2, height))

mesh = bpy.data.meshes.new('mesh')
graph = bpy.data.objects.new('graph', mesh)
bpy.context.collection.objects.link(graph)
bpy.context.view_layer.objects.active = graph
bpy.context.active_object.select_set(state=True)
mesh = bpy.context.object.data
bm = bmesh.new()

for site in bravais_lattice(20,20,0):
    if np.abs(site[0]) < a*15 and np.abs(site[1]) < a*15:
        bm.verts.new(site)

bm.to_mesh(mesh)
bm.free()

mat=bpy.data.materials.new('Carbon')
mat.diffuse_color=(0,0,0,1)
bpy.ops.mesh.primitive_uv_sphere_add(location=(0,0,0), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom A'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom A'].parent = bpy.data.objects['graph']
bpy.ops.mesh.primitive_uv_sphere_add(location=(0,-a,0), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom B'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom B'].parent = bpy.data.objects['graph']
bpy.ops.mesh.primitive_uv_sphere_add(location=(np.sqrt(3)*a/2,-3*a/2,0), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom C'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom C'].parent = bpy.data.objects['graph']
bpy.ops.mesh.primitive_uv_sphere_add(location=(np.sqrt(3)*a,-a,0), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom D'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom D'].parent = bpy.data.objects['graph']
bpy.ops.mesh.primitive_uv_sphere_add(location=(np.sqrt(3)*a,0,0), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom E'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom E'].parent = bpy.data.objects['graph']
bpy.ops.mesh.primitive_uv_sphere_add(location=(np.sqrt(3)*a/2,a/2,0), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom F'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom F'].parent = bpy.data.objects['graph']

def cylinder_between(x1, y1, z1, x2, y2, z2, r, name):
  dx = x2 - x1
  dy = y2 - y1
  dz = z2 - z1    
  dist = math.sqrt(dx**2 + dy**2 + dz**2)
  bpy.ops.mesh.primitive_cylinder_add(
      radius = r, 
      depth = dist,
      location = (dx/2 + x1, dy/2 + y1, dz/2 + z1)   
  ) 
  phi = math.atan2(dy, dx) 
  theta = math.acos(dz/dist) 
  bpy.context.object.rotation_euler[1] = theta 
  bpy.context.object.rotation_euler[2] = phi 
  bond = bpy.context.object
  bpy.context.active_object.name = name
  bond.data.materials.append(mat)

cylinder_between(0,0,0,0,-a,0,0.1,'Bond 1')
cylinder_between(0,-a,0,np.sqrt(3)*a/2, -3*a/2,0, 0.1 ,'Bond 2')
cylinder_between(np.sqrt(3)*a/2, -3*a/2,0,np.sqrt(3)*a, -a, 0, 0.1 ,'Bond 3')
cylinder_between(np.sqrt(3)*a,-a, 0, np.sqrt(3)*a, 0, 0, 0.1 ,'Bond 4')
cylinder_between(np.sqrt(3)*a, 0, 0, np.sqrt(3)*a/2, a/2, 0, 0.1 ,'Bond 5')
cylinder_between(np.sqrt(3)*a/2, a/2, 0, 0, 0, 0, 0.1 ,'Bond 6')
bpy.data.objects['Bond 1'].parent = bpy.data.objects['graph']
bpy.data.objects['Bond 2'].parent = bpy.data.objects['graph']
bpy.data.objects['Bond 3'].parent = bpy.data.objects['graph']
bpy.data.objects['Bond 4'].parent = bpy.data.objects['graph']
bpy.data.objects['Bond 5'].parent = bpy.data.objects['graph']
bpy.data.objects['Bond 6'].parent = bpy.data.objects['graph']

bpy.data.objects['graph'].instance_type = 'VERTS'

#######################

mesh_lower = bpy.data.meshes.new('mesh_lower')
graph_lower = bpy.data.objects.new('graph_lower', mesh_lower)
bpy.context.collection.objects.link(graph_lower)
bpy.context.view_layer.objects.active = graph_lower
bpy.context.active_object.select_set(state=True)
mesh_lower = bpy.context.object.data
bm_lower = bmesh.new()

for site in bravais_lattice(20,20,0):
    if np.abs(site[0]) < a*15 and np.abs(site[1]) < a*15:
        bm_lower.verts.new(R.dot(site))

bm_lower.to_mesh(mesh_lower)
bm_lower.free()

bpy.ops.mesh.primitive_uv_sphere_add(location=(0,0,0), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom A Lower
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom A Lower'].parent = bpy.data.objects['graph_lower']
bpy.ops.mesh.primitive_uv_sphere_add(location=R.dot((0,-a,0)), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom B Lower'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom B Lower'].parent = bpy.data.objects['graph_lower']
bpy.ops.mesh.primitive_uv_sphere_add(location=R.dot((np.sqrt(3)*a/2,-3*a/2,0)), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom C Lower'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom C Lower'].parent = bpy.data.objects['graph_lower']
bpy.ops.mesh.primitive_uv_sphere_add(location=R.dot((np.sqrt(3)*a,-a,0)), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom D Lower'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom D Lower'].parent = bpy.data.objects['graph_lower']
bpy.ops.mesh.primitive_uv_sphere_add(location=R.dot((np.sqrt(3)*a,0,0)), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom E Lower'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom E Lower'].parent = bpy.data.objects['graph_lower']
bpy.ops.mesh.primitive_uv_sphere_add(location=R.dot((np.sqrt(3)*a/2,a/2,0)), radius = .3, ring_count = 48, segments = 48)
sphere=bpy.context.object
bpy.context.active_object.name = 'Carbon Atom F Lower'
sphere.data.materials.append(mat)
bpy.data.objects['Carbon Atom F Lower'].parent = bpy.data.objects['graph_lower']

c1,c2,c3 = R.dot((0,-a,0))
cylinder_between(0,0,0,c1,c2,c3,0.1,'Bond 1 Lower')
d1,d2,d3 = R.dot((np.sqrt(3)*a/2, -3*a/2,0))
cylinder_between(c1,c2,c3,d1,d2,d3, 0.1 ,'Bond 2 Lower')
c1,c2,c3 = R.dot((np.sqrt(3)*a, -a, 0))
cylinder_between(d1,d2,d3,c1,c2,c3, 0.1 ,'Bond 3 Lower')
d1,d2,d3 = R.dot((np.sqrt(3)*a, 0, 0))
cylinder_between(c1,c2,c3,d1,d2,d3, 0.1 ,'Bond 4 Lower')
c1,c2,c3 = R.dot((np.sqrt(3)*a/2, a/2, 0))
cylinder_between(d1,d2,d3,c1,c2,c3, 0.1 ,'Bond 5 Lower')
cylinder_between(c1,c2,c3, 0, 0, 0, 0.1 ,'Bond 6 Lower')
bpy.data.objects['Bond 1 Lower'].parent = bpy.data.objects['graph_lower']
bpy.data.objects['Bond 2 Lower'].parent = bpy.data.objects['graph_lower']
bpy.data.objects['Bond 3 Lower'].parent = bpy.data.objects['graph_lower']
bpy.data.objects['Bond 4 Lower'].parent = bpy.data.objects['graph_lower']
bpy.data.objects['Bond 5 Lower'].parent = bpy.data.objects['graph_lower']
bpy.data.objects['Bond 6 Lower'].parent = bpy.data.objects['graph_lower']

bpy.data.objects['graph_lower'].instance_type = 'VERTS'

##########

bn_mat=bpy.data.materials.new('Boron Nitride')
#bn_mat.diffuse_color=(0,.66,.87)
bn_mat.diffuse_color=(0,.4,1,1)
bpy.ops.mesh.primitive_cube_add(location=(-4*a,8*a,-1*a))
bn=bpy.context.object
bpy.context.active_object.name = 'BN'
bpy.context.view_layer.objects.active.scale = (20*a, 20*a, 0.5*a)
bn.data.materials.append(bn_mat)
#bn.rotation_euler = (0, 0, 1)

##########

au_mat=bpy.data.materials.new('Gold')
#au_mat.diffuse_color=(.98,.74,.64)
au_mat.diffuse_color=(1,.75,.25,1)
bpy.ops.mesh.primitive_cube_add()
gold=bpy.context.object
bpy.context.active_object.name = 'Au'
bpy.context.view_layer.objects.active.scale = (8*a, 8*a, 0.5*a)
gold.data.materials.append(au_mat)
gold.location=(17.95*a,-11.95*a,-1.4*a)

##########

sio2_mat=bpy.data.materials.new('Silicon Oxide')
#sio2_mat.diffuse_color=(0.55,.44,0.69)
#sio2_mat.diffuse_color=(0.49,.32,0.72)
sio2_mat.diffuse_color=(0.42,.24,0.67,1)
bpy.ops.mesh.primitive_cube_add()
sio2=bpy.context.object
bpy.context.active_object.name = 'SiO2'
bpy.context.view_layer.objects.active.scale = (30*a,30*a, 2*a)
sio2.data.materials.append(sio2_mat)
sio2.location = (-4*a,10*a,-3*a)

##########

si_mat=bpy.data.materials.new('Silicon')
si_mat.diffuse_color=(0.5,0.5,0.5,1)
bpy.ops.mesh.primitive_cube_add()
si=bpy.context.object
bpy.context.active_object.name = 'Si'
bpy.context.view_layer.objects.active.scale = (30*a,30*a, 4*a)
si.data.materials.append(si_mat)
si.location = (-4*a,10*a,-9*a)

##########

w_mat=bpy.data.materials.new('W')
#w_mat.diffuse_color=(0.5,0.5,0.5)
w_mat.specular_color=(0.5,0.5,0.5)
bpy.ops.mesh.primitive_cone_add(vertices = 128)
tip=bpy.context.object
bpy.context.active_object.name = 'Tip'
bpy.context.view_layer.objects.active.scale = (5*a,5*a, -10*a)
tip.data.materials.append(w_mat)
tip.location = (5*a,-5*a,15*a)