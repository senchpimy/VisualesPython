import pyvista as pv
from pathlib import Path
import numpy as np
import time

def flatten_scene(data, mesh_list):
    if isinstance(data, pv.DataSet) and data.n_points > 0:
        mesh_list.append(data)
    elif isinstance(data, pv.MultiBlock):
        for block in data:
            flatten_scene(block, mesh_list)

model_path = Path("./model/scene.gltf")
original_data = pv.read(model_path)
all_meshes = []
flatten_scene(original_data, all_meshes)

indices_to_keep = [
    5, 9, 10, 11, 14, 15, 17, 19, 21, 22, 23, 24, 25, 26, 27, 29, 
    35, 36, 37, 39, 40, 45, 46, 49
]
kept_meshes = [all_meshes[i] for i in indices_to_keep]
def indices_to_keep_f(*args): return [indices_to_keep[i] for i in args]
group_indices = [
    indices_to_keep_f(0, 2, 3, 13, 14, 19, 20, 21),
    indices_to_keep_f(5, 10, 11, 12, 15, 16, 18, 22, 23),
    indices_to_keep_f(4, 6, 7, 8, 9, 16, 17)
]
groups_of_meshes = [[all_meshes[i].copy() for i in group] for group in group_indices]

z_offset = 1.0
group_configs = [
    {'meshes': groups_of_meshes[0], 'color': '#FF6347', 'label_file': '1.stl', 'final_pos': np.array([0, 0, z_offset])},
    {'meshes': groups_of_meshes[1], 'color': '#4682B4', 'label_file': '2.stl', 'final_pos': np.array([0, 0, 0])},
    {'meshes': groups_of_meshes[2], 'color': '#32CD32', 'label_file': '3.stl', 'final_pos': np.array([0, 0, -z_offset])}
]
for config in group_configs:
    config['initial_center'] = np.mean([mesh.center for mesh in config['meshes']], axis=0)

plotter = pv.Plotter(window_size=[1600, 1000])
plotter.window_title = "Despiece Final"
plotter.background_color = 'black'
plotter.enable_3_lights()

combined_mesh = pv.MultiBlock(kept_meshes).combine()
model_bounds = combined_mesh.bounds
model_size = np.linalg.norm([model_bounds[1]-model_bounds[0], model_bounds[3]-model_bounds[2], model_bounds[5]-model_bounds[4]])
x_pos_labels = (model_bounds[0] - model_size*0.9)

for config in group_configs:
    config['actors'] = [plotter.add_mesh(mesh, style='wireframe', color=config['color'], line_width=2) for mesh in config['meshes']]
    #print(act.__class__ for act in config['actors']) # generator object

padding = model_size * 0.05

for i, config in enumerate(group_configs):
    label_mesh = pv.read(config['label_file'])
    label_size = np.linalg.norm(np.array(label_mesh.bounds[1::2]) - np.array(label_mesh.bounds[0::2]))
    scale_factor = model_size / (label_size * 2)
    
    valz = config['initial_center'][2]
    static_pos = np.array([x_pos_labels, 0, valz])
    config['label_static_pos'] = static_pos
    
    label_mesh.scale(scale_factor, inplace=True)
    initial_label_pos = np.array([static_pos[0] - label_mesh.bounds[1], 0 - label_mesh.center[1], valz])
    label_mesh.translate(initial_label_pos, inplace=True)
    config['label_actor'] = plotter.add_mesh(label_mesh, color='#FFFFFF')
    
    bounds = label_mesh.bounds
    line_start_pos = np.array([bounds[1], (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
    
    vector = config['initial_center'] - line_start_pos
    direction = vector / np.linalg.norm(vector)
    
    line_start_with_padding = line_start_pos + direction * padding
    line_end_with_padding = config['initial_center'] - direction * padding
    
    line = pv.Line(line_start_with_padding, line_end_with_padding)
    config['line_mesh'] = line
    line_actor = plotter.add_mesh(line, color='grey')
    line_actor.prop.line_width = 2
    
plotter.camera_position = 'xy'
plotter.camera.zoom(1.1)
plotter.show(interactive=False, auto_close=False)

animation_speed = 0.5
while plotter.iren is not None and plotter.iren.initialized:
    t = np.abs(np.sin(time.time() * animation_speed))
    for config in group_configs:
        current_displacement = config['final_pos'] * t
        group_current_center = config['initial_center'] + current_displacement
        print("*"*50)
        for actor in config['actors']:
            actor.position = current_displacement
            print("Un actor")
        label_actor = config['label_actor']
        static_label_pos = config['label_static_pos']
        label_displacement = np.array([0, 0, current_displacement[2]])
        label_actor.position = label_displacement
        
        bounds = label_actor.bounds
        line_mesh = config['line_mesh']
        
        current_line_start = np.array([bounds[1], (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        vector = group_current_center - current_line_start
        direction = vector / np.linalg.norm(vector)

        line_mesh.points[0] = current_line_start + direction * padding
        line_mesh.points[1] = group_current_center - direction * padding
            
    plotter.iren.process_events()
    plotter.render()
    time.sleep(0.01)

plotter.close()
