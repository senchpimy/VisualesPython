import time
from camera_handler import run_detector

import pyvista as pv
from pathlib import Path
import numpy as np
from multiprocessing import Process, Queue, freeze_support
from queue import Empty
import cv2

class Gestures:
    EXPANDED_HAND = "Mano expandida"
    L_FINGERS = "Dedos L"
    THREE_FINGERS = "3 Dedos"

def cubic_bezier_easing(t):
    if t < 0.0: t = 0.0
    if t > 1.0: t = 1.0
    
    return t * t * (3.0 - 2.0 * t)

def _cubic_bezier(t_param, p1, p2):
    return 3 * (1 - t_param)**2 * t_param * p1 + 3 * (1 - t_param) * t_param**2 * p2 + t_param**3

def cubic_bezier_ease_in_out(t, c1x=0.42, c1y=0.0, c2x=0.58, c2y=1.0):
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0

    lower_bound = 0.0
    upper_bound = 1.0
    t_param = t # Una buena estimación inicial

    for _ in range(12):
        x_guess = _cubic_bezier(t_param, c1x, c2x)
        
        if abs(x_guess - t) < 1e-6:
            break
        
        if x_guess > t:
            upper_bound = t_param
        else:
            lower_bound = t_param
        
        t_param = (upper_bound + lower_bound) / 2.0

    return _cubic_bezier(t_param, c1y, c2y)


def main_app():
    gesture_queue = Queue(maxsize=2)
    detector_proc = Process(target=run_detector, args=(gesture_queue,), daemon=True)
    detector_proc.start()
    print("\nProcesos de detección y visualización iniciados.")

    print("Cargando modelo 3D...")
    model_path = Path("./model/scene.gltf")
    if not model_path.exists():
        print(f"ERROR: No se encuentra el modelo en '{model_path}'.")
        return
    
    # Cargamos de el modelo solo las partes que nos interesan
    def flatten_scene(data, mesh_list):
        if isinstance(data, pv.DataSet) and data.n_points > 0:
            mesh_list.append(data)
        elif isinstance(data, pv.MultiBlock):
            for block in data:
                flatten_scene(block, mesh_list)

    original_data = pv.read(model_path)
    all_meshes = []
    flatten_scene(original_data, all_meshes)
    indices_to_keep = [5, 9, 10, 11, 14, 15, 17, 19, 21, 22, 23, 24, 25, 26, 27, 29, 35, 36, 37, 39, 40, 45, 46, 49]
    kept_meshes = [all_meshes[i] for i in indices_to_keep]
    def indices_to_keep_f(*args): return [kept_meshes[i] for i in args]
    group_indices = [
        indices_to_keep_f(0, 2, 3, 13, 14, 19, 20, 21),
        indices_to_keep_f(5, 10, 11, 12, 15, 16, 18, 22, 23),
        indices_to_keep_f(4, 6, 7, 8, 9, 16, 17)
    ]
    groups_of_meshes_orig = group_indices

    # Creamos la ventana
    plotter = pv.Plotter(window_size=[1080, 720])
    plotter.window_title = "Control de Despiece con Gestos"
    plotter.background_color = 'black'
    plotter.enable_3_lights()
    #plotter.add_axes()

    group_configs = [
            {'meshes': groups_of_meshes_orig[0], 'color': '#FF6347', 'label_file': '1.stl', 'pos': np.array([0, 0, 0]), 'total_rotation': 0.0, 'animation_t': 0.0},
            {'meshes': groups_of_meshes_orig[1], 'color': '#4682B4', 'label_file': '2.stl', 'pos': np.array([0, 0, 0]), 'total_rotation': 0.0, 'animation_t': 0.0},
            {'meshes': groups_of_meshes_orig[2], 'color': '#32CD32', 'label_file': '3.stl', 'pos': np.array([0, 0, 0]), 'total_rotation': 0.0, 'animation_t': 0.0}
    ]
    z_offset = 1.0

    group_configs[0]['final_pos'] = np.array([0, 0, z_offset])
    group_configs[1]['final_pos'] = np.array([0, 0, 0])
    group_configs[2]['final_pos'] = np.array([0, 0, -z_offset])
    
    # Añadimos los modelos como wireframes
    for config in group_configs:
        config['initial_center'] = np.mean([mesh.center for mesh in config['meshes']], axis=0)
        config['actors']=[plotter.add_mesh(mesh, style='wireframe', color=config['color'], line_width=2) for mesh in config['meshes']]
    
    # Obtenemos los tamaños de los modelos y posicionamos las etiquetas
    combined_mesh = pv.MultiBlock(kept_meshes).combine()
    model_bounds = combined_mesh.bounds
    model_size = np.linalg.norm([model_bounds[1]-model_bounds[0], model_bounds[3]-model_bounds[2], model_bounds[5]-model_bounds[4]])
    x_pos_labels = (model_bounds[0] - model_size*0.9)
    padding = model_size * 0.05

    coords = []
    for config in group_configs:
        #Hacemos mas pequeños los modelos de cada grupo
        label_mesh = pv.read(config['label_file'])
        scale_factor = model_size / (np.linalg.norm(np.array(label_mesh.bounds[1::2]) - np.array(label_mesh.bounds[0::2])) * 2)
        valz = config['initial_center'][2]
        static_pos = np.array([x_pos_labels, 0, valz])
        print(f"Scale factor {scale_factor}")
        label_mesh.scale(scale_factor, inplace=True)
        initial_label_pos = np.array([static_pos[0] - label_mesh.bounds[1], 0 - label_mesh.center[1], valz])
        label_mesh.translate(initial_label_pos, inplace=True)
        config['label_mesh'] = label_mesh
        config['label_actor'] = plotter.add_mesh(label_mesh, color='#FFFFFF')

        bounds = label_mesh.bounds
        line_start_pos = np.array([bounds[1], (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2])
        vector = config['initial_center'] - line_start_pos
        direction = vector / np.linalg.norm(vector)
        coord1 = line_start_pos + direction * padding
        print(f"Coord {coord1}")
        coord2 = config['initial_center'] - direction * padding
        line = pv.Line(coord1, coord2)
        coords.append([coord1, coord2])
        config['line_mesh'] = line
        line_actor = plotter.add_mesh(line, color='white')
        line_actor.prop.line_width = 2

    #Agregamos la lineas de los ejes
    axis_length = model_size * 0.7
    arrowhead_size = axis_length * 0.1
    axis_line_width = 3

    x_axis_end = np.array([axis_length, 0, 0])
    plotter.add_mesh(pv.Line([0, 0, 0], x_axis_end), color='white', line_width=axis_line_width)
    
    x_arrow_p1 = np.array([axis_length - arrowhead_size, arrowhead_size, 0])
    x_arrow_p2 = np.array([axis_length - arrowhead_size, -arrowhead_size, 0])
    
    plotter.add_mesh(pv.Line(x_axis_end, x_arrow_p1), color='white', line_width=axis_line_width)
    plotter.add_mesh(pv.Line(x_axis_end, x_arrow_p2), color='white', line_width=axis_line_width)

    y_axis_end = np.array([0, axis_length, 0])
    plotter.add_mesh(pv.Line([0, 0, 0], y_axis_end), color='white', line_width=axis_line_width)

    y_arrow_p1 = np.array([arrowhead_size, axis_length - arrowhead_size, 0])
    y_arrow_p2 = np.array([-arrowhead_size, axis_length - arrowhead_size, 0])

    plotter.add_mesh(pv.Line(y_axis_end, y_arrow_p1), color='white', line_width=axis_line_width)
    plotter.add_mesh(pv.Line(y_axis_end, y_arrow_p2), color='white', line_width=axis_line_width)

    plotter.add_point_labels(
        points=[[axis_length + arrowhead_size, 0, 0], [0, axis_length + arrowhead_size, 0]],
        labels=['', ''],
        font_size=2,
        shape=None,
        text_color='white'
    )

    current_gesture = "Ninguno"
    selected_group_index = None
    animation_speed = 0.05       # Velocidad de la animación
    current_hand_angle = 90.0  

    plotter.show(interactive=False, auto_close=False)
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.1)

    while plotter.iren is not None and plotter.iren.initialized:
        try:
            data = gesture_queue.get_nowait()
            new_gesture,new_angle = data["gesture"], data["angle"]
            if new_gesture != current_gesture:
                current_gesture = new_gesture
                print(f"Gesto cambiado a: {current_gesture}")
            current_hand_angle = new_angle
        except Empty:
            pass

        match current_gesture:
            case Gestures.EXPANDED_HAND:
                selected_group_index = 0
            case Gestures.L_FINGERS:
                selected_group_index = 1
            case Gestures.THREE_FINGERS:
                selected_group_index = 2
            case _:
                selected_group_index = None

        for i, config in enumerate(group_configs):
            if i == selected_group_index:
                config['animation_t'] = min(1.0, config['animation_t'] + animation_speed)
            else:
                config['animation_t'] = max(0.0, config['animation_t'] - animation_speed)

            linear_t = config['animation_t']
            if i == selected_group_index:
                working_meshes = config['meshes']
                original_meshes_for_group = groups_of_meshes_orig[i]
                for mesh_idx, working_mesh in enumerate(working_meshes):
                    original_mesh = original_meshes_for_group[mesh_idx]
                    working_mesh.points = original_mesh.points.copy()

                for mesh in working_meshes:
                    mesh.rotate_z(1.0, point=config['initial_center'], inplace=True)
                #final_t = cubic_bezier_easing(linear_t)
                final_t = cubic_bezier_ease_in_out(linear_t)
            else:
                final_t = linear_t
                if 'last_known_hand_angle' in config:
                    del config['last_known_hand_angle']
            
            first_actor_position = config['actors'][0].position
            #t = config['animation_t']
            #first_actor_position = config['actors'][0].position
            
            #if t > 0.001 or not np.allclose(first_actor_position, [0, 0, 0], atol=1e-3):
            if linear_t > 0.001 or not np.allclose(first_actor_position, [0, 0, 0], atol=1e-3):
                current_displacement = config['final_pos'] * final_t
                
                for actor in config['actors']:
                    actor.position = current_displacement

                label_actor = config['label_actor']
                line_mesh = config['line_mesh']
                
                label_displacement = np.array([0, 0, current_displacement[2]])
                label_actor.position = label_displacement
                
                group_current_center = config['initial_center'] + current_displacement
                label_bounds = label_actor.bounds
                current_line_start = np.array([label_bounds[1], (label_bounds[2] + label_bounds[3]) / 2, (label_bounds[4] + label_bounds[5]) / 2])
                
                vector = group_current_center - current_line_start
                if np.linalg.norm(vector) > 1e-6:
                    direction = vector / np.linalg.norm(vector)
                    line_mesh.points[0] = current_line_start + direction * padding
                    line_mesh.points[1] = group_current_center - direction * padding

        plotter.render()
        plotter.iren.process_events()

    print("Cerrando...")
    detector_proc.terminate()
    plotter.close()

if __name__ == "__main__":
    freeze_support()
    main_app()
