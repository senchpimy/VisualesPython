import pyvista as pv
from pathlib import Path
import random

def add_blocks_recursively(plotter, data, color_list, color_index):
    """
    Añade mallas de forma recursiva, asignando un color único a cada malla final.
    """
    if isinstance(data, pv.DataSet) and data.n_points > 0:
        color = color_list[color_index[0] % len(color_list)]
        color_index[0] += 1
        
        plotter.add_mesh(data, color=color, smooth_shading=True)
        return

    if isinstance(data, pv.MultiBlock):
        for i in range(data.n_blocks):
            block = data[i]
            add_blocks_recursively(plotter, block, color_list, color_index)

model_path = Path("./model/scene.gltf")
if not model_path.exists():
    raise FileNotFoundError(f"El archivo del modelo no se encontró en: {model_path}")

print(f"Cargando modelo desde: {model_path}...")
data = pv.read(model_path)

plotter = pv.Plotter(window_size=[1280, 720])
plotter.window_title = "Visor de Estructura GLTF (Colores Aleatorios)"
plotter.background_color = 'black'


colores = [
    '#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD', 
    '#FF4500', '#20B2AA', '#ADFF2F', '#DA70D6', '#00FFFF',
    '#F08080', '#7B68EE', '#98FB98', '#FFB6C1', '#87CEEB'
]

color_counter = [0] 

add_blocks_recursively(plotter, data, colores, color_counter)

plotter.show()

