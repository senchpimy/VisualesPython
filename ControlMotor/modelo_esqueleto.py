import pyvista as pv
from pathlib import Path

model_path = Path("./model/scene.gltf")
if not model_path.exists():
    raise FileNotFoundError(f"El archivo del modelo no se encontró en: {model_path}")

print(f"Cargando modelo desde: {model_path}...")
mesh = pv.read(model_path)

plotter = pv.Plotter(window_size=[1280, 720])
plotter.window_title = "Visor Wireframe"
plotter.background_color = 'black'

print("Añadiendo malla al plotter en modo 'wireframe'...")
plotter.add_mesh(mesh, style='wireframe', color='white', line_width=1)

plotter.show()
