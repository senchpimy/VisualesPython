import moderngl
import numpy as np
import pyglet
import sys
import threading
import queue
import time
import random

vertex_shader = """
#version 330
in vec2 in_vert;
void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

fragment_shader = """
#version 330
#define MAX_STEPS 128

uniform vec2 iResolution;
uniform mat4 view_rotation;
uniform mat4 object_rotation;

out vec4 fragColor;

float ring1(vec2 p){
    float size = 0.45;
    float thickness = 0.02;
    float deg = 140.;
    float d = abs(length(p) - size) - thickness;
    float a = radians(deg);
    d = max(dot(p, vec2(cos(a), sin(a))), d);
    a = radians(-deg);
    d = max(dot(p, vec2(cos(a), sin(a))), d);
    return d;
}

float GetDist(vec3 p) {
    vec3 rotated_p = (object_rotation * vec4(p, 1.0)).xyz;
    float d_2d = ring1(rotated_p.xy);
    float thickness_z = 0.03;
    float d = max(d_2d, abs(rotated_p.z) - thickness_z);
    return d;
}

vec3 RayMarch(vec3 ro, vec3 rd) {
    float t = 0.0;
    for(int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + t * rd;
        float d = GetDist(p);
        if (d < 0.001) {
            return vec3(0.1, 0.8, 1.0);
        }
        t += d;
        if (t > 20.0) {
            break;
        }
    }
    return vec3(0.05, 0.05, 0.1);
}

vec3 setup_camera(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 f = normalize(l - p);
    vec3 r = normalize(cross(vec3(0, 1, 0), f));
    vec3 u = cross(f, r);
    vec3 c = p + f * z;
    vec3 i = c + uv.x * r + uv.y * u;
    return normalize(i - p);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution) / iResolution.y;
    vec3 ro_initial = vec3(0.0, 0.0, -3.0);
    vec3 ro = (view_rotation * vec4(ro_initial, 1.0)).xyz;
    vec3 rd = setup_camera(uv, ro, vec3(0.0, 0.0, 0.0), 1.0);
    vec3 col = RayMarch(ro, rd);
    col = pow(col, vec3(1.0/2.2));
    fragColor = vec4(col, 1.0);
}
"""

import camera
from multiprocessing import Queue
cola = Queue()

class RenderWindow(pyglet.window.Window):
    def __init__(self, **kwargs):
        title = kwargs.pop("title", "ModernGL Window")
        super().__init__(**kwargs, resizable=True)
        self.set_caption(title)

        self.ctx = moderngl.create_context()
        self.prog = self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )
        vertices = np.array([-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
        self.vbo = self.ctx.buffer(vertices.astype("f4").tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, self.vbo, "in_vert")

        # Diccionario para almacenar las rotaciones
        self.rotations = {
            "view": {"x": 0.0, "y": 0.0, "z": 0.0},
            "object": {"x": 0.0, "y": 0.0, "z": 0.0},
        }

        self.lock = threading.Lock()
        self.running = True
        self.input_thread = threading.Thread(target=self.read_input_loop, daemon=True)
        self.input_thread.start()

        print("\n--- ¡Control en Tiempo Real Activado! ---")
        print("Escribe comandos en esta terminal y presiona Enter.")
        print("Ejemplos:")
        print("  vy 45   (Rotar VISTA 45 grados en Y)")
        print("  ox 90   (Rotar OBJETO 90 grados en X)")
        print("Comandos: [v/o][x/y/z] [grados] (ej: vx, vy, vz, ox, oy, oz)")
        print("----------------------------------------\n")

    def create_rotation_matrix(self, rx, ry, rz):
        rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
        cos_x, sin_x = np.cos(rx), np.sin(rx)
        cos_y, sin_y = np.cos(ry), np.sin(ry)
        cos_z, sin_z = np.cos(rz), np.sin(rz)

        rot_x = np.array(
            [[1, 0, 0, 0], [0, cos_x, -sin_x, 0], [0, sin_x, cos_x, 0], [0, 0, 0, 1]]
        )
        rot_y = np.array(
            [[cos_y, 0, sin_y, 0], [0, 1, 0, 0], [-sin_y, 0, cos_y, 0], [0, 0, 0, 1]]
        )
        rot_z = np.array(
            [[cos_z, -sin_z, 0, 0], [sin_z, cos_z, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        return (rot_y @ rot_x @ rot_z).astype("f4")

    def read_input_loop(self):
        while self.running:
            try:

                #line = sys.stdin.readline().strip().lower()
                line = cola.get()
                if not line or line is None:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    print(
                        f"Error: Comando '{line}' no válido. Usa formato 'eje valor', ej: vy 45"
                    )
                    continue
                cmd, value_str = parts
                value = float(value_str)
                target_map = {"v": "view", "o": "object"}
                if len(cmd) != 2 or cmd[0] not in target_map or cmd[1] not in "xyz":
                    print(f"Error: Eje '{cmd}' no reconocido.")
                    continue
                target, axis = target_map[cmd[0]], cmd[1]
                with self.lock:
                    self.rotations[target][axis] = value
                print(f"-> Rotación actualizada: {target} en {axis.upper()} a {value}°")
            except (ValueError, IndexError):
                print(
                    f"Error: Entrada no válida. Asegúrate de que el valor sea un número."
                )
            except Exception as e:
                if self.running:
                    print(f"Error inesperado en el hilo de entrada: {e}")

    def on_draw(self):
        self.clear()

        with self.lock:
            view_rot = self.rotations["view"]
            obj_rot = self.rotations["object"]

        view_matrix = self.create_rotation_matrix(
            view_rot["x"], view_rot["y"], view_rot["z"]
        )
        object_matrix = self.create_rotation_matrix(
            obj_rot["x"], obj_rot["y"], obj_rot["z"]
        )

        self.prog["iResolution"].value = tuple(self.get_size())
        self.prog["view_rotation"].write(view_matrix.tobytes())
        self.prog["object_rotation"].write(object_matrix.tobytes())

        self.vao.render(moderngl.TRIANGLE_STRIP)

    def on_close(self):
        print("Cerrando aplicación...")
        self.running = False
        super().on_close()


if __name__ == "__main__":
    threading.Thread(target=camera.get_data, args=(cola,), daemon=True).start()
    window = RenderWindow(width=800, height=600, title="Aro Interactivo (ModernGL)")
    pyglet.app.run()


