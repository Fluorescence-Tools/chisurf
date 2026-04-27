from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

try:  # Use PyOpenGL for raw GL entry points just like pyqtgraph's QGLWidget
    from OpenGL import GL  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    GL = None

from ..config import _DISPLAY_CONFIG
from .base import Renderer
from .scene import Geometry, Scene, SceneObject


# Minimal set of OpenGL enum values used by this renderer. These are
# stable across OpenGL versions, so we can hard-code them and avoid an
# additional PyOpenGL dependency while still driving Qt's GL functions.
GL_TRIANGLES = 0x0004
GL_LINES = 0x0001
GL_POINTS = 0x0000
GL_LINE_STRIP = 0x0003
GL_POINT_SPRITE = 0x8861

GL_DEPTH_TEST = 0x0B71
GL_CULL_FACE = 0x0B44
GL_BACK = 0x0405

GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100

GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303

GL_PROGRAM_POINT_SIZE = 0x8642
GL_FLOAT = 0x1406


def _invoke_menu(menu: QtWidgets.QMenu, pos: QtCore.QPoint):
    exec_fn = getattr(menu, "exec", None) or getattr(menu, "exec_", None)
    if exec_fn is None:
        raise AttributeError("QMenu.exec/exec_ not available")
    return exec_fn(pos)


@dataclass
class _DrawData:
    primitive: int
    positions: np.ndarray
    colors: np.ndarray
    normals: np.ndarray
    render_mode: str
    size: float = 4.0
    width: float = 2.0
    depth_test: bool = True
    glyph: Optional[str] = None
    radii: Optional[np.ndarray] = None
    material: Optional[dict] = None

@dataclass
class _LabelData:
    pos: np.ndarray
    text: str
    color: QtGui.QColor
    depth_test: bool = True


@dataclass
class _GpuDrawCall:
    primitive: int
    vertex_count: int
    positions_vbo: QtGui.QOpenGLBuffer
    colors_vbo: QtGui.QOpenGLBuffer
    normals_vbo: QtGui.QOpenGLBuffer
    render_mode: str
    size: float
    width: float
    depth_test: bool
    glyph: Optional[str] = None
    radii_vbo: Optional[QtGui.QOpenGLBuffer] = None
    material: Optional[dict] = None

class QtGLRenderer(QtWidgets.QOpenGLWidget, Renderer):
    """Qt-native OpenGL renderer with lightweight VBO caching.

    The controller pushes :class:`Scene` updates via :meth:`set_scene`; we
    translate them into GPU buffers and render with a single simple shader.
    """

    def __init__(
        self,
        controller,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        Renderer.__init__(self, parent=parent)
        QtWidgets.QOpenGLWidget.__init__(self, parent)
        self._gl = None
        self._controller = controller
        self._scene: Optional[Scene] = None
        self._draw_data: list[_DrawData] = []
        self._gpu_calls: list[_GpuDrawCall] = []
        self._labels: list[_LabelData] = []
        self._needs_upload: bool = False
        self._program: Optional[QtGui.QOpenGLShaderProgram] = None
        self._pos_attr = -1
        self._color_attr = -1
        self._normal_attr = -1
        self._mvp_uniform = -1
        self._normal_matrix_uniform = -1
        self._view_matrix_uniform = -1
        self._light_dir_uniform = -1
        self._ambient_uniform = -1
        self._spec_strength_uniform = -1
        self._shininess_uniform = -1
        self._rim_strength_uniform = -1
        self._rim_power_uniform = -1
        self._fog_density_uniform = -1
        self._fog_color_uniform = -1
        self._glyph_mode_uniform = -1
        self._point_size_uniform = -1
        self._radius_attr = -1
        self._background = (0.0, 0.0, 0.0, 1.0)

        lighting_cfg = (_DISPLAY_CONFIG.get("lighting") or {})
        light_dir = lighting_cfg.get("light_direction", [0.0, 0.0, 1.0])
        self._light_direction = QtGui.QVector3D(
            float(light_dir[0]), float(light_dir[1]), float(light_dir[2])
        )
        self._ambient_strength = float(lighting_cfg.get("ambient_strength", 0.55))
        self._specular_strength = float(lighting_cfg.get("specular_strength", 0.18))
        self._shininess = float(lighting_cfg.get("shininess", 38.0))
        self._rim_strength = float(lighting_cfg.get("rim_strength", 0.18))
        self._rim_power = float(lighting_cfg.get("rim_power", 2.4))
        self._fog_density = 0.0
        self._grid_visible = False
        self._grid_size = 20.0
        self._grid_spacing = 1.0
        self._distance = 30.0
        self._elevation = 20.0
        self._azimuth = 45.0
        self._target_radius = 10.0
        self._near_clip = 0.1
        self._far_clip = 1000.0
        self._min_near_clip = 0.01
        self._max_near_clip = 10.0
        self._clip_wheel_scale = 0.85
        self._grid_draw_data: Optional[_DrawData] = None
        self._opts = {
            "center": QtGui.QVector3D(0.0, 0.0, 0.0),
            "fov": 45.0,
        }
        self.opts = self._opts  # Compatibility with picking helpers
        self._drag_selecting = False
        self._drag_start: Optional[QtCore.QPoint] = None
        self._rubber_band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, self)
        self._drag_modifiers = QtCore.Qt.NoModifier
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self._panning = False
        self._pan_offset = np.zeros(3, dtype=float)

    # ------------------------------------------------------------------
    # Renderer interface
    # ------------------------------------------------------------------
    def widget(self) -> QtWidgets.QWidget:
        return self

    def set_scene(self, scene: Optional[Scene]) -> None:
        """Attach a Scene, rebuild VBOs, and request a repaint."""
        self._scene = scene
        self._prepare_draw_data(scene)
        self._update_center_opt()
        self.update()

    def clear(self) -> None:
        self._scene = None
        self._draw_data = []
        self._release_gpu_calls()
        self.update()

    def configure_grid(self, size: float, spacing: float) -> None:
        self._grid_size = float(size)
        self._grid_spacing = max(float(spacing), 0.1)
        self._grid_draw_data = None
        self.update()

    def set_background_color(self, color) -> None:
        if isinstance(color, (tuple, list)) and len(color) >= 3:
            rgba = tuple(float(c) for c in color[:4]) if len(color) >= 4 else (*color[:3], 1.0)
        else:
            qcolor = QtGui.QColor(color)
            rgba = (
                qcolor.redF(),
                qcolor.greenF(),
                qcolor.blueF(),
                qcolor.alphaF(),
            )
        self._background = rgba
        self.update()

    def set_grid_visible(self, visible: bool) -> None:
        self._grid_visible = bool(visible)
        self._grid_draw_data = None
        self._prepare_draw_data(self._scene)
        self.update()

    def configure_camera(
        self,
        *,
        near_clip: float,
        far_clip: float,
        min_near_clip: float,
        max_near_clip: float,
        clip_wheel_scale: float,
    ) -> None:
        self._min_near_clip = max(float(min_near_clip), 1e-5)
        self._max_near_clip = max(float(max_near_clip), self._min_near_clip * 1.01)
        self._clip_wheel_scale = clip_wheel_scale if 0.0 < clip_wheel_scale < 1.0 else 0.85
        self._near_clip = self._clamp_near_clip(float(near_clip))
        far_val = max(float(far_clip), self._near_clip * 10.0)
        self._far_clip = far_val
        self.update()

    def fit_to_radius(self, radius: float) -> None:
        self._target_radius = max(float(radius), 1.0)
        self._distance = max(self._target_radius * 3.0, 5.0)

        target_near = max(self._target_radius * 0.02, self._min_near_clip)
        self._near_clip = self._clamp_near_clip(target_near)

        max_extent = self._distance + self._target_radius
        far_min = self._near_clip * 10.0
        self._far_clip = max(float(max_extent) * 1.2, far_min)

        self.update()

    def reset_view(self, distance: float, elevation: float, azimuth: float) -> None:
        self._distance = max(float(distance), 1.0)
        self._elevation = float(elevation)
        self._azimuth = float(azimuth)
        self._pan_offset = np.zeros(3, dtype=float)
        self._update_center_opt()
        self.update()

    def look_at(self, target: np.ndarray) -> None:
        """Set the camera to look at the given world-space coordinate."""
        center = np.zeros(3, dtype=float)
        if self._scene is not None:
            try:
                center = np.asarray(self._scene.center, dtype=float)
            except Exception:
                center = np.zeros(3, dtype=float)
        self._pan_offset = np.asarray(target, dtype=float) - center
        self._update_center_opt()
        self.update()

    def set_distance(self, distance: float) -> None:
        """Set the distance from the target point."""
        self._distance = max(float(distance), 0.1)
        self.update()

    def set_orientation(self, elevation: float, azimuth: float) -> None:
        """Set camera elevation and azimuth."""
        self._elevation = float(elevation)
        self._azimuth = float(azimuth)
        self.update()

    # Compatibility helpers -------------------------------------------------
    def cameraPosition(self) -> QtGui.QVector3D:
        pos = self._camera_position()
        return QtGui.QVector3D(float(pos[0]), float(pos[1]), float(pos[2]))

    # ------------------------------------------------------------------
    # Qt OpenGL overrides
    # ------------------------------------------------------------------
    def initializeGL(self) -> None:
        if GL is None:
            self._notify_gl_failure(
                "PyOpenGL is required for the Qt renderer. Install PyOpenGL first."
            )
            return

        self._gl = GL
        gl = GL
        gl.glEnable(GL_DEPTH_TEST)
        gl.glEnable(GL_CULL_FACE)
        gl.glCullFace(GL_BACK)
        gl.glEnable(GL_PROGRAM_POINT_SIZE)
        gl.glEnable(GL_POINT_SPRITE)
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        self._create_program()

    def resizeGL(self, width: int, height: int) -> None:
        gl = self._gl
        if gl is None:
            return
        gl.glViewport(0, 0, width, max(height, 1))

    def paintGL(self) -> None:
        if GL is None:
            return

        gl = self._gl
        if gl is None:
            return

        r, g_col, b, a = self._background
        gl.glClearColor(r, g_col, b, a)
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        if self._program is None:
            return

        if not self._gpu_calls and self._draw_data:
            self._upload_draw_data()
        elif self._needs_upload:
            self._release_gpu_calls()
            self._upload_draw_data()
            self._needs_upload = False

        if not self._gpu_calls:
            return

        mvp, view = self._build_matrices()
        normal_matrix = QtGui.QMatrix4x4(view)
        normal_matrix.setColumn(3, QtGui.QVector4D(0.0, 0.0, 0.0, 1.0))
        self._program.bind()
        self._program.setUniformValue(self._mvp_uniform, mvp)
        self._program.setUniformValue(self._normal_matrix_uniform, normal_matrix)
        self._program.setUniformValue(self._view_matrix_uniform, view)
        self._program.setUniformValue(self._light_dir_uniform, self._light_direction)
        self._program.setUniformValue(self._ambient_uniform, float(self._ambient_strength))
        self._program.setUniformValue(self._spec_strength_uniform, float(self._specular_strength))
        self._program.setUniformValue(self._shininess_uniform, float(self._shininess))
        self._program.setUniformValue(self._rim_strength_uniform, float(self._rim_strength))
        self._program.setUniformValue(self._rim_power_uniform, float(self._rim_power))
        self._program.setUniformValue(self._fog_density_uniform, float(self._fog_density))
        fog_color = QtGui.QVector3D(float(r), float(g_col), float(b))
        self._program.setUniformValue(self._fog_color_uniform, fog_color)

        for call in self._gpu_calls:
            if call.render_mode == "overlay":
                gl.glDisable(GL_DEPTH_TEST)
            else:
                gl.glEnable(GL_DEPTH_TEST)
            if call.render_mode == "transparent":
                gl.glEnable(GL_BLEND)
                gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            else:
                gl.glDisable(GL_BLEND)

            glyph_mode = 1 if (call.glyph == "sphere" and call.primitive == GL_POINTS) else 0
            self._program.setUniformValue(self._glyph_mode_uniform, glyph_mode)
            self._program.setUniformValue(self._point_size_uniform, float(call.size))

            mat = call.material or {}
            self._program.setUniformValue(self._spec_strength_uniform, float(mat.get("specular_strength", self._specular_strength)))
            self._program.setUniformValue(self._shininess_uniform, float(mat.get("shininess", self._shininess)))
            self._program.setUniformValue(self._rim_strength_uniform, float(mat.get("rim_strength", self._rim_strength)))
            self._program.setUniformValue(self._rim_power_uniform, float(mat.get("rim_power", self._rim_power)))

            if call.primitive == GL_POINTS:
                gl.glPointSize(call.size)
                if glyph_mode:
                    gl.glEnable(GL_POINT_SPRITE)
                else:
                    gl.glDisable(GL_POINT_SPRITE)
            elif call.primitive == GL_LINES:
                gl.glLineWidth(call.width)
                gl.glDisable(GL_POINT_SPRITE)
            else:
                gl.glDisable(GL_POINT_SPRITE)

            call.positions_vbo.bind()
            self._program.enableAttributeArray(self._pos_attr)
            self._program.setAttributeBuffer(
                self._pos_attr,
                GL_FLOAT,
                0,
                3,
            )
            call.colors_vbo.bind()
            self._program.enableAttributeArray(self._color_attr)
            self._program.setAttributeBuffer(
                self._color_attr,
                GL_FLOAT,
                0,
                4,
            )

            call.normals_vbo.bind()
            self._program.enableAttributeArray(self._normal_attr)
            self._program.setAttributeBuffer(
                self._normal_attr,
                GL_FLOAT,
                0,
                3,
            )

            if self._radius_attr != -1:
                if call.radii_vbo is not None:
                    call.radii_vbo.bind()
                    self._program.enableAttributeArray(self._radius_attr)
                    self._program.setAttributeBuffer(self._radius_attr, GL_FLOAT, 0, 1)
                else:
                    self._program.disableAttributeArray(self._radius_attr)
                    self._program.setAttributeValue(self._radius_attr, 0.0)

            gl.glDrawArrays(call.primitive, 0, call.vertex_count)

            if call.primitive == GL_POINTS:
                gl.glPointSize(1.0)
                gl.glDisable(GL_POINT_SPRITE)

            call.positions_vbo.release()
            call.colors_vbo.release()
            call.normals_vbo.release()
            if call.radii_vbo is not None:
                call.radii_vbo.release()
            self._program.disableAttributeArray(self._pos_attr)
            self._program.disableAttributeArray(self._color_attr)
            self._program.disableAttributeArray(self._normal_attr)
            if self._radius_attr != -1:
                self._program.disableAttributeArray(self._radius_attr)

        # Restore global material uniforms so we don't accidentally leak state into next frame
        self._program.setUniformValue(self._spec_strength_uniform, float(self._specular_strength))
        self._program.setUniformValue(self._shininess_uniform, float(self._shininess))
        self._program.setUniformValue(self._rim_strength_uniform, float(self._rim_strength))
        self._program.setUniformValue(self._rim_power_uniform, float(self._rim_power))

        self._program.release()
        self._render_labels()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _notify_gl_failure(self, message: str) -> None:
        if self._controller is not None:
            handler = getattr(self._controller, "on_renderer_error", None)
            if callable(handler):
                handler(message)
                return
        raise RuntimeError(message)

    def _create_program(self) -> None:
        vert_src = """#version 120
        attribute vec3 position;
        attribute vec4 color;
        attribute vec3 normal;
        attribute float radius;
        uniform mat4 mvp;
        uniform mat4 normalMatrix;
        uniform mat4 viewMatrix;
        varying vec4 v_color;
        varying vec3 v_normal;
        varying vec3 v_viewPos;
        uniform int glyphMode;
        uniform float pointSize;
        void main() {
            vec4 worldPos = vec4(position, 1.0);
            gl_Position = mvp * worldPos;
            v_color = color;
            v_normal = normalize((normalMatrix * vec4(normal, 0.0)).xyz);
            vec4 viewPos = viewMatrix * worldPos;
            v_viewPos = viewPos.xyz;
            if (glyphMode != 0) {
                gl_PointSize = (radius > 0.0) ? radius : pointSize;
            }
        }
        """

        frag_src = """#version 120
        varying vec4 v_color;
        varying vec3 v_normal;
        varying vec3 v_viewPos;
        uniform vec3 lightDir;
        uniform float ambientStrength;
        uniform float specStrength;
        uniform float shininess;
        uniform float rimStrength;
        uniform float rimPower;
        uniform float fogDensity;
        uniform vec3 fogColor;
        uniform int glyphMode;
        void main() {
            vec3 n = normalize(v_normal);
            if (glyphMode == 1) {
                vec2 coord = gl_PointCoord * 2.0 - 1.0;
                float dist2 = dot(coord, coord);
                if (dist2 > 1.0) {
                    discard;
                }
                float z = sqrt(max(0.0, 1.0 - dist2));
                n = normalize(vec3(coord, z));
            }
            vec3 l = normalize(lightDir);
            vec3 viewDir = normalize(-v_viewPos);
            float lambert = max(dot(n, l), 0.0);
            float lighting = ambientStrength + (1.0 - ambientStrength) * lambert;

            float spec = 0.0;
            if (lambert > 0.0) {
                vec3 reflectDir = reflect(-l, n);
                spec = pow(max(dot(viewDir, reflectDir), 0.0), shininess) * specStrength;
            }

            vec3 baseColor = v_color.rgb;
            
            // Fresnel for jelly/bubble look: edges are more opaque and reflective
            float fresnel = pow(clamp(1.0 - dot(n, viewDir), 0.0, 1.0), 2.5);
            
            // 1. Shading (Diffuse + Ambient)
            // Use a slightly lower ambient to make rim and reflections pop
            float ambient = ambientStrength * 0.7;
            float diffuse = (1.0 - ambient) * lambert;
            vec3 shaded = baseColor * (ambient + diffuse);
            
            // 2. Rim lighting (edge glow)
            float rim = pow(clamp(1.0 - dot(n, viewDir), 0.0, 1.0), rimPower);
            shaded += baseColor * rim * rimStrength;
            
            // 3. Procedural Environment Reflection (Fake MatCap)
            vec3 R = reflect(-viewDir, n);
            vec3 skyCol = vec3(0.5, 0.7, 1.0);
            vec3 groundCol = vec3(0.05, 0.05, 0.1);
            vec3 envReflection = mix(groundCol, skyCol, smoothstep(-0.2, 0.4, R.y));
            
            // Add a "sun" highlight
            vec3 sunDir = normalize(vec3(0.5, 1.0, 0.5)); 
            float sun = pow(max(0.0, dot(R, sunDir)), max(10.0, shininess));
            envReflection += vec3(1.5) * sun;
            
            // Blend reflection based on Fresnel
            float reflectMul = clamp(specStrength * (0.1 + 0.6 * fresnel), 0.0, 1.0);
            vec3 finalColor = mix(shaded, envReflection, reflectMul);
            
            // Point-source specular highlight
            finalColor += spec * vec3(1.0);
            
            // 4. Fog
            float fogFactor = clamp(1.0 - exp(-fogDensity * length(v_viewPos)), 0.0, 1.0);
            finalColor = mix(finalColor, fogColor, fogFactor);

            // 5. Transparency with Fresnel
            float finalAlpha = v_color.a;
            if (finalAlpha < 0.99) {
                finalAlpha = mix(finalAlpha * 0.3, clamp(finalAlpha + 0.5, 0.0, 1.0), fresnel);
            }
            
            gl_FragColor = vec4(finalColor, clamp(finalAlpha + spec * 0.3 + sun * 0.3, 0.0, 1.0));
        }
        """

        program = QtGui.QOpenGLShaderProgram(self.context())
        program.addShaderFromSourceCode(QtGui.QOpenGLShader.Vertex, vert_src)
        program.addShaderFromSourceCode(QtGui.QOpenGLShader.Fragment, frag_src)
        program.link()
        log = program.log()
        if log:
            print("QtGLRenderer shader log:\n", log)
        self._program = program
        self._pos_attr = program.attributeLocation("position")
        self._color_attr = program.attributeLocation("color")
        self._normal_attr = program.attributeLocation("normal")
        self._mvp_uniform = program.uniformLocation("mvp")
        self._normal_matrix_uniform = program.uniformLocation("normalMatrix")
        self._view_matrix_uniform = program.uniformLocation("viewMatrix")
        self._light_dir_uniform = program.uniformLocation("lightDir")
        self._ambient_uniform = program.uniformLocation("ambientStrength")
        self._spec_strength_uniform = program.uniformLocation("specStrength")
        self._shininess_uniform = program.uniformLocation("shininess")
        self._rim_strength_uniform = program.uniformLocation("rimStrength")
        self._rim_power_uniform = program.uniformLocation("rimPower")
        self._fog_density_uniform = program.uniformLocation("fogDensity")
        self._fog_color_uniform = program.uniformLocation("fogColor")
        self._glyph_mode_uniform = program.uniformLocation("glyphMode")
        self._point_size_uniform = program.uniformLocation("pointSize")
        self._radius_attr = program.attributeLocation("radius")

    def _render_labels(self) -> None:
        if not self._labels:
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)

        mvp, _ = self._build_matrices()
        w = self.width()
        h = self.height()

        for label in self._labels:
            # Project 3D to 2D
            pos4 = QtGui.QVector4D(float(label.pos[0]), float(label.pos[1]), float(label.pos[2]), 1.0)
            clip_pos = mvp * pos4
            if clip_pos.w() <= 0:
                continue
            
            ndc = QtGui.QVector3D(clip_pos.x() / clip_pos.w(), clip_pos.y() / clip_pos.w(), clip_pos.z() / clip_pos.w())
            if ndc.x() < -1 or ndc.x() > 1 or ndc.y() < -1 or ndc.y() > 1 or ndc.z() < -1 or ndc.z() > 1:
                continue
            
            win_x = (ndc.x() + 1.0) * 0.5 * w
            win_y = (1.0 - ndc.y()) * 0.5 * h
            
            painter.setPen(label.color)
            # Draw shadow for readability
            painter.setPen(QtGui.QColor(0, 0, 0, 150))
            painter.drawText(int(win_x) + 1, int(win_y) + 1, label.text)
            painter.setPen(label.color)
            painter.drawText(int(win_x), int(win_y), label.text)

        painter.end()

    def _build_matrices(self) -> tuple[QtGui.QMatrix4x4, QtGui.QMatrix4x4]:
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        aspect = width / float(height)
        proj = QtGui.QMatrix4x4()
        proj.perspective(45.0, aspect, self._near_clip, self._far_clip)

        view = QtGui.QMatrix4x4()
        view.translate(0.0, 0.0, -self._distance)
        view.rotate(self._elevation, 1.0, 0.0, 0.0)
        view.rotate(self._azimuth, 0.0, 0.0, 1.0)
        center = np.zeros(3, dtype=float)
        if self._scene is not None:
            try:
                center = np.asarray(self._scene.center, dtype=float)
            except Exception:
                center = np.zeros(3, dtype=float)
        center = center + self._pan_offset
        view.translate(-center[0], -center[1], -center[2])
        return proj * view, view

    def _camera_position(self) -> np.ndarray:
        center = np.zeros(3, dtype=float)
        if self._scene is not None:
            try:
                center = np.asarray(self._scene.center, dtype=float)
            except Exception:
                center = np.zeros(3, dtype=float)
        center = center + self._pan_offset
        theta = math.radians(self._azimuth)
        phi = math.radians(self._elevation)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        rel_x = sin_theta * sin_phi * self._distance
        rel_y = -cos_theta * sin_phi * self._distance
        rel_z = cos_phi * self._distance
        return center + np.array([rel_x, rel_y, rel_z], dtype=float)

    def _prepare_draw_data(self, scene: Optional[Scene]) -> None:
        """Convert Scene objects into CPU-side arrays ready for VBO upload."""
        self._draw_data = []
        self._labels = []
        if scene is None:
            self._needs_upload = True
            return
        for obj in scene.objects:
            if obj.geometry.kind == "text":
                self._prepare_labels(obj)
                continue
            draw = self._geometry_to_draw_data(obj)
            if draw is not None:
                self._draw_data.append(draw)
        if self._grid_visible:
            grid_draw = self._build_grid_draw_data(scene.radius if scene else self._target_radius)
            if grid_draw is not None:
                self._draw_data.insert(0, grid_draw)
        self._needs_upload = True

    def _prepare_labels(self, obj: SceneObject) -> None:
        geom = obj.geometry
        labels = geom.meta.get("labels", [])
        positions = np.asarray(geom.positions, dtype=float)
        colors = geom.colors
        
        n = min(len(labels), positions.shape[0])
        for i in range(n):
            col = QtGui.QColor(255, 255, 255)
            if colors is not None and i < colors.shape[0]:
                c = colors[i]
                col = QtGui.QColor(int(c[0]*255), int(c[1]*255), int(c[2]*255), int(c[3]*255))
            
            self._labels.append(_LabelData(
                pos=positions[i],
                text=str(labels[i]),
                color=col,
                depth_test=obj.render_mode != "overlay"
            ))

    def _geometry_to_draw_data(self, obj: SceneObject) -> Optional[_DrawData]:
        geom = obj.geometry
        if geom.positions is None:
            return None
        positions = np.asarray(geom.positions, dtype=np.float32)
        if positions.size == 0:
            return None
        normals = None
        if geom.indices is not None:
            try:
                idx = np.asarray(geom.indices, dtype=np.int32).reshape(-1)
                positions = positions[idx]
                if geom.colors is not None:
                    colors = np.asarray(geom.colors, dtype=np.float32)
                    if colors.shape[0] == idx.max() + 1:
                        colors = colors[idx]
                    else:
                        colors = np.resize(colors, (positions.shape[0], colors.shape[1]))
                else:
                    colors = None
                if geom.normals is not None:
                    normals = np.asarray(geom.normals, dtype=np.float32)
                    if normals.shape[0] == idx.max() + 1:
                        normals = normals[idx]
                    else:
                        normals = np.resize(normals, (positions.shape[0], 3))
                else:
                    normals = None
            except Exception:
                colors = None
                normals = None
        else:
            colors = None
            if geom.normals is not None:
                normals = np.asarray(geom.normals, dtype=np.float32)

        if colors is None and geom.colors is not None:
            colors = np.asarray(geom.colors, dtype=np.float32)
            if colors.shape[0] != positions.shape[0]:
                if colors.shape[0] == 1:
                    colors = np.repeat(colors, positions.shape[0], axis=0)
                else:
                    colors = np.resize(colors, (positions.shape[0], colors.shape[1]))
        if colors is None:
            base_color = geom.meta.get("base_color") if geom.meta else None
            if isinstance(base_color, (tuple, list)) and len(base_color) >= 4:
                base = np.array(base_color[:4], dtype=np.float32)
            else:
                base = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            colors = np.tile(base, (positions.shape[0], 1))
        colors = colors.astype(np.float32, copy=False)

        if normals is None or normals.shape[0] != positions.shape[0]:
            normals = np.zeros_like(positions, dtype=np.float32)
            normals[:, 2] = 1.0
        else:
            normals = normals.astype(np.float32, copy=False)

        primitive = self._primitive_for_geometry(geom)
        if primitive is None:
            return None
        size = float(geom.meta.get("size", geom.meta.get("radius", 4.0))) if geom.meta else 4.0
        width = float(geom.meta.get("width", 2.0)) if geom.meta else 2.0
        depth_test = obj.render_mode != "overlay"
        glyph = None
        if geom.meta:
            glyph = geom.meta.get("glyph")

        return _DrawData(
            primitive=primitive,
            positions=positions,
            colors=colors,
            normals=normals,
            render_mode=obj.render_mode,
            width=width,
            depth_test=depth_test,
            glyph=glyph,
            radii=np.asarray(geom.radii, dtype=np.float32) if geom.radii is not None else None,
            material=obj.material,
        )

    def _primitive_for_geometry(self, geom: Geometry) -> Optional[int]:
        kind = geom.kind.lower()
        if kind == "mesh":
            return GL_TRIANGLES
        if kind == "line":
            mode = (geom.meta or {}).get("mode", "lines")
            if mode == "line_strip":
                return GL_LINE_STRIP
            return GL_LINES
        if kind == "points":
            return GL_POINTS
        return None

    def _build_grid_draw_data(self, radius: float) -> Optional[_DrawData]:
        half = max(self._grid_size, radius * 1.2)
        spacing = self._grid_spacing
        if spacing <= 0.0:
            return None
        lines = []
        n = int(math.ceil(half / spacing))
        for i in range(-n, n + 1):
            x = i * spacing
            lines.append([[x, -half, 0.0], [x, half, 0.0]])
            lines.append([[-half, x, 0.0], [half, x, 0.0]])
        if not lines:
            return None
        positions = np.asarray(lines, dtype=np.float32).reshape(-1, 3)
        color = np.array([0.4, 0.4, 0.4, 1.0], dtype=np.float32)
        colors = np.tile(color, (positions.shape[0], 1))
        normals = np.zeros_like(positions, dtype=np.float32)
        normals[:, 2] = 1.0
        return _DrawData(
            primitive=GL_LINES,
            positions=positions,
            colors=colors,
            normals=normals,
            render_mode="overlay",
            width=1.0,
            depth_test=False,
        )

    def _upload_draw_data(self) -> None:
        """Upload cached draw data into VBOs (idempotent until scene changes)."""
        if self._gl is None or self._program is None:
            return
        self._release_gpu_calls()
        gl = self._gl
        for draw in self._draw_data:
            vbo_pos = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)
            vbo_pos.create()
            vbo_pos.bind()
            vbo_pos.allocate(draw.positions.tobytes(), draw.positions.nbytes)
            vbo_pos.release()

            vbo_col = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)
            vbo_col.create()
            vbo_col.bind()
            vbo_col.allocate(draw.colors.tobytes(), draw.colors.nbytes)
            vbo_col.release()

            vbo_norm = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)
            vbo_norm.create()
            vbo_norm.bind()
            vbo_norm.allocate(draw.normals.tobytes(), draw.normals.nbytes)
            vbo_norm.release()

            gpu_call = _GpuDrawCall(
                primitive=draw.primitive,
                vertex_count=draw.positions.shape[0],
                positions_vbo=vbo_pos,
                colors_vbo=vbo_col,
                normals_vbo=vbo_norm,
                render_mode=draw.render_mode,
                size=draw.size,
                width=draw.width,
                depth_test=draw.depth_test,
                glyph=draw.glyph,
                material=draw.material,
            )
            self._gpu_calls.append(gpu_call)

            if draw.radii is not None and draw.radii.size == draw.positions.shape[0]:
                vbo_rad = QtGui.QOpenGLBuffer(QtGui.QOpenGLBuffer.VertexBuffer)
                vbo_rad.create()
                vbo_rad.bind()
                vbo_rad.allocate(draw.radii.tobytes(), draw.radii.nbytes)
                vbo_rad.release()
                gpu_call.radii_vbo = vbo_rad

        self._needs_upload = False

    def _release_gpu_calls(self) -> None:
        for call in self._gpu_calls:
            if call.positions_vbo.isCreated():
                call.positions_vbo.destroy()
            if call.colors_vbo.isCreated():
                call.colors_vbo.destroy()
            if call.normals_vbo.isCreated():
                call.normals_vbo.destroy()
        self._gpu_calls = []

    def _update_center_opt(self) -> None:
        center = np.zeros(3, dtype=float)
        if self._scene is not None:
            try:
                center = np.asarray(self._scene.center, dtype=float)
            except Exception:
                center = np.zeros(3, dtype=float)
        center = center + self._pan_offset
        self._opts["center"] = QtGui.QVector3D(float(center[0]), float(center[1]), float(center[2]))

    # ------------------------------------------------------------------
    # Input handling (basic orbit controls)
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.RightButton:
            menu = QtWidgets.QMenu(self)
            reset_act = menu.addAction("Reset view")
            action = menu.exec_(event.globalPos())
            if action is reset_act and self._controller is not None:
                try:
                    self._controller.reset_view()
                except Exception:
                    pass
            event.accept()
            return

        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = True
            self._last_mouse_pos = event.pos()
            event.accept()
            return

        if event.button() == QtCore.Qt.LeftButton:
            mods = event.modifiers()
            if mods & QtCore.Qt.ShiftModifier:
                self._drag_selecting = True
                self._drag_start = event.pos()
                self._drag_modifiers = mods
                rect = QtCore.QRect(self._drag_start, QtCore.QSize(0, 0))
                self._rubber_band.setGeometry(rect)
                self._rubber_band.show()
                event.accept()
                return
            self._last_mouse_pos = event.pos()
            if self._controller is not None:
                try:
                    self._controller.handle_mouse_click(event)
                except Exception:
                    pass
            event.accept()
            return

        self._last_mouse_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._drag_selecting and self._drag_start is not None:
            rect = QtCore.QRect(self._drag_start, event.pos()).normalized()
            self._rubber_band.setGeometry(rect)
            event.accept()
            return

        if (event.buttons() & QtCore.Qt.MiddleButton) and self._last_mouse_pos is not None:
            delta = event.pos() - self._last_mouse_pos
            self._pan_from_delta(delta.x(), delta.y())
            self._last_mouse_pos = event.pos()
            event.accept()
            return

        if event.buttons() & QtCore.Qt.LeftButton and hasattr(self, "_last_mouse_pos"):
            delta = event.pos() - self._last_mouse_pos
            self._azimuth += delta.x() * 0.5
            self._elevation = (self._elevation + delta.y() * 0.5) % 360.0
            self._last_mouse_pos = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MiddleButton:
            self._panning = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        mods = event.modifiers()
        delta_steps = int(event.angleDelta().y() / 120.0)
        if mods & QtCore.Qt.ControlModifier:
            if delta_steps != 0:
                for _ in range(abs(delta_steps)):
                    if delta_steps > 0:
                        new_near = self._near_clip * self._clip_wheel_scale
                    else:
                        new_near = self._near_clip / self._clip_wheel_scale
                    self._near_clip = self._clamp_near_clip(new_near)
                self.update()
            event.accept()
            return

        delta = delta_steps
        if delta != 0:
            self._distance = max(self._distance * (1.0 - 0.1 * delta), 2.0)
            self.update()
        super().wheelEvent(event)

    def _clamp_near_clip(self, value: float) -> float:
        val = max(float(value), self._min_near_clip)
        val = min(val, self._max_near_clip)
        return val

    def _pan_from_delta(self, dx: float, dy: float) -> None:
        width = max(self.width(), 1)
        height = max(self.height(), 1)
        if width <= 0 or height <= 0:
            return
        fov = math.radians(float(self._opts.get("fov", 45.0)))
        half_tan = math.tan(fov / 2.0)
        if half_tan <= 0:
            return
        aspect = width / float(height)
        scale_y = 2.0 * self._distance * half_tan / height
        scale_x = scale_y * aspect

        right = self._camera_right_vector()
        up = self._camera_up_vector()

        shift = (-dx * scale_x) * right + (dy * scale_y) * up
        self._pan_offset += shift
        self._update_center_opt()
        self.update()

    def _camera_forward_vector(self) -> np.ndarray:
        az = math.radians(self._azimuth)
        el = math.radians(self._elevation)
        forward = np.array(
            [math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)],
            dtype=float,
        )
        norm = float(np.linalg.norm(forward))
        if norm <= 1e-8:
            return np.array([0.0, 0.0, -1.0], dtype=float)
        return forward / norm

    def _camera_right_vector(self) -> np.ndarray:
        forward = self._camera_forward_vector()
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        norm = float(np.linalg.norm(right))
        if norm <= 1e-8:
            world_up = np.array([0.0, 1.0, 0.0], dtype=float)
            right = np.cross(forward, world_up)
            norm = float(np.linalg.norm(right))
            if norm <= 1e-8:
                return np.array([1.0, 0.0, 0.0], dtype=float)
        return right / norm

    def _camera_up_vector(self) -> np.ndarray:
        right = self._camera_right_vector()
        forward = self._camera_forward_vector()
        up = np.cross(right, forward)
        norm = float(np.linalg.norm(up))
        if norm <= 1e-8:
            return np.array([0.0, 1.0, 0.0], dtype=float)
        return up / norm

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.LeftButton
            and self._drag_selecting
            and self._controller is not None
        ):
            rect = self._rubber_band.geometry()
            self._rubber_band.hide()
            self._drag_selecting = False
            self._drag_start = None
            if rect.width() > 2 and rect.height() > 2:
                try:
                    self._controller.handle_rect_selection(rect, self._drag_modifiers)
                except Exception:
                    pass
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # type: ignore[name-defined]
        handled = False
        if self._controller is not None:
            try:
                handled = bool(self._controller.handle_key_event(event))
            except Exception:
                handled = False
        if handled:
            return
        super().keyPressEvent(event)
