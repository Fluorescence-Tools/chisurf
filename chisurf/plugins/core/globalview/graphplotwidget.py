import numpy as np
import pyqtgraph as pg
from qtpy import QtCore
from qtpy.QtWidgets import QGraphicsLineItem

clickedPen = pg.mkPen('m', width=4)

# Node type constants — must match wizard.py numbering
_NODE_FIT = 0
_NODE_PARAM_FIXED = 1
_NODE_PARAM_LINKED = 2
_NODE_PARAM_FREE = 3
_PARAM_TYPES = {_NODE_PARAM_FIXED, _NODE_PARAM_LINKED, _NODE_PARAM_FREE}


class GraphPlotWidget(pg.GraphItem):

    linkRequested = QtCore.Signal(int, int)
    linkRemovalRequested = QtCore.Signal(int)

    @property
    def selected_nodes_idx(self) -> list[int]:
        return [n.index() for n in self.selectedNodes]

    def __init__(
            self,
            update_callback: callable = None,
            n_selected_nodes: int = 2,
            *args,
            **kwargs
    ):
        self.dragPoint = None
        self.dragOffset = None
        self.selectedNodes = list()

        self.n_selected_nodes = n_selected_nodes
        self.update_callback = update_callback

        self.textItems = list()
        self.arrows = list()
        self.fit_circles = list()

        self.node_types = None
        self.fit_to_params = None
        self.tempLine = None
        self.dragStartIdx = -1
        self.dragStartPos = None
        self.dragFromParameter = False
        self.dragGroup = list()
        self.origPositions = None

        super(GraphPlotWidget, self).__init__(*args, **kwargs)
        self.scatter.sigClicked.connect(self.handle_click)

    def setData(self, **kwds):
        self.node_types = kwds.pop('node_types', None)
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        if 'adj' in self.data and self.node_types is not None:
            self.fit_to_params = {}
            for a, b in self.data['adj']:
                if self.node_types[a] == _NODE_FIT and self.node_types[b] != _NODE_FIT:
                    self.fit_to_params.setdefault(a, []).append(b)
                elif self.node_types[b] == _NODE_FIT and self.node_types[a] != _NODE_FIT:
                    self.fit_to_params.setdefault(b, []).append(a)
        self.setTexts(self.text)
        if 'adj' in self.data:
            self.setArrows(self.data['adj'])
        self.updateGraph()

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def setArrows(self, connections):
        for _ in connections:
            arrow = pg.ArrowItem()
            arrow.setParentItem(self)
            arrow_style = {
                'angle': 0,
                'baseAngle': -30,
                'tipAngle': 30,
                'headLen': 10,
                'headWidth': 4,
                'tailLen': None,
                'pxMode': True,
                'pen': {
                    'color': 'w',
                    'width': 1
                }
            }
            arrow.setStyle(**arrow_style)
            self.arrows.append(arrow)

    def updateGraph(self):
        # print("def updateGraph(self):")
        if len(self.data) == 0:
            return

        # Update test
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

        # Update arrows (angles)
        # print("self.data['pos']:", self.data['pos'])
        for i, item in enumerate(self.arrows):
            v0 = self.data['pos'][self.data['adj'][i][0]]
            v1 = self.data['pos'][self.data['adj'][i][1]]
            d = v1 - v0
            n = d / np.linalg.norm(d)
            angle = np.arccos(n @ np.array([1.0, 0.0], dtype=np.float64))
            if d[1] > 0:
                angle *= -1
            angle_deg = np.degrees(angle)
            item.setPos(*v0)
            item.setStyle(angle=angle_deg)

    def _is_parameter(self, idx: int) -> bool:
        if self.node_types is None or idx >= len(self.node_types):
            return False
        return self.node_types[idx] in _PARAM_TYPES

    def _show_temp_line(self, x1, y1, x2, y2):
        self._hide_temp_line()
        line = QGraphicsLineItem(x1, y1, x2, y2)
        line.setPen(pg.mkPen('y', width=2, style=QtCore.Qt.DashLine))
        line.setParentItem(self)
        self.tempLine = line

    def _hide_temp_line(self):
        if self.tempLine is not None:
            if self.tempLine.scene() is not None:
                self.tempLine.scene().removeItem(self.tempLine)
            self.tempLine = None

    def mouseDragEvent(self, ev, dynamic_drag_nodes: bool = None):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return

        if ev.isStart():
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos
            self.dragStartIdx = ind
            self.dragStartPos = self.data['pos'][ind].copy()
            self.origPositions = self.data['pos'].copy()
            self.dragFromParameter = self._is_parameter(ind)

            if self.dragFromParameter:
                sx, sy = self.dragStartPos
                self._show_temp_line(sx, sy, sx, sy)
            else:
                self.dragGroup = []
                if self.fit_to_params is not None:
                    for p in self.fit_to_params.get(ind, []):
                        self.dragGroup.append(p)

            ev.accept()
            return

        elif ev.isFinish():
            self._hide_temp_line()

            if self.dragPoint is None:
                ev.ignore()
                return

            ind = self.dragPoint.data()[0]

            if self.dragFromParameter:
                pts = self.scatter.pointsAt(ev.pos())
                if len(pts) > 0:
                    target_idx = pts[0].data()[0]
                    if ind != target_idx and self._is_parameter(target_idx):
                        self.data['pos'][ind] = self.dragStartPos
                        self.linkRequested.emit(ind, target_idx)
                        self.updateGraph()
                        self.dragPoint = None
                        return
                self.dragPoint = None
                return

            self.dragGroup = []
            self.updateGraph()
            self.dragPoint = None
            return

        else:
            if self.dragPoint is None:
                ev.ignore()
                return

            if self.dragFromParameter:
                if self.tempLine is not None:
                    sx, sy = self.dragStartPos
                    mx, my = ev.pos().x(), ev.pos().y()
                    self.tempLine.setLine(sx, sy, mx, my)
                ev.accept()
                return

            ind = self.dragPoint.data()[0]
            new_pos = ev.pos() + self.dragOffset
            dx = new_pos[0] - self.origPositions[ind][0]
            dy = new_pos[1] - self.origPositions[ind][1]
            self.data['pos'][ind] = new_pos
            for p in self.dragGroup:
                self.data['pos'][p][0] = self.origPositions[p][0] + dx
                self.data['pos'][p][1] = self.origPositions[p][1] + dy
            self.updateGraph()
            ev.accept()

    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pts = self.pointsAt(ev.pos())
            if len(pts) > 0:
                self.ptsClicked = pts
                ev.accept()
                self.sigClicked.emit(self, self.ptsClicked, ev)
            else:
                ev.ignore()
        else:
            ev.ignore()

    @staticmethod
    def _dist_to_segment(p, a, b):
        ax, ay = a
        bx, by = b
        px, py = p
        abx, aby = bx - ax, by - ay
        apx, apy = px - ax, py - ay
        t = (apx * abx + apy * aby) / (abx * abx + aby * aby + 1e-12)
        t = max(0.0, min(1.0, t))
        cx, cy = ax + t * abx, ay + t * aby
        return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

    def mouseDoubleClickEvent(self, ev):
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return
        pos = ev.pos()
        click_pt = (pos.x(), pos.y())
        adj = self.data.get('adj')
        graph_pos = self.data.get('pos')
        if adj is None or graph_pos is None:
            ev.ignore()
            return
        threshold = max(0.05, self.data.get('size', 0.02) * 2.5)
        best_edge = None
        best_dist = threshold
        for i, (src, tgt) in enumerate(adj):
            if not (self._is_parameter(src) and self._is_parameter(tgt)):
                continue
            d = self._dist_to_segment(
                click_pt,
                (graph_pos[src][0], graph_pos[src][1]),
                (graph_pos[tgt][0], graph_pos[tgt][1]),
            )
            if d < best_dist:
                best_dist = d
                best_edge = i
        if best_edge is not None:
            src = adj[best_edge][0]
            self.linkRemovalRequested.emit(src)
            ev.accept()
            return
        ev.ignore()

    def handle_click(self, plot, points):
        if len(self.selectedNodes) >= self.n_selected_nodes:
            p = self.selectedNodes.pop(0)
            p.resetPen()
        for point in points:
            point.setPen(clickedPen)
            self.selectedNodes.append(point)
        if self.update_callback is not None:
            self.update_callback()
        # print(self.selected_nodes_idx)
        # clicked_point_index = np.argmin(np.linalg.norm(self.scatter.data - pos, axis=1))
        # print(f"Clicked on point with index: {clicked_point_index}")
