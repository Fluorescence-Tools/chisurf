import numba as nb
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore

clickedPen = pg.mkPen('m', width=4)


@nb.jit(nopython=True)
def update_node_positions(
        pos,
        node,
        connections,
        hard_sphere_radius: float = 0.025,
        repulsion: float = 0.01,
        spring_constant: float = 0.1
):
    # Repulsion force to other nodes
    n_nodes = len(pos)
    for other_node in range(n_nodes):
        if other_node != node:
            dx = pos[node][0] - pos[other_node][0]
            dy = pos[node][1] - pos[other_node][1]
            distance = (dx ** 2 + dy ** 2) ** 0.5

            repulsion_force = 0.0
            if distance < hard_sphere_radius:
                repulsion_force = repulsion
            else:
                continue

            pos[other_node] = (
                pos[other_node][0] - repulsion_force * dx / distance,
                pos[other_node][1] - repulsion_force * dy / distance
            )
            pos[node] = (
                pos[node][0] + repulsion_force * dx / distance,
                pos[node][1] + repulsion_force * dy / distance
            )
    for ia, ib in connections:
        if ia != node and ib != node:
            continue
        dx = pos[ia][0] - pos[ib][0]
        dy = pos[ia][1] - pos[ib][1]
        distance = (dx ** 2 + dy ** 2) ** 0.5
        attractive_force = spring_constant * distance
        pos[ia] = (
            pos[ia][0] - attractive_force * dx,
            pos[ia][1] - attractive_force * dx
        )
        pos[ib] = (
            pos[ib][0] + attractive_force * dy,
            pos[ib][1] + attractive_force * dy
        )

    return pos


class GraphPlotWidget(pg.GraphItem):

    @property
    def selected_nodes_idx(self) -> list[int]:
        return [n.index() for n in self.selectedNodes]

    def __init__(
            self,
            update_callback: callable = None,
            dynamic_drag_nodes: bool = False,
            n_selected_nodes: int = 2,
            *args,
            **kwargs
    ):
        self.dragPoint = None
        self.dragOffset = None
        self.selectedNodes = list()
        self.dynamic_drag_nodes = dynamic_drag_nodes

        self.n_selected_nodes = n_selected_nodes
        self.update_callback = update_callback

        self.textItems = list()
        self.arrows = list()
        self.fit_circles = list()

        super(GraphPlotWidget, self).__init__(*args, **kwargs)
        self.scatter.sigClicked.connect(self.handle_click)

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
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
        print("def updateGraph(self):")
        if len(self.data) == 0:
            return

        # Update test
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

        # Update arrows (angles)
        print("self.data['pos']:", self.data['pos'])
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

    def mouseDragEvent(self, ev, dynamic_drag_nodes: bool = None):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        if dynamic_drag_nodes is None:
            dynamic_drag_nodes = self.dynamic_drag_nodes

        if ev.isStart():
            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            pos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(pos)
            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - pos

        elif ev.isFinish():
            # ind = self.dragPoint.data()[0]
            # edges = self.data['adj']
            self.dragPoint = None
            # self.data['pos'] = update_node_positions(self.data['pos'], ind, edges)
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        ind = self.dragPoint.data()[0]
        self.data['pos'][ind] = ev.pos() + self.dragOffset
        if dynamic_drag_nodes:
            edges = self.data['adj']
            self.data['pos'] = update_node_positions(self.data['pos'], ind, edges)

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
                # print "no spots"
                ev.ignore()
        else:
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
        print(self.selected_nodes_idx)
        # clicked_point_index = np.argmin(np.linalg.norm(self.scatter.data - pos, axis=1))
        # print(f"Clicked on point with index: {clicked_point_index}")
