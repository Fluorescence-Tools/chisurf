import sys
import random
from qtpy.QtCore import Qt, QBasicTimer, QPoint
from qtpy.QtGui import QPainter, QColor
from qtpy.QtWidgets import QFrame, QApplication, QMainWindow

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

try:
    from chisurf.plugins.misc.tetris_game.sound import SoundManager
except ImportError:
    class SoundManager:
        def __init__(self): self._muted = True
        @property
        def muted(self): return self._muted
        @muted.setter
        def muted(self, value): pass
        def toggle(self): pass
        def play(self, name): pass
        def ensure_sounds(self): pass

BoardWidth = 10
BoardHeight = 22

class Tetromino:
    shapes = [
        [[0, -1], [0, 0], [0, 1], [0, 2]],
        [[-1, -1], [-1, 0], [0, 0], [1, 0]],
        [[1, -1], [-1, 0], [0, 0], [1, 0]],
        [[0, -1], [1, -1], [0, 0], [1, 0]],
        [[0, -1], [1, -1], [-1, 0], [0, 0]],
        [[-1, -1], [0, -1], [1, -1], [0, 0]],
        [[-1, -1], [0, -1], [0, 0], [1, 0]],
    ]
    colors = [
        QColor(0, 255, 255),
        QColor(0, 0, 255),
        QColor(255, 127, 0),
        QColor(255, 255, 0),
        QColor(0, 255, 0),
        QColor(128, 0, 128),
        QColor(255, 0, 0),
    ]

    def __init__(self, shape=None):
        self.shape = random.randrange(len(Tetromino.shapes)) if shape is None else shape
        self.coords = [QPoint(x, y) for x, y in Tetromino.shapes[self.shape]]

    def rotate(self):
        if self.shape == 3:
            return self
        rotated = Tetromino(self.shape)
        rotated.coords = [QPoint(-pt.y(), pt.x()) for pt in self.coords]
        return rotated

class Board(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.timer = QBasicTimer()
        self.isStarted = False
        self.isPaused = False
        self.sound = SoundManager()
        self.clearBoard()
        self.currentPiece = None
        self.curX = 0
        self.curY = 0
        self.nextPiece = Tetromino()
        self.speed = 300
        self.setFocusPolicy(Qt.StrongFocus)
        self.start()

    def start(self):
        if self.isPaused:
            return
        self.isStarted = True
        self.sound.ensure_sounds()
        self.clearBoard()
        self.newPiece()
        self.timer.start(self.speed, self)

    def pause(self):
        if not self.isStarted:
            return
        self.isPaused = not self.isPaused
        if self.isPaused:
            self.timer.stop()
        else:
            self.timer.start(self.speed, self)
        self.update()

    def clearBoard(self):
        self.board = [
            [QColor(0, 0, 0) for _ in range(BoardWidth)]
            for _ in range(BoardHeight)
        ]

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.contentsRect()
        boardTop = rect.bottom() - BoardHeight * self.squareHeight()

        for i in range(BoardHeight):
            for j in range(BoardWidth):
                color = self.board[i][j]
                if color.isValid():
                    self.drawSquare(
                        painter,
                        rect.left() + j * self.squareWidth(),
                        boardTop + i * self.squareHeight(),
                        color
                    )

        if self.currentPiece:
            for p in self.currentPiece.coords:
                x = self.curX + p.x()
                y = self.curY + p.y()
                self.drawSquare(
                    painter,
                    rect.left() + x * self.squareWidth(),
                    boardTop + y * self.squareHeight(),
                    Tetromino.colors[self.currentPiece.shape]
                )

    def gameOver(self):
        self.timer.stop()
        self.isStarted = False
        self.sound.play('game_over')

        fill_color = QColor(128, 128, 128)
        for row in range(BoardHeight):
            for col in range(BoardWidth):
                self.board[row][col] = fill_color
        self.update()

    def keyPressEvent(self, event):
        if not self.isStarted or self.currentPiece is None:
            super(Board, self).keyPressEvent(event)
            return

        key = event.key()
        if key == Qt.Key_P:
            self.pause()
            return
        elif key == Qt.Key_R:
            self.start()
            self.parent().statusBar().showMessage(
                'P pause | R restart | M sound')
            return
        elif key == Qt.Key_M:
            self.sound.toggle()
            status = 'ON' if not self.sound.muted else 'OFF'
            self.parent().statusBar().showMessage(f'Sound: {status} — Press M to toggle')
            return

        if self.isPaused:
            return

        if key == Qt.Key_Left:
            self.tryMove(self.currentPiece, self.curX - 1, self.curY)
        elif key == Qt.Key_Right:
            self.tryMove(self.currentPiece, self.curX + 1, self.curY)
        elif key == Qt.Key_Down:
            self.dropDown()
        elif key == Qt.Key_Up:
            if self.tryMove(self.currentPiece.rotate(), self.curX, self.curY):
                self.sound.play('rotate')
        else:
            super(Board, self).keyPressEvent(event)

    def timerEvent(self, event):
        if event.timerId() == self.timer.timerId():
            if self.isPaused:
                return
            self.oneLineDown()
        else:
            super(Board, self).timerEvent(event)

    def oneLineDown(self):
        if not self.tryMove(self.currentPiece, self.curX, self.curY + 1):
            self.pieceDropped()

    def dropDown(self):
        newY = self.curY
        while self.tryMove(self.currentPiece, self.curX, newY + 1):
            newY += 1
        self.pieceDropped()

    def pieceDropped(self):
        for p in self.currentPiece.coords:
            x = self.curX + p.x()
            y = self.curY + p.y()
            self.board[y][x] = Tetromino.colors[self.currentPiece.shape]
        self.sound.play('drop')
        self.removeFullLines()

        if not self.isStarted:
            return
        self.newPiece()

    def removeFullLines(self):
        newBoard = []
        linesRemoved = 0
        for row in self.board:
            if all(color.isValid() and color != QColor(0, 0, 0) for color in row):
                linesRemoved += 1
            else:
                newBoard.append(row)
        for _ in range(linesRemoved):
            newBoard.insert(0, [QColor(0, 0, 0) for _ in range(BoardWidth)])
        if linesRemoved > 0:
            self.board = newBoard
            self.sound.play('line_clear')
            self.update()

    def newPiece(self):
        self.currentPiece = self.nextPiece
        self.nextPiece = Tetromino()
        self.curX = BoardWidth // 2
        self.curY = -min(p.y() for p in self.currentPiece.coords)
        if not self.tryMove(self.currentPiece, self.curX, self.curY):
            self.currentPiece = None
            self.timer.stop()
            self.isStarted = False
            self.gameOver()

    def tryMove(self, piece, newX, newY):
        for p in piece.coords:
            x = newX + p.x()
            y = newY + p.y()
            if x < 0 or x >= BoardWidth or y < 0 or y >= BoardHeight:
                return False
            if self.board[y][x] != QColor(0, 0, 0):
                return False
        self.currentPiece = piece
        self.curX = newX
        self.curY = newY
        self.update()
        return True

    def squareWidth(self):
        return self.contentsRect().width() // BoardWidth

    def squareHeight(self):
        return self.contentsRect().height() // BoardHeight

    def drawSquare(self, painter, x, y, color):
        painter.fillRect(
            x + 1, y + 1,
            self.squareWidth() - 2, self.squareHeight() - 2,
            color
        )
        painter.setPen(color.lighter())
        painter.drawLine(x, y + self.squareHeight() - 1, x, y)
        painter.drawLine(x, y, x + self.squareWidth() - 1, y)
        painter.setPen(color.darker())
        painter.drawLine(
            x + 1, y + self.squareHeight() - 1,
            x + self.squareWidth() - 1, y + self.squareHeight() - 1
        )
        painter.drawLine(
            x + self.squareWidth() - 1, y + self.squareHeight() - 1,
            x + self.squareWidth() - 1, y + 1
        )

@persist_plugin_state("tetris")
class Tetris(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.board = Board(self)
        self.setCentralWidget(self.board)
        self.statusBar().showMessage(
            'P pause | R restart | M sound')
        self.setFixedSize(BoardWidth * 20, BoardHeight * 20)
        self.setWindowTitle('Tetris')
        self.show()

    def closeEvent(self, event):
        self.board.timer.stop()
        event.accept()
        self.deleteLater()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    tetris = Tetris()
    sys.exit(app.exec())
