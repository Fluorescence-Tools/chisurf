import sys
import random
from PyQt5.QtCore import Qt, QBasicTimer, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QFont
from PyQt5.QtWidgets import QFrame, QApplication, QMainWindow

# Game constants
WindowWidth = 800
WindowHeight = 600

PaddleWidth = 10
PaddleHeight = 100
PaddleSpeed = 10    # You can tweak this for desired speed

BallSize = 15
BallSpeedX = 5
BallSpeedY = 4

CPU_SPEED = 8  # How fast the CPU paddle moves


class PongBoard(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFixedSize(WindowWidth, WindowHeight)

        # Game mode (True = 1 player vs CPU, False = 2 players)
        self.vsComputer = True

        # Paddle movement state for player 1 (left)
        self.upPressed = False
        self.downPressed = False

        # Paddle movement state for player 2 (right)
        self.wPressed = False
        self.sPressed = False

        self.timer = QBasicTimer()
        self.isStarted = False

        # Player 1 paddle (left)
        self.playerY = (WindowHeight - PaddleHeight) / 2
        # Player 2 or CPU paddle (right)
        self.cpuY = (WindowHeight - PaddleHeight) / 2

        # Ball
        self.ballPos = QPointF(WindowWidth / 2, WindowHeight / 2)
        self.ballVel = QPointF(0, 0)

        # Scores
        self.playerScore = 0
        self.cpuScore = 0

        self.startGame()

    def startGame(self):
        self.resetBall()
        self.playerScore = 0
        self.cpuScore = 0
        self.timer.start(16, self)  # ~60 FPS
        self.isStarted = True

    def resetBall(self):
        self.ballPos = QPointF(
            WindowWidth / 2 - BallSize / 2,
            WindowHeight / 2 - BallSize / 2
        )
        vx = BallSpeedX if random.choice([True, False]) else -BallSpeedX
        vy = random.choice([-BallSpeedY, BallSpeedY])
        self.ballVel = QPointF(vx, vy)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_P:
            if self.isStarted:
                self.timer.stop()
            else:
                self.timer.start(16, self)
            self.isStarted = not self.isStarted

        elif event.key() == Qt.Key_M:
            # Toggle between 1 player vs CPU and 2 players mode
            self.vsComputer = not self.vsComputer
            self.resetBall()
            # Update window title based on game mode
            self.parent().setWindowTitle('Pong with CPU Opponent' if self.vsComputer else 'Pong - 2 Players')

        # Player 1 controls (left paddle)
        elif event.key() == Qt.Key_Up:
            self.upPressed = True
        elif event.key() == Qt.Key_Down:
            self.downPressed = True

        # Player 2 controls (right paddle, only in 2 player mode)
        elif event.key() == Qt.Key_W:
            self.wPressed = True
        elif event.key() == Qt.Key_S:
            self.sPressed = True

        super(PongBoard, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        # Player 1 controls (left paddle)
        if event.key() == Qt.Key_Up:
            self.upPressed = False
        elif event.key() == Qt.Key_Down:
            self.downPressed = False

        # Player 2 controls (right paddle)
        elif event.key() == Qt.Key_W:
            self.wPressed = False
        elif event.key() == Qt.Key_S:
            self.sPressed = False

        super(PongBoard, self).keyReleaseEvent(event)

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return

        # Player 1 paddle movement (left)
        if self.upPressed:
            self.playerY = max(self.playerY - PaddleSpeed, 0)
        if self.downPressed:
            self.playerY = min(self.playerY + PaddleSpeed, WindowHeight - PaddleHeight)

        # Handle right paddle movement based on game mode
        if self.vsComputer:
            # CPU controls the right paddle
            self.moveCPUPaddle()
        else:
            # Player 2 controls the right paddle
            if self.wPressed:
                self.cpuY = max(self.cpuY - PaddleSpeed, 0)
            if self.sPressed:
                self.cpuY = min(self.cpuY + PaddleSpeed, WindowHeight - PaddleHeight)

        self.moveBall()
        self.update()

    def moveBall(self):
        self.ballPos += self.ballVel

        # Top/bottom collisions
        if self.ballPos.y() <= 0 or self.ballPos.y() + BallSize >= WindowHeight:
            self.ballVel.setY(-self.ballVel.y())

        # Left paddle collision
        playerRect = QRectF(10, self.playerY, PaddleWidth, PaddleHeight)
        ballRect = QRectF(self.ballPos.x(), self.ballPos.y(), BallSize, BallSize)
        if ballRect.intersects(playerRect):
            self.ballVel.setX(abs(self.ballVel.x()))

        # Right paddle collision
        cpuRect = QRectF(WindowWidth - PaddleWidth - 10, self.cpuY, PaddleWidth, PaddleHeight)
        if ballRect.intersects(cpuRect):
            self.ballVel.setX(-abs(self.ballVel.x()))

        # Score?
        if self.ballPos.x() < 0:
            self.cpuScore += 1
            self.resetBall()
        elif self.ballPos.x() + BallSize > WindowWidth:
            self.playerScore += 1
            self.resetBall()

    def moveCPUPaddle(self):
        center = self.cpuY + PaddleHeight / 2
        if center < self.ballPos.y():
            self.cpuY += CPU_SPEED
        elif center > self.ballPos.y() + BallSize:
            self.cpuY -= CPU_SPEED
        self.cpuY = max(0, min(self.cpuY, WindowHeight - PaddleHeight))

    def paintEvent(self, event):
        painter = QPainter(self)

        # Background
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        # Draw center dashed line
        pen = painter.pen()
        pen.setColor(QColor(200, 200, 200))
        painter.setPen(pen)
        dash_len = 20
        x_center = WindowWidth // 2
        y = 0
        while y < WindowHeight:
            painter.drawLine(x_center, y, x_center, y + dash_len // 2)
            y += dash_len

        # Draw paddles
        painter.fillRect(10, int(self.playerY), PaddleWidth, PaddleHeight, QColor(255, 255, 255))
        painter.fillRect(WindowWidth - PaddleWidth - 10, int(self.cpuY),
                         PaddleWidth, PaddleHeight, QColor(255, 255, 255))

        # Draw ball
        painter.fillRect(int(self.ballPos.x()), int(self.ballPos.y()),
                         BallSize, BallSize, QColor(255, 255, 255))

        # Draw scores
        painter.setFont(QFont('Arial', 30))
        painter.drawText(WindowWidth // 4, 50, str(self.playerScore))
        painter.drawText(WindowWidth * 3 // 4, 50, str(self.cpuScore))


class Pong(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.board = PongBoard(self)
        self.setCentralWidget(self.board)
        self.board.setFocus()
        self.statusBar().showMessage('Player 1: ↑ ↓ to move | Player 2: W S to move | M to toggle mode | P to pause')
        self.resize(WindowWidth, WindowHeight)
        self.setWindowTitle('Pong with CPU Opponent')
        self.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    pong = Pong()
    sys.exit(app.exec_())
