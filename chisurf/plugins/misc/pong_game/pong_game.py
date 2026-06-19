import sys
import math
import random
from qtpy.QtCore import Qt, QBasicTimer, QRectF, QPointF, QTimer
from qtpy.QtGui import QPainter, QColor, QFont, QPen, QBrush, QRadialGradient, QLinearGradient
from qtpy.QtWidgets import QFrame, QApplication, QMainWindow, QMessageBox

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

try:
    from chisurf.plugins.misc.pong_game.sound import SoundManager
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

WindowWidth = 800
WindowHeight = 600

PaddleWidth = 12
PaddleHeight = 90
PaddleSpeed = 8

BallSize = 14
BaseBallSpeedX = 6
BaseBallSpeedY = 5

CPU_SPEED = 7
WIN_SCORE = 7


class Particle:
    def __init__(self, x, y, vx, vy, color, life=20):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.life = life
        self.max_life = life

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.2
        self.life -= 1

    def draw(self, painter):
        alpha = int(255 * self.life / self.max_life)
        c = QColor(self.color)
        c.setAlpha(alpha)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(c))
        size = 3 + 2 * (self.life / self.max_life)
        painter.drawEllipse(QPointF(int(self.x), int(self.y)), size, size)


class PongBoard(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFixedSize(WindowWidth, WindowHeight)

        self.sound = SoundManager()

        self.vsComputer = True
        self.upPressed = False
        self.downPressed = False
        self.wPressed = False
        self.sPressed = False

        self.timer = QBasicTimer()
        self.isStarted = False
        self.isPaused = False
        self.serve = True
        self.serve_timer = 0
        self.rally_count = 0

        self.playerY = (WindowHeight - PaddleHeight) / 2
        self.cpuY = (WindowHeight - PaddleHeight) / 2

        self.ballPos = QPointF(WindowWidth / 2, WindowHeight / 2)
        self.ballVel = QPointF(0, 0)

        self.playerScore = 0
        self.cpuScore = 0

        self.particles = []
        self.flash_timer = 0

        self.startGame()

    def startGame(self):
        self.playerScore = 0
        self.cpuScore = 0
        self.rally_count = 0
        self.sound.ensure_sounds()
        self.particles.clear()
        self.beginServe()
        self.timer.start(16, self)
        self.isStarted = True
        self.isPaused = False

    def beginServe(self):
        self.serve = True
        self.serve_timer = 90
        self.ballPos = QPointF(WindowWidth / 2 - BallSize / 2, WindowHeight / 2 - BallSize / 2)
        self.ballVel = QPointF(0, 0)

    def resetBall(self):
        self.beginServe()
        self.rally_count = 0

    def spawnParticles(self, x, y, color, count=12):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 5)
            self.particles.append(Particle(
                x, y,
                math.cos(angle) * speed,
                math.sin(angle) * speed - 1,
                color, random.randint(10, 25)
            ))

    def keyPressEvent(self, event):
        k = event.key()
        if k == Qt.Key_P:
            if not self.isStarted:
                return
            self.isPaused = not self.isPaused
            if self.isPaused:
                self.timer.stop()
            else:
                self.timer.start(16, self)
            self.update()
            return
        if k == Qt.Key_R:
            self.startGame()
            return
        if k == Qt.Key_M:
            self.vsComputer = not self.vsComputer
            self.resetBall()
            self.parent().setWindowTitle(
                'Pong vs CPU' if self.vsComputer else 'Pong - 2 Players'
            )
            return
        if k == Qt.Key_N:
            self.sound.toggle()
            status = 'ON' if not self.sound.muted else 'OFF'
            self.parent().statusBar().showMessage(f'Sound: {status} — Press N to toggle')
            return
        if k == Qt.Key_Up:
            self.upPressed = True
        elif k == Qt.Key_Down:
            self.downPressed = True
        elif k == Qt.Key_W:
            self.wPressed = True
        elif k == Qt.Key_S:
            self.sPressed = True
        super(PongBoard, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        k = event.key()
        if k == Qt.Key_Up:
            self.upPressed = False
        elif k == Qt.Key_Down:
            self.downPressed = False
        elif k == Qt.Key_W:
            self.wPressed = False
        elif k == Qt.Key_S:
            self.sPressed = False
        super(PongBoard, self).keyReleaseEvent(event)

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return
        if self.isPaused:
            return

        if self.serve:
            self.serve_timer -= 1
            if self.serve_timer <= 0:
                self.serve = False
                self.launchBall()
            self.update()
            return

        if self.upPressed:
            self.playerY = max(self.playerY - PaddleSpeed, 0)
        if self.downPressed:
            self.playerY = min(self.playerY + PaddleSpeed, WindowHeight - PaddleHeight)

        if self.vsComputer:
            self.moveCPUPaddle()
        else:
            if self.wPressed:
                self.cpuY = max(self.cpuY - PaddleSpeed, 0)
            if self.sPressed:
                self.cpuY = min(self.cpuY + PaddleSpeed, WindowHeight - PaddleHeight)

        self.moveBall()
        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)
        if self.flash_timer > 0:
            self.flash_timer -= 1
        self.update()

    def launchBall(self):
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        direction = 1 if random.choice([True, False]) else -1
        speed = BaseBallSpeedX + self.rally_count * 0.15
        self.ballVel = QPointF(
            math.cos(angle) * speed * direction,
            math.sin(angle) * speed
        )

    def moveBall(self):
        self.ballPos += self.ballVel

        if self.ballPos.y() <= 0:
            self.ballPos.setY(0)
            self.ballVel.setY(abs(self.ballVel.y()))
            self.sound.play('wall_bounce')
        elif self.ballPos.y() + BallSize >= WindowHeight:
            self.ballPos.setY(WindowHeight - BallSize)
            self.ballVel.setY(-abs(self.ballVel.y()))
            self.sound.play('wall_bounce')

        bx = self.ballPos.x()
        by = self.ballPos.y()
        bs = BallSize
        pr = QRectF(10, self.playerY, PaddleWidth, PaddleHeight)
        br = QRectF(bx, by, bs, bs)

        if br.intersects(pr) and self.ballVel.x() < 0:
            offset = ((by + bs / 2) - (self.playerY + PaddleHeight / 2)) / (PaddleHeight / 2)
            speed = math.hypot(self.ballVel.x(), self.ballVel.y())
            speed = min(speed + 0.3, BaseBallSpeedX * 3)
            angle = offset * math.pi / 3
            self.ballVel = QPointF(abs(math.cos(angle) * speed), math.sin(angle) * speed)
            self.rally_count += 1
            self.sound.play('paddle_hit')
            self.spawnParticles(10 + PaddleWidth, by + bs / 2, QColor(255, 255, 200), 8)
            self.flash_timer = 4

        cr = QRectF(WindowWidth - PaddleWidth - 10, self.cpuY, PaddleWidth, PaddleHeight)
        if br.intersects(cr) and self.ballVel.x() > 0:
            offset = ((by + bs / 2) - (self.cpuY + PaddleHeight / 2)) / (PaddleHeight / 2)
            speed = math.hypot(self.ballVel.x(), self.ballVel.y())
            speed = min(speed + 0.3, BaseBallSpeedX * 3)
            angle = offset * math.pi / 3
            self.ballVel = QPointF(-abs(math.cos(angle) * speed), math.sin(angle) * speed)
            self.rally_count += 1
            self.sound.play('paddle_hit')
            self.spawnParticles(WindowWidth - 10 - PaddleWidth, by + bs / 2, QColor(255, 200, 255), 8)
            self.flash_timer = 4

        if bx < -BallSize:
            self.cpuScore += 1
            self.sound.play('score')
            self.spawnParticles(0, by + bs / 2, QColor(255, 100, 100), 20)
            if self.cpuScore >= WIN_SCORE:
                self.gameOver("CPU wins!")
            else:
                self.resetBall()
        elif bx > WindowWidth:
            self.playerScore += 1
            self.sound.play('score')
            self.spawnParticles(WindowWidth, by + bs / 2, QColor(100, 255, 100), 20)
            if self.playerScore >= WIN_SCORE:
                self.gameOver("You win!")
            else:
                self.resetBall()

    def moveCPUPaddle(self):
        pred_y = self.ballPos.y() + BallSize / 2
        if self.ballVel.x() > 0:
            time_to_reach = (WindowWidth - PaddleWidth - 20 - self.ballPos.x()) / abs(self.ballVel.x())
            pred_y += self.ballVel.y() * time_to_reach
            pred_y = max(0, min(pred_y, WindowHeight))
        target = self.cpuY + PaddleHeight / 2
        diff = pred_y - target
        if abs(diff) > 15:
            self.cpuY += CPU_SPEED * (1 if diff > 0 else -1)
        self.cpuY = max(0, min(self.cpuY, WindowHeight - PaddleHeight))

    def gameOver(self, message):
        self.timer.stop()
        self.isStarted = False
        self.sound.play('game_over')
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 200))
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont('Arial', 48, QFont.Bold))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{message}\nPress R to restart")
        painter.end()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg = QLinearGradient(0, 0, 0, WindowHeight)
        bg.setColorAt(0, QColor(10, 10, 30))
        bg.setColorAt(1, QColor(20, 20, 50))
        painter.fillRect(self.rect(), QBrush(bg))

        pen = painter.pen()
        pen.setColor(QColor(60, 60, 100))
        pen.setWidth(2)
        painter.setPen(pen)
        dash_len = 24
        x_center = WindowWidth // 2
        y = 0
        while y < WindowHeight:
            painter.drawLine(x_center, y, x_center, y + dash_len // 2)
            y += dash_len

        if self.flash_timer > 0:
            flash = QColor(255, 255, 255, self.flash_timer * 20)
            painter.fillRect(self.rect(), flash)

        pad_grad_left = QLinearGradient(0, 0, PaddleWidth, 0)
        pad_grad_left.setColorAt(0, QColor(100, 200, 255))
        pad_grad_left.setColorAt(1, QColor(60, 100, 255))
        painter.setBrush(QBrush(pad_grad_left))
        painter.setPen(QPen(QColor(200, 230, 255), 1))
        painter.drawRoundedRect(10, int(self.playerY), PaddleWidth, PaddleHeight, 4, 4)

        if self.vsComputer:
            pad_grad_right = QLinearGradient(0, 0, PaddleWidth, 0)
            pad_grad_right.setColorAt(0, QColor(255, 100, 100))
            pad_grad_right.setColorAt(1, QColor(200, 50, 50))
        else:
            pad_grad_right = QLinearGradient(0, 0, PaddleWidth, 0)
            pad_grad_right.setColorAt(0, QColor(255, 200, 100))
            pad_grad_right.setColorAt(1, QColor(255, 150, 50))
        painter.setBrush(QBrush(pad_grad_right))
        painter.drawRoundedRect(
            WindowWidth - PaddleWidth - 10, int(self.cpuY),
            PaddleWidth, PaddleHeight, 4, 4
        )

        ball_grad = QRadialGradient(BallSize / 2, BallSize / 2, BallSize / 2)
        ball_grad.setColorAt(0, QColor(255, 255, 255))
        ball_grad.setColorAt(0.6, QColor(255, 255, 200))
        ball_grad.setColorAt(1, QColor(200, 200, 100))
        painter.setBrush(QBrush(ball_grad))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            int(self.ballPos.x()), int(self.ballPos.y()),
            BallSize, BallSize
        )

        for p in self.particles:
            p.draw(painter)

        painter.setPen(QColor(200, 200, 220))
        painter.setFont(QFont('Arial', 14))
        painter.drawText(WindowWidth // 4, 30, f"Player: {self.playerScore}")
        painter.drawText(WindowWidth * 3 // 4 - 60, 30, f"CPU: {self.cpuScore}")

        info = f"Rally: {self.rally_count}  |  P: pause  R: restart  M: mode  N: sound"
        painter.setFont(QFont('Arial', 10))
        painter.setPen(QColor(120, 120, 150))
        painter.drawText(WindowWidth // 2 - 150, WindowHeight - 10, info)

        if self.isPaused:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 160))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Arial', 36, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "PAUSED")

        if self.serve and not self.isPaused:
            painter.setPen(QColor(255, 255, 200))
            painter.setFont(QFont('Arial', 20, QFont.Bold))
            t = max(1, self.serve_timer // 15 + 1)
            painter.drawText(self.rect(), Qt.AlignCenter, f"{t}...")


@persist_plugin_state("pong")
class Pong(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.board = PongBoard(self)
        self.setCentralWidget(self.board)
        self.board.setFocus()
        self.statusBar().showMessage('↑ ↓ to move | P pause | R restart | M 1p/2p | N sound')
        self.setFixedSize(WindowWidth, WindowHeight)
        self.setWindowTitle('Pong vs CPU')
        self.show()

    def closeEvent(self, event):
        self.board.timer.stop()
        event.accept()
        self.deleteLater()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    pong = Pong()
    sys.exit(app.exec())
