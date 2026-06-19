import sys
import math
import random
from qtpy.QtCore import Qt, QBasicTimer, QRectF, QPointF, QSize
from qtpy.QtGui import QPainter, QColor, QFont, QBrush, QPen, QLinearGradient, QRadialGradient
from qtpy.QtWidgets import QFrame, QApplication, QMainWindow

try:
    from chisurf.gui.misc_helpers import persist_plugin_state
except ImportError:
    persist_plugin_state = lambda n: lambda c: c

try:
    from chisurf.plugins.misc.breakout_game.sound import SoundManager
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

W = 800
H = 650

PaddleW = 100
PaddleH = 16
PaddleY = H - 40

BallSize = 12
BaseBallSpeed = 5.5

BrickRows = 8
BrickCols = 10
BrickW = (W - 60) // BrickCols
BrickH = 22
BrickTop = 50

Lives = 3

RowColors = [
    QColor(255, 60, 60),
    QColor(255, 120, 40),
    QColor(255, 200, 40),
    QColor(100, 220, 60),
    QColor(60, 180, 255),
    QColor(100, 100, 255),
    QColor(180, 80, 255),
    QColor(255, 80, 200),
]

RowHardness = [2, 2, 2, 1, 1, 1, 1, 1]


class Brick:
    def __init__(self, x, y, w, h, color, hp):
        self.rect = QRectF(x, y, w, h)
        self.color = color
        self.max_hp = hp
        self.hp = hp
        self.alive = True

    def hit(self):
        self.hp -= 1
        if self.hp <= 0:
            self.alive = False
            return True
        return False

    def draw(self, painter):
        if not self.alive:
            return
        ratio = self.hp / self.max_hp
        c = QColor(self.color)
        if self.hp < self.max_hp:
            c = c.lighter(130)
        grad = QLinearGradient(self.rect.topLeft(), self.rect.bottomLeft())
        grad.setColorAt(0, c.lighter(140))
        grad.setColorAt(1, c)
        painter.setBrush(QBrush(grad))
        painter.setPen(QPen(c.darker(150), 1))
        painter.drawRoundedRect(self.rect, 3, 3)


class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.vx = random.uniform(-3, 3)
        self.vy = random.uniform(-4, -1)
        self.color = color
        self.life = 20

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.15
        self.life -= 1

    def draw(self, painter):
        if self.life <= 0:
            return
        alpha = int(255 * self.life / 20)
        c = QColor(self.color)
        c.setAlpha(alpha)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(c))
        painter.drawEllipse(QPointF(self.x, self.y), 3, 3)


class BreakoutBoard(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)
        self.setFixedSize(W, H)

        self.sound = SoundManager()

        self.paddleX = (W - PaddleW) / 2
        self.ballPos = QPointF(W / 2, PaddleY - BallSize)
        self.ballVel = QPointF(0, 0)

        self.score = 0
        self.lives = Lives
        self.level = 1
        self.isStarted = False
        self.isPaused = False
        self.serve = True
        self.ball_stuck = True

        self.particles = []
        self.flash = 0

        self.bricks = []
        self.timer = QBasicTimer()
        self.keys_held = set()

        self.initLevel()

    def initLevel(self):
        self.bricks.clear()
        self.particles.clear()
        pad = 30
        for r in range(BrickRows):
            color = RowColors[r % len(RowColors)]
            hp = RowHardness[r % len(RowHardness)]
            for c in range(BrickCols):
                x = pad + c * BrickW + 4
                y = BrickTop + r * (BrickH + 4)
                self.bricks.append(Brick(x, y, BrickW - 8, BrickH, color, hp))
        self.paddleX = (W - PaddleW) / 2
        self.serveBall()

    def serveBall(self):
        self.serve = True
        self.ball_stuck = True
        self.ballPos = QPointF(self.paddleX + PaddleW / 2 - BallSize / 2, PaddleY - BallSize)

    def launchBall(self):
        angle = random.uniform(-math.pi / 4, math.pi / 4)
        direction = 1 if random.choice([True, False]) else -1
        self.ballVel = QPointF(
            math.cos(angle) * BaseBallSpeed * direction,
            -abs(math.sin(angle) * BaseBallSpeed)
        )
        self.serve = False
        self.ball_stuck = False
        self.sound.play('launch')

    def start(self):
        self.score = 0
        self.lives = Lives
        self.level = 1
        self.sound.ensure_sounds()
        self.initLevel()
        self.timer.start(16, self)
        self.isStarted = True
        self.isPaused = False

    def gameOver(self, won=False):
        self.timer.stop()
        self.isStarted = False
        self.sound.play('game_over')
        self.update()

    def spawnParticles(self, x, y, color, count=15):
        for _ in range(count):
            self.particles.append(Particle(x, y, color))

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
            self.start()
            return
        if k == Qt.Key_M:
            self.sound.toggle()
            status = 'ON' if not self.sound.muted else 'OFF'
            self.parent().statusBar().showMessage(f'Sound: {status} — Press M to toggle')
            return
        self.keys_held.add(k)
        if k == Qt.Key_Space:
            if self.ball_stuck:
                self.launchBall()
        super(BreakoutBoard, self).keyPressEvent(event)

    def keyReleaseEvent(self, event):
        self.keys_held.discard(event.key())
        super(BreakoutBoard, self).keyReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.isStarted:
            mx = event.x() - PaddleW / 2
            self.paddleX = max(0, min(W - PaddleW, mx))
            if self.ball_stuck:
                self.ballPos.setX(self.paddleX + PaddleW / 2 - BallSize / 2)

    def timerEvent(self, event):
        if event.timerId() != self.timer.timerId():
            return
        if self.isPaused:
            return
        if not self.isStarted:
            return
        move_step = 6
        if Qt.Key_Left in self.keys_held:
            self.paddleX = max(0, self.paddleX - move_step)
        if Qt.Key_Right in self.keys_held:
            self.paddleX = min(W - PaddleW, self.paddleX + move_step)
        if self.ball_stuck:
            self.ballPos.setX(self.paddleX + PaddleW / 2 - BallSize / 2)
            self.update()
            return

        self.ballPos += self.ballVel
        bx, by = self.ballPos.x(), self.ballPos.y()
        bvx, bvy = self.ballVel.x(), self.ballVel.y()
        bs = BallSize

        if by <= 0:
            self.ballPos.setY(0)
            self.ballVel.setY(abs(bvy))
            self.sound.play('wall_bounce')
        if bx <= 0:
            self.ballPos.setX(0)
            self.ballVel.setX(abs(bvx))
            self.sound.play('wall_bounce')
        if bx + bs >= W:
            self.ballPos.setX(W - bs)
            self.ballVel.setX(-abs(bvx))
            self.sound.play('wall_bounce')

        pr = QRectF(self.paddleX, PaddleY, PaddleW, PaddleH)
        br = QRectF(bx, by, bs, bs)
        if br.intersects(pr) and bvy > 0:
            offset = ((bx + bs / 2) - (self.paddleX + PaddleW / 2)) / (PaddleW / 2)
            speed = math.hypot(bvx, bvy)
            speed = min(speed + 0.2, BaseBallSpeed * 2.5)
            angle = offset * math.pi / 2.5
            self.ballVel = QPointF(math.sin(angle) * speed, -abs(math.cos(angle) * speed))
            self.ballPos.setY(PaddleY - bs)
            self.sound.play('paddle_hit')
            self.spawnParticles(bx + bs / 2, PaddleY, QColor(255, 255, 200), 6)

        hit_any = False
        for brick in self.bricks:
            if not brick.alive:
                continue
            if br.intersects(brick.rect):
                dx1 = bx + bs - brick.rect.left()
                dx2 = brick.rect.right() - bx
                dy1 = by + bs - brick.rect.top()
                dy2 = brick.rect.bottom() - by
                min_dx = min(dx1, dx2)
                min_dy = min(dy1, dy2)
                if min_dx < min_dy:
                    self.ballVel.setX(-bvx)
                else:
                    self.ballVel.setY(-bvy)
                destroyed = brick.hit()
                if destroyed:
                    self.score += 10 * (brick.max_hp + 1)
                    self.spawnParticles(
                        brick.rect.center().x(), brick.rect.center().y(),
                        brick.color, 12
                    )
                    self.sound.play('brick_break')
                else:
                    self.spawnParticles(
                        brick.rect.center().x(), brick.rect.center().y(),
                        brick.color, 4
                    )
                hit_any = True
                break

        if hit_any:
            speed = math.hypot(self.ballVel.x(), self.ballVel.y())
            speed = min(speed, BaseBallSpeed * 2.5)
            self.ballVel = self.ballVel / math.hypot(self.ballVel.x(), self.ballVel.y()) * speed

        if by + bs >= H:
            self.lives -= 1
            if self.lives > 0:
                self.serveBall()
            else:
                self.gameOver(False)
                self.update()

        alive = sum(1 for b in self.bricks if b.alive)
        if alive == 0:
            self.level += 1
            self.sound.play('level_up')
            self.initLevel()

        for p in self.particles[:]:
            p.update()
            if p.life <= 0:
                self.particles.remove(p)

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        bg = QLinearGradient(0, 0, 0, H)
        bg.setColorAt(0, QColor(8, 8, 30))
        bg.setColorAt(1, QColor(18, 18, 50))
        painter.fillRect(self.rect(), QBrush(bg))

        for brick in self.bricks:
            brick.draw(painter)

        pg = QLinearGradient(0, PaddleY, 0, PaddleY + PaddleH)
        pg.setColorAt(0, QColor(180, 220, 255))
        pg.setColorAt(1, QColor(80, 140, 255))
        painter.setBrush(QBrush(pg))
        painter.setPen(QPen(QColor(200, 230, 255), 1))
        painter.drawRoundedRect(int(self.paddleX), PaddleY, PaddleW, PaddleH, 6, 6)

        bg_ball = QRadialGradient(BallSize / 2, BallSize / 2, BallSize / 2)
        bg_ball.setColorAt(0, QColor(255, 255, 255))
        bg_ball.setColorAt(0.6, QColor(255, 255, 180))
        bg_ball.setColorAt(1, QColor(200, 200, 100))
        painter.setBrush(QBrush(bg_ball))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(
            int(self.ballPos.x()), int(self.ballPos.y()),
            BallSize, BallSize
        )

        for p in self.particles:
            p.draw(painter)

        painter.setPen(QColor(200, 200, 220))
        painter.setFont(QFont('Arial', 14, QFont.Bold))
        painter.drawText(15, 25, f"Score: {self.score}")
        painter.drawText(W // 2 - 40, 25, f"Level: {self.level}")
        painter.drawText(W - 100, 25, f"Lives: {'♥' * self.lives}")

        if not self.isStarted:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 180))
            painter.setPen(QColor(255, 255, 200))
            painter.setFont(QFont('Arial', 36, QFont.Bold))
            if self.lives <= 0:
                painter.drawText(self.rect(), Qt.AlignCenter, f"Game Over!\nScore: {self.score}\nPress R to restart")
            else:
                painter.drawText(self.rect(), Qt.AlignCenter,
                                 "Breakout\n\nPress R to start")
        elif self.isPaused:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 160))
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont('Arial', 36, QFont.Bold))
            painter.drawText(self.rect(), Qt.AlignCenter, "PAUSED")

        elif self.ball_stuck:
            painter.setPen(QColor(200, 220, 255))
            painter.setFont(QFont('Arial', 14))
            painter.drawText(W // 2 - 80, H // 2 + 40, "Press SPACE or click to launch")


@persist_plugin_state("breakout")
class Breakout(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.board = BreakoutBoard(self)
        self.setCentralWidget(self.board)
        self.board.setFocus()
        self.statusBar().showMessage(
            '← → mouse | SPACE launch | P pause | R restart | M sound')
        self.setFixedSize(W, H)
        self.setWindowTitle('Breakout')
        self.show()

    def closeEvent(self, event):
        self.board.timer.stop()
        event.accept()
        self.deleteLater()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    game = Breakout()
    sys.exit(app.exec())
