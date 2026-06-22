"""
Script to create an icon for the BH SPC Acquisition plugin.
"""

import os
import sys
from qtpy.QtWidgets import QApplication
from qtpy.QtGui import QPainter, QColor, QFont, QPixmap, QPen, QBrush
from qtpy.QtCore import Qt, QRect, QPoint

def create_icon(output_path, size=64):
    """Create an icon for the Acquisition plugin."""
    app = QApplication(sys.argv)
    
    # Create a pixmap
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    
    # Create a painter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Draw background
    painter.setPen(Qt.NoPen)
    painter.setBrush(QBrush(QColor(30, 30, 70)))
    painter.drawRoundedRect(0, 0, size, size, 10, 10)
    
    # Draw a decay curve
    painter.setPen(QPen(QColor(0, 200, 0), 2))
    points = []
    for i in range(size):
        x = i
        y = int(size * 0.2 + size * 0.6 * (1 - i / size) ** 2)
        points.append(QPoint(x, y))
    
    for i in range(len(points) - 1):
        painter.drawLine(points[i], points[i + 1])
    
    # Draw a correlation curve
    painter.setPen(QPen(QColor(0, 150, 200), 2))
    points = []
    for i in range(size):
        x = i
        y = int(size * 0.8 - size * 0.3 * (1 - i / size) ** 0.5)
        points.append(QPoint(x, y))
    
    for i in range(len(points) - 1):
        painter.drawLine(points[i], points[i + 1])
    
    # Draw text
    painter.setPen(QColor(255, 255, 255))
    font = QFont("Arial", size // 8, QFont.Bold)
    painter.setFont(font)
    painter.drawText(QRect(0, 0, size, size // 4), Qt.AlignCenter, "SMACQ")
    
    # End painting
    painter.end()
    
    # Save the pixmap
    pixmap.save(output_path)
    
    print(f"Icon saved to {output_path}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "icon.png")
    create_icon(output_path)
