"""Custom title bar for MDI sub-windows with full QSS styling support"""

from __future__ import annotations

from qtpy import QtWidgets, QtCore, QtGui


def shorten_text_middle(text: str, font_metrics: QtGui.QFontMetrics, max_width: int, min_chars: int = 10) -> str:
    """Shorten text with ellipsis in the middle to fit within max_width.
    
    Args:
        text: The text to shorten
        font_metrics: QFontMetrics to measure text width
        max_width: Maximum width in pixels
        min_chars: Minimum number of characters to show (won't shorten below this)
    
    Returns:
        Shortened text with '...' in the middle if needed
    """
    if not text:
        return text
    
    # Check if text already fits
    if font_metrics.horizontalAdvance(text) <= max_width:
        return text
    
    # Don't shorten very short text
    if len(text) <= min_chars:
        return text
    
    ellipsis = "..."
    ellipsis_width = font_metrics.horizontalAdvance(ellipsis)
    
    # Binary search for the right amount of text to keep
    left = 1
    right = len(text) - 1
    best_text = text[:min_chars // 2] + ellipsis + text[-(min_chars // 2):]
    
    while left <= right:
        mid = (left + right) // 2
        # Keep mid characters from start and mid from end
        start_chars = mid
        end_chars = mid
        
        shortened = text[:start_chars] + ellipsis + text[-end_chars:] if end_chars > 0 else text[:start_chars] + ellipsis
        width = font_metrics.horizontalAdvance(shortened)
        
        if width <= max_width:
            best_text = shortened
            left = mid + 1
        else:
            right = mid - 1
    
    return best_text


class CustomTitleBar(QtWidgets.QWidget):
    """Custom title bar for MDI sub-windows that is fully stylable via QSS
    
    The title bar provides minimize, maximize/restore, and close buttons,
    and supports dragging the window. All visual styling is controlled
    through QSS using object names.
    
    Object names for QSS styling:
    - customTitleBar: The title bar widget itself
    - titleLabel: The window title label
    - minimizeButton: The minimize button
    - maximizeButton: The maximize/restore button
    - closeButton: The close button
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.dragging = False
        self.offset = QtCore.QPoint()
        self._full_title = "Document"  # Store the full title

        self.setObjectName("customTitleBar")
        self.setFixedHeight(28)

        self.setup_ui()

    def setup_ui(self):
        """Setup the title bar UI components"""
        layout = QtWidgets.QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(2)

        # Window icon (optional, can be set later)
        self.icon_label = QtWidgets.QLabel()
        self.icon_label.setObjectName("titleIcon")
        self.icon_label.setFixedSize(16, 16)
        self.icon_label.setScaledContents(True)
        self.icon_label.hide()  # Hidden by default
        layout.addWidget(self.icon_label)

        # Window title
        self.title_label = QtWidgets.QLabel("Document")
        self.title_label.setObjectName("titleLabel")
        layout.addWidget(self.title_label)

        # Spacer
        layout.addStretch()

        # Window control buttons
        self.minimize_btn = QtWidgets.QPushButton("−")
        self.maximize_btn = QtWidgets.QPushButton("□")
        self.close_btn = QtWidgets.QPushButton("×")

        self.minimize_btn.setObjectName("minimizeButton")
        self.maximize_btn.setObjectName("maximizeButton")
        self.close_btn.setObjectName("closeButton")

        # Set fixed sizes for buttons
        for btn in [self.minimize_btn, self.maximize_btn, self.close_btn]:
            btn.setFixedSize(24, 24)

        # Connect buttons
        self.minimize_btn.clicked.connect(self.minimize_window)
        self.maximize_btn.clicked.connect(self.toggle_maximize)
        self.close_btn.clicked.connect(self.close_window)

        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)

        self.setLayout(layout)

    def set_title(self, title: str):
        """Set window title"""
        self._full_title = title
        self._update_title_display()
    
    def _update_title_display(self):
        """Update the displayed title based on available width"""
        if not hasattr(self, 'title_label') or not hasattr(self, '_full_title'):
            return
        
        # Calculate available width for title
        # Account for: icon, buttons, margins, and spacing
        total_width = self.width()
        icon_width = self.icon_label.width() + 5 if self.icon_label.isVisible() else 0
        buttons_width = (self.minimize_btn.width() + self.maximize_btn.width() + 
                        self.close_btn.width() + 3 * 2)  # 3 buttons + spacing
        margins = 10  # Left and right margins
        available_width = total_width - icon_width - buttons_width - margins - 20  # Extra padding
        
        # Ensure we have some minimum width
        if available_width < 50:
            available_width = 50
        
        # Shorten the title if needed
        font_metrics = self.title_label.fontMetrics()
        shortened_title = shorten_text_middle(self._full_title, font_metrics, available_width)
        
        self.title_label.setText(shortened_title)
        
        # Set tooltip to show full title if shortened
        if shortened_title != self._full_title:
            self.title_label.setToolTip(self._full_title)
        else:
            self.title_label.setToolTip("")

    def set_icon(self, icon: QtGui.QIcon):
        """Set window icon"""
        if not icon.isNull():
            pixmap = icon.pixmap(16, 16)
            self.icon_label.setPixmap(pixmap)
            self.icon_label.show()
        else:
            self.icon_label.hide()

    def mousePressEvent(self, event):
        """Start dragging window"""
        if event.button() == QtCore.Qt.LeftButton:
            self.dragging = True
            self.offset = event.pos()

    def mouseMoveEvent(self, event):
        """Drag window"""
        if self.dragging and self.parent_window:
            new_pos = self.parent_window.pos() + event.pos() - self.offset
            self.parent_window.move(new_pos)

    def mouseReleaseEvent(self, event):
        """Stop dragging"""
        self.dragging = False

    def mouseDoubleClickEvent(self, event):
        """Toggle maximize on double click"""
        if event.button() == QtCore.Qt.LeftButton:
            self.toggle_maximize()
    
    def resizeEvent(self, event):
        """Update title display when title bar is resized"""
        super().resizeEvent(event)
        self._update_title_display()

    def minimize_window(self):
        """Minimize the window"""
        if self.parent_window:
            self.parent_window.showMinimized()

    def toggle_maximize(self):
        """Toggle between normal and maximized state"""
        if self.parent_window:
            if self.parent_window.isMaximized():
                self.parent_window.showNormal()
                self.maximize_btn.setText("□")
            else:
                self.parent_window.showMaximized()
                self.maximize_btn.setText("❐")

    def close_window(self):
        """Close the window"""
        if self.parent_window:
            self.parent_window.close()


class CustomMdiSubWindow(QtWidgets.QMdiSubWindow):
    """Custom MDI sub-window with custom title bar decorations
    
    This class provides an MDI sub-window with a custom title bar
    that can be fully styled via QSS. The window includes a resize grip
    for resizing from the bottom-right corner. When minimized, it appears
    as an icon in the MDI area.
    
    Object names for QSS styling:
    - windowContainer: The main container frame
    - customTitleBar: The title bar (see CustomTitleBar for details)
    """

    def __init__(self, title="Document", parent=None):
        super().__init__(parent)

        # Store original geometry for restore
        self._normal_geometry = None
        self._is_minimized = False
        
        # Remove default window frame - we'll handle minimize ourselves
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)

        # Create main container
        self.container = QtWidgets.QFrame()
        self.container.setObjectName("windowContainer")

        # Create layout for container
        container_layout = QtWidgets.QVBoxLayout()
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        # Add custom title bar
        self.title_bar = CustomTitleBar(self)
        self.title_bar.set_title(title)
        container_layout.addWidget(self.title_bar)

        # Content area
        self.content_widget = QtWidgets.QWidget()
        self.content_layout = QtWidgets.QVBoxLayout()
        self.content_layout.setContentsMargins(5, 5, 5, 5)
        self.content_widget.setLayout(self.content_layout)
        container_layout.addWidget(self.content_widget)

        self.container.setLayout(container_layout)
        self.setWidget(self.container)

        # Add resize grip
        self.size_grip = QtWidgets.QSizeGrip(self.container)
        self.size_grip.setFixedSize(15, 15)
        self.size_grip.setStyleSheet("""
            QSizeGrip {
                background-color: transparent;
                image: none;
            }
        """)
        
        # Monitor for view mode changes
        self._check_view_mode_timer = QtCore.QTimer()
        self._check_view_mode_timer.timeout.connect(self._update_title_bar_visibility)
        self._check_view_mode_timer.start(500)  # Check every 500ms
        self._last_view_mode = None

    def resizeEvent(self, event):
        """Position size grip in bottom right corner and update title"""
        super().resizeEvent(event)
        self.size_grip.move(
            self.container.width() - self.size_grip.width(),
            self.container.height() - self.size_grip.height()
        )
        # Update title bar display when window is resized
        if hasattr(self, 'title_bar'):
            self.title_bar._update_title_display()

    def set_content(self, widget):
        """Set the content widget"""
        # Clear existing content
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add new content
        self.content_layout.addWidget(widget)

    def set_title(self, title: str):
        """Set window title"""
        self.title_bar.set_title(title)
        self.setWindowTitle(title)  # Also set the actual window title

    def setWindowTitle(self, title: str):
        """Override to update both the actual title and custom title bar"""
        super().setWindowTitle(title)
        if hasattr(self, 'title_bar'):
            self.title_bar.set_title(title)

    def setWindowIcon(self, icon: QtGui.QIcon):
        """Override to update both the actual icon and custom title bar"""
        super().setWindowIcon(icon)
        if hasattr(self, 'title_bar'):
            self.title_bar.set_icon(icon)
    
    def showMinimized(self):
        """Custom minimize behavior - show as icon in MDI area"""
        if self._is_minimized:
            return
            
        # Store current geometry
        self._normal_geometry = self.geometry()
        self._is_minimized = True
        
        # Hide the main container but keep the window visible
        self.container.hide()
        
        # Resize to icon size (width x height for minimized icon)
        icon_width = 160
        icon_height = 32
        
        # Position at bottom of MDI area
        if self.mdiArea():
            mdi_rect = self.mdiArea().viewport().rect()
            # Find a spot at the bottom for this minimized window
            x_pos = 5
            y_pos = mdi_rect.height() - icon_height - 5
            
            # Check for other minimized windows and position accordingly
            for window in self.mdiArea().subWindowList():
                if window != self and hasattr(window, '_is_minimized') and window._is_minimized:
                    other_geom = window.geometry()
                    if abs(other_geom.y() - y_pos) < icon_height:
                        x_pos = max(x_pos, other_geom.x() + other_geom.width() + 5)
            
            self.setGeometry(x_pos, y_pos, icon_width, icon_height)
        else:
            self.resize(icon_width, icon_height)
        
        # Create minimized icon widget if not exists
        if not hasattr(self, 'minimized_widget'):
            self.minimized_widget = QtWidgets.QWidget()
            self.minimized_widget.setObjectName("minimizedIcon")
            
            layout = QtWidgets.QHBoxLayout(self.minimized_widget)
            layout.setContentsMargins(5, 2, 5, 2)
            layout.setSpacing(5)
            
            # Icon
            self.min_icon_label = QtWidgets.QLabel()
            self.min_icon_label.setFixedSize(16, 16)
            self.min_icon_label.setScaledContents(True)
            if not self.windowIcon().isNull():
                self.min_icon_label.setPixmap(self.windowIcon().pixmap(16, 16))
            layout.addWidget(self.min_icon_label)
            
            # Title
            self.min_title_label = QtWidgets.QLabel(self.windowTitle())
            self.min_title_label.setObjectName("minimizedTitle")
            layout.addWidget(self.min_title_label)
            layout.addStretch()
            
            # Restore button
            self.restore_btn = QtWidgets.QPushButton("▢")
            self.restore_btn.setObjectName("restoreButton")
            self.restore_btn.setFixedSize(20, 20)
            self.restore_btn.clicked.connect(self.showNormal)
            layout.addWidget(self.restore_btn)
            
            # Make the widget clickable to restore
            self.minimized_widget.mouseDoubleClickEvent = lambda e: self.showNormal()
        
        # Replace the main widget with minimized icon
        self.setWidget(self.minimized_widget)
        self.minimized_widget.show()
    
    def showNormal(self):
        """Restore from minimized state"""
        if not self._is_minimized:
            super().showNormal()
            return
            
        self._is_minimized = False
        
        # Restore the container
        self.setWidget(self.container)
        self.container.show()
        
        # Restore geometry
        if self._normal_geometry:
            self.setGeometry(self._normal_geometry)
        else:
            # Default size if no stored geometry
            self.resize(450, 350)
        
        super().showNormal()
    
    def showMaximized(self):
        """Ensure we're not minimized when maximizing"""
        if self._is_minimized:
            self.showNormal()
        super().showMaximized()
    
    def _update_title_bar_visibility(self):
        """Update title bar visibility based on MDI view mode"""
        mdi = self.mdiArea()
        if not mdi:
            return
        
        # Check if MDI is in tabbed view mode
        is_tabbed = mdi.viewMode() == QtWidgets.QMdiArea.TabbedView
        
        # Only update if mode changed
        if is_tabbed != self._last_view_mode:
            self._last_view_mode = is_tabbed
            
            if is_tabbed:
                # Hide custom title bar in tabbed mode
                self.title_bar.hide()
                self.size_grip.hide()
                # Adjust content margins
                self.content_layout.setContentsMargins(2, 2, 2, 2)
            else:
                # Show custom title bar in subwindow mode
                self.title_bar.show()
                self.size_grip.show()
                # Restore content margins
                self.content_layout.setContentsMargins(5, 5, 5, 5)
    
    def showEvent(self, event):
        """Update title bar visibility when window is shown"""
        super().showEvent(event)
        self._update_title_bar_visibility()
    
    def closeEvent(self, event):
        """Clean up timer on close"""
        if hasattr(self, '_check_view_mode_timer'):
            self._check_view_mode_timer.stop()
        super().closeEvent(event)
