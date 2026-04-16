from qtpy import QtCore, QtGui, QtWidgets

from .constants import RibbonButtonStyle
from .menu import RibbonMenu


def handle_ribbon_context_menu(widget: QtWidgets.QWidget, event: QtGui.QContextMenuEvent):
    """Handle right-click context menu for ribbon buttons."""
    # Find ribbon bar
    ribbon = widget
    while ribbon is not None and ribbon.__class__.__name__ != 'RibbonBar':
        ribbon = ribbon.parent()
        
    if ribbon is None:
        if isinstance(widget, QtWidgets.QToolButton):
            QtWidgets.QToolButton.contextMenuEvent(widget, event)
        else:
            QtWidgets.QWidget.contextMenuEvent(widget, event)
        return
        
    # Find panel and category
    panel = widget
    while panel is not None and panel.__class__.__name__ != 'RibbonPanel':
        panel = panel.parent()
        
    category = widget
    while category is not None and 'Category' not in category.__class__.__name__:
        category = category.parent()
        
    panel_title = getattr(panel, 'title', lambda: "UnknownPanel")() if panel else "UnknownPanel"
    category_title = getattr(category, 'title', lambda: "UnknownCategory")() if category else "UnknownCategory"
    
    # RibbonSplitButton text logic
    if hasattr(widget, 'text'):
        text = widget.text()
    elif hasattr(widget, '_actionButton'):
        text = widget._actionButton.text()
    else:
        text = ""
    text = text.replace('\n', ' ').strip()
    
    # Calculate Button ID
    btn_id = f"{category_title}::{panel_title}::{text}"
    
    menu = QtWidgets.QMenu(widget)
    
    # Status
    is_in_qat = hasattr(ribbon, '_qat_button_ids') and btn_id in ribbon._qat_button_ids
    
    qat_action = menu.addAction("Remove from Quick Access Toolbar" if is_in_qat else "Add to Quick Access Toolbar")
    menu.addSeparator()
    hide_action = menu.addAction("Hide this item")
    
    action = menu.exec_(event.globalPos())
    
    if action == qat_action:
        if is_in_qat:
            if hasattr(ribbon, 'removeButtonFromQuickAccess'):
                ribbon.removeButtonFromQuickAccess(btn_id)
        else:
            if hasattr(ribbon, 'addButtonToQuickAccess'):
                ribbon.addButtonToQuickAccess(btn_id, widget)
    elif action == hide_action:
        if hasattr(ribbon, 'hideButton'):
            ribbon.hideButton(btn_id, widget)

class RibbonMenuButton(QtWidgets.QToolButton):
    """Menu button with dropdown arrow for ribbon."""
    
    def __init__(self, parent=None):
        """Create a new menu button.
        
        :param parent: The parent widget.
        """
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        
    def setMenu(self, menu: QtWidgets.QMenu):
        """Set the menu for the button.
        
        :param menu: The menu to set.
        """
        super().setMenu(menu)
        # Ensure proper styling for menu indicator
        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.MenuButtonPopup)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        handle_ribbon_context_menu(self, event)


class RibbonDelayedMenuButton(QtWidgets.QToolButton):
    """Delayed popup menu button for ribbon."""
    
    def __init__(self, parent=None):
        """Create a new delayed menu button.
        
        :param parent: The parent widget.
        """
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.DelayedPopup)
        
    def setMenu(self, menu: QtWidgets.QMenu):
        """Set the menu for the button.
        
        :param menu: The menu to set.
        """
        super().setMenu(menu)
        self.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.DelayedPopup)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        handle_ribbon_context_menu(self, event)


class RibbonSplitButton(QtWidgets.QWidget):
    """Split button with action and dropdown menu for ribbon."""
    
    #: Signal emitted when the action part is clicked
    actionClicked = QtCore.Signal()
    
    def __init__(self, parent=None):
        """Create a new split button.
        
        :param parent: The parent widget.
        """
        super().__init__(parent)
        
        self._mainLayout = QtWidgets.QHBoxLayout(self)
        self._mainLayout.setContentsMargins(0, 0, 0, 0)
        self._mainLayout.setSpacing(0)
        
        # Action button
        self._actionButton = QtWidgets.QToolButton()
        self._actionButton.setAutoRaise(True)
        self._actionButton.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._actionButton.clicked.connect(self.actionClicked)
        
        # Menu button
        self._menuButton = QtWidgets.QToolButton()
        self._menuButton.setAutoRaise(True)
        self._menuButton.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._menuButton.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self._menuButton.setText("v")
        self._menuButton.setFixedSize(16, 16)
        
        self._mainLayout.addWidget(self._actionButton)
        self._mainLayout.addWidget(self._menuButton)
        
    def setAction(self, action: QtWidgets.QAction):
        """Set the action for the button.
        
        :param action: The action to set.
        """
        self._actionButton.setText(action.text())
        self._actionButton.setIcon(action.icon())
        self._actionButton.setToolTip(action.toolTip())
        self._actionButton.setStatusTip(action.statusTip())
        self._actionButton.setShortcut(action.shortcut())
        
    def setMenu(self, menu: QtWidgets.QMenu):
        """Set the menu for the dropdown.
        
        :param menu: The menu to set.
        """
        self._menuButton.setMenu(menu)
        
    def setText(self, text: str):
        """Set the button text.
        
        :param text: The text to set.
        """
        self._actionButton.setText(text)
        
    def setIcon(self, icon: QtGui.QIcon):
        """Set the button icon.
        
        :param icon: The icon to set.
        """
        self._actionButton.setIcon(icon)
        
    def setToolTip(self, tooltip: str):
        """Set the button tooltip.
        
        :param tooltip: The tooltip to set.
        """
        self._actionButton.setToolTip(tooltip)
        
    def actionButton(self) -> QtWidgets.QToolButton:
        """Get the action button.
        
        :return: The action button.
        """
        return self._actionButton
        
    def menuButton(self) -> QtWidgets.QToolButton:
        """Get the menu button.
        
        :return: The menu button.
        """
        return self._menuButton

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        handle_ribbon_context_menu(self, event)


class RibbonToolButton(QtWidgets.QToolButton):
    """Tool button that is showed in the ribbon."""

    _buttonStyle: RibbonButtonStyle

    _largeButtonIconSize = 64
    _mediumButtonIconSize = 48
    _smallButtonIconSize = 32

    _maximumIconSize = 64

    def __init__(self, parent=None):
        """Create a new ribbon tool button.

        :param parent: The parent widget.
        """
        super().__init__(parent)

        # Styles
        self.setButtonStyle(RibbonButtonStyle.Large)
        self.setAutoRaise(True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent):
        handle_ribbon_context_menu(self, event)

    def setMaximumIconSize(self, size: int):
        """Set the maximum icon size of the button.

        :param size: The maximum icon size of the button.
        """
        self._maximumIconSize = size
        self.setButtonStyle(self._buttonStyle)

    def maximumIconSize(self) -> int:
        """Get the maximum icon size of the button.

        :return: The maximum icon size of the button.
        """
        return self._maximumIconSize

    def setButtonStyle(self, style: RibbonButtonStyle):
        """Set the button style of the button.

        :param style: The button style of the button.
        """
        self._buttonStyle = style
        if style == RibbonButtonStyle.Small:
            height = self._smallButtonIconSize
            height = min(height, self._maximumIconSize)
            self.setIconSize(QtCore.QSize(height, height))
            self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.setStyleSheet(
                """
                RibbonToolButton::menu-indicator {
                    subcontrol-origin: padding;
                    subcontrol-position: right;
                    right: -5px;
                }
                """
            )
        elif style == RibbonButtonStyle.Medium:
            height = self._mediumButtonIconSize
            height = min(height, self._maximumIconSize)
            self.setIconSize(QtCore.QSize(height, height))
            self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.setStyleSheet(
                """
                RibbonToolButton::menu-indicator {
                    subcontrol-origin: padding;
                    subcontrol-position: right;
                    right: -5px;
                }
                """
            )
        elif style == RibbonButtonStyle.Large:
            height = self._largeButtonIconSize
            height = min(height, self._maximumIconSize)
            self.setIconSize(QtCore.QSize(height, height))
            self.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            self.setStyleSheet(
                """
                RibbonToolButton[popupMode="0"]::menu-indicator {
                    subcontrol-origin: padding;
                    subcontrol-position: bottom;
                    bottom: -5px;
                }
                RibbonToolButton[popupMode="2"]::menu-indicator {
                    subcontrol-origin: padding;
                    subcontrol-position: bottom;
                    bottom: -5px;
                }
                """
            )

    def buttonStyle(self) -> RibbonButtonStyle:
        """Get the button style of the button.

        :return: The button style of the button.
        """
        return self._buttonStyle

    def addRibbonMenu(self) -> RibbonMenu:
        """Add a ribbon menu for the button.

        :return: The added ribbon menu.
        """
        menu = RibbonMenu()
        self.setMenu(menu)
        return menu
        
    def addMenuButton(self) -> RibbonMenuButton:
        """Convert this button to a menu button.
        
        :return: A new menu button with the same properties.
        """
        menu_button = RibbonMenuButton(self.parent())
        menu_button.setText(self.text())
        menu_button.setIcon(self.icon())
        menu_button.setToolTip(self.toolTip())
        menu_button.setStatusTip(self.statusTip())
        menu_button.setShortcut(self.shortcut())
        menu_button.setButtonStyle(self._buttonStyle)
        return menu_button
        
    def addDelayedMenuButton(self) -> RibbonDelayedMenuButton:
        """Convert this button to a delayed menu button.
        
        :return: A new delayed menu button with the same properties.
        """
        delayed_button = RibbonDelayedMenuButton(self.parent())
        delayed_button.setText(self.text())
        delayed_button.setIcon(self.icon())
        delayed_button.setToolTip(self.toolTip())
        delayed_button.setStatusTip(self.statusTip())
        delayed_button.setShortcut(self.shortcut())
        delayed_button.setButtonStyle(self._buttonStyle)
        return delayed_button
        
    def createSplitButton(self) -> RibbonSplitButton:
        """Create a split button from this button.
        
        :return: A new split button with the same properties.
        """
        split_button = RibbonSplitButton(self.parent())
        split_button.setText(self.text())
        split_button.setIcon(self.icon())
        split_button.setToolTip(self.toolTip())
        split_button.setStatusTip(self.statusTip())
        split_button.setShortcut(self.shortcut())
        return split_button
