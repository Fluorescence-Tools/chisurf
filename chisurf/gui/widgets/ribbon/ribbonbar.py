import typing

from qtpy import QtCore, QtGui, QtWidgets

from .category import (
    RibbonCategory,
    RibbonContextCategories,
    RibbonContextCategory,
    RibbonNormalCategory,
)
from .constants import RibbonCategoryStyle, RibbonStyle, contextColors
from .menu import RibbonMenu
from .tabbar import RibbonTabBar
from .titlewidget import RibbonApplicationButton, RibbonTitleWidget


class RibbonStackedWidget(QtWidgets.QStackedWidget):
    """Stacked widget that is used to display the ribbon."""

    def __init__(self, parent=None):
        """Create a new ribbon stacked widget.

        :param parent: The parent widget.
        """
        super().__init__(parent)
        effect = QtWidgets.QGraphicsDropShadowEffect()
        effect.setOffset(2, 2)
        self.setGraphicsEffect(effect)


class RibbonBar(QtWidgets.QWidget):
    """The RibbonBar class is the top level widget that contains the ribbon."""

    #: Signal, the help button was clicked.
    helpButtonClicked = QtCore.Signal(bool)

    #: hide the ribbon bar automatically when the mouse press outside the ribbon bar
    _autoHideRibbon = False

    #: The categories of the ribbon.
    _categories: typing.Dict[str, RibbonCategory] = {}
    _contextCategoryCount = 0

    #: Maximum rows
    _maxRows = 6

    #: Whether the ribbon is visible.
    _ribbonVisible = True

    #: heights of the ribbon elements
    _ribbonHeight = 110

    #: current tab index
    _currentTabIndex = 0

    @typing.overload
    def __init__(self, title: str = "Ribbon Bar Title", maxRows=6, parent=None):
        pass

    @typing.overload
    def __init__(self, parent=None):
        pass

    def __init__(self, *args, **kwargs):
        """Create a new ribbon.

        :param title: The title of the ribbon.
        :param maxRows: The maximum number of rows.
        :param parent: The parent widget of the ribbon.
        """
        if (args and not isinstance(args[0], QtWidgets.QWidget)) or ("title" in kwargs or "maxRows" in kwargs):
            title = args[0] if len(args) > 0 else kwargs.get("title", "Ribbon Bar Title")
            maxRows = args[1] if len(args) > 1 else kwargs.get("maxRows", 6)
            parent = args[2] if len(args) > 2 else kwargs.get("parent", None)
        else:
            title = ""
            maxRows = 6
            parent = args[1] if len(args) > 1 else kwargs.get("parent", None)
        super().__init__(parent)
        self._categories = {}
        self._maxRows = maxRows
        self.setFixedHeight(self._ribbonHeight)

        self._titleWidget = RibbonTitleWidget(title, self)
        self._stackedWidget = RibbonStackedWidget(self)

        # Main layout
        self._mainLayout = QtWidgets.QVBoxLayout(self)
        self._mainLayout.setContentsMargins(0, 0, 0, 0)
        self._mainLayout.setSpacing(0)
        self._mainLayout.addWidget(self._titleWidget, 0)
        self._mainLayout.addWidget(self._stackedWidget, 1)
        self._mainLayout.setSizeConstraint(QtWidgets.QLayout.SizeConstraint.SetMinAndMaxSize)

        # Search functionality
        self._searchField = QtWidgets.QLineEdit(self)
        self._searchField.setPlaceholderText("Search...")
        self._searchField.setClearButtonEnabled(True)
        self._searchField.setFixedWidth(150)
        self._searchField.setStyleSheet("""
            QLineEdit {
                background-color: #454545;
                color: white;
                border: 1px solid #666;
                border-radius: 3px;
                padding: 2px 5px;
            }
            QLineEdit:focus {
                border: 1px solid #1e90ff;
            }
        """)
        self._searchField.textChanged.connect(self._onSearchChanged)
        
        # Add search field to right toolbar (to the left of other buttons)
        # Use insertWidget before the collapse button
        self._titleWidget.rightToolBar().insertWidget(self._titleWidget._collapseRibbonButtonAction, self._searchField)

        # Connect signals
        self._titleWidget.helpButtonClicked.connect(self.helpButtonClicked)
        self._titleWidget.collapseRibbonButtonClicked.connect(self._collapseButtonClicked)
        self._titleWidget.tabBar().currentChanged.connect(self.showCategoryByIndex)  # type: ignore
        self._titleWidget.tabBar().doubleClicked.connect(self._onTabBarDoubleClicked)  # type: ignore
        
        # Ribbon state
        self._qat_button_ids = []
        self._hidden_button_ids = []
        self._qat_buttons = {}  # btn_id -> QToolButton
        
        self._loadRibbonState()
        
        # Add context menu to ribbon bar itself to unhide items
        self.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._showRibbonContextMenu)
        
        self.setRibbonStyle(RibbonStyle.Default)

    def _loadRibbonState(self):
        """Load QAT and Hidden state from QSettings."""
        settings = QtCore.QSettings("ChiSurf", "RibbonState")
        val = settings.value("qat_buttons", [])
        self._qat_button_ids = val if isinstance(val, list) else [val] if val else []
        val2 = settings.value("hidden_buttons", [])
        self._hidden_button_ids = val2 if isinstance(val2, list) else [val2] if val2 else []
            
    def _saveRibbonState(self):
        """Save QAT and Hidden state to QSettings."""
        settings = QtCore.QSettings("ChiSurf", "RibbonState")
        settings.setValue("qat_buttons", self._qat_button_ids)
        settings.setValue("hidden_buttons", self._hidden_button_ids)

    def registerTargetButton(self, button: QtWidgets.QWidget):
        """Called when a panel adds a button. Process hiding and QAT."""
        btn_id = getattr(button, '_ribbon_btn_id', None)
        if btn_id is None:
            # Calculate it dynamically
            panel = button
            while panel is not None and panel.__class__.__name__ != 'RibbonPanel':
                panel = panel.parent()
                
            category = button
            while category is not None and 'Category' not in category.__class__.__name__:
                category = category.parent()
                
            panel_title = getattr(panel, 'title', lambda: "UnknownPanel")() if panel else "UnknownPanel"
            category_title = getattr(category, 'title', lambda: "UnknownCategory")() if category else "UnknownCategory"
            
            if hasattr(button, 'text'):
                text = button.text()
            elif hasattr(button, '_actionButton'):
                text = button._actionButton.text()
            else:
                text = ""
            text = text.replace('\n', ' ').strip()
            
            btn_id = f"{category_title}::{panel_title}::{text}"
            button._ribbon_btn_id = btn_id
            
        if btn_id in self._hidden_button_ids:
            # Hide the RibbonPanelItemWidget parent
            parent = button.parent()
            if parent and parent.__class__.__name__ == 'RibbonPanelItemWidget':
                parent.hide()
            else:
                button.hide()
                
            panel = button
            while panel is not None and panel.__class__.__name__ != 'RibbonPanel':
                panel = panel.parent()
            if panel and hasattr(panel, 'reflow'):
                panel.reflow()
            
        if btn_id in self._qat_button_ids:
            if btn_id not in self._qat_buttons:
                self.addButtonToQuickAccess(btn_id, button, save=False)

    def addButtonToQuickAccess(self, btn_id: str, source_button: QtWidgets.QWidget, save=True):
        """Replicate a button and add it to the quick access toolbar."""
        if btn_id in self._qat_buttons:
            return
            
        qat_button = QtWidgets.QToolButton()
        
        # Pull properties
        icon = getattr(source_button, '_ribbon_icon', None) or (source_button.icon() if hasattr(source_button, 'icon') else None)
        text = getattr(source_button, '_ribbon_text', None) or (source_button.text() if hasattr(source_button, 'text') else "")
        tooltip = getattr(source_button, '_ribbon_tooltip', None) or (source_button.toolTip() if hasattr(source_button, 'toolTip') else "")
        slot = getattr(source_button, '_ribbon_slot', None)
        
        if icon:
            qat_button.setIcon(icon)
        else:
            qat_button.setText(text)
            
        if tooltip:
            qat_button.setToolTip(tooltip)
        else:
            qat_button.setToolTip(text)
            
        if slot:
            qat_button.clicked.connect(slot)
            
        qat_button.setAutoRaise(True)
        qat_button.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        
        def qat_context_menu(pos):
            menu = QtWidgets.QMenu(qat_button)
            remove_action = menu.addAction("Remove from Quick Access Toolbar")
            action = menu.exec_(qat_button.mapToGlobal(pos))
            if action == remove_action:
                self.removeButtonFromQuickAccess(btn_id)
                
        qat_button.customContextMenuRequested.connect(qat_context_menu)
        
        self.addQuickAccessButton(qat_button)
        self._qat_buttons[btn_id] = qat_button
        
        if btn_id not in self._qat_button_ids:
            self._qat_button_ids.append(btn_id)
            
        if save:
            self._saveRibbonState()

    def removeButtonFromQuickAccess(self, btn_id: str):
        """Remove a previously added button from QAT."""
        if btn_id in self._qat_button_ids:
            self._qat_button_ids.remove(btn_id)
            self._saveRibbonState()
            
        if btn_id in self._qat_buttons:
            qat_button = self._qat_buttons.pop(btn_id)
            # Remove action from toolbar
            self._titleWidget.quickAccessToolBar().removeAction(qat_button.defaultAction())
            action = getattr(qat_button, 'defaultAction', lambda: None)()
            if action:
                self._titleWidget.quickAccessToolBar().removeAction(action)
            else:
                 # Workaround: find the layout and remove it, or use setParent(None)
                 qat_button.setParent(None)
            qat_button.deleteLater()

    def hideButton(self, btn_id: str, button: QtWidgets.QWidget):
        """Hide a button from the ribbon."""
        if btn_id not in self._hidden_button_ids:
            self._hidden_button_ids.append(btn_id)
            self._saveRibbonState()
            
        parent = button.parent()
        if parent and parent.__class__.__name__ == 'RibbonPanelItemWidget':
            parent.hide()
        else:
            button.hide()
            
        panel = button
        while panel is not None and panel.__class__.__name__ != 'RibbonPanel':
            panel = panel.parent()
        if panel and hasattr(panel, 'reflow'):
            panel.reflow()

    def _showRibbonContextMenu(self, pos: QtCore.QPoint):
        """Context menu for the ribbon bar background."""
        if not self._hidden_button_ids:
            return
            
        menu = QtWidgets.QMenu(self)
        reset_action = menu.addAction("Show All Hidden Items")
        
        action = menu.exec_(self.mapToGlobal(pos))
        if action == reset_action:
            self._hidden_button_ids.clear()
            self._saveRibbonState()
            
            for category in self._categories.values():
                for panel in category.panels().values():
                    for widget in panel.widgets():
                        parent = widget.parent()
                        if parent and parent.__class__.__name__ == 'RibbonPanelItemWidget':
                            parent.show()
                        widget.show()
                        
                        btn_id = getattr(widget, '_ribbon_btn_id', None)
                        if btn_id in self._hidden_button_ids:
                            self._hidden_button_ids.remove(btn_id)
                            
                    if hasattr(panel, 'reflow'):
                        panel.reflow()
                        
    def _onSearchChanged(self, text: str):
        """Handle search field text changes."""
        text = text.lower().strip()
        first_match_category = None
        
        for category_name, category in self._categories.items():
            category_has_match = False
            for panel_name, panel in category.panels().items():
                for widget in panel.widgets():
                    matches = False
                    if text:
                        # Check text and tooltip
                        widget_text = ""
                        if hasattr(widget, 'text'):
                            widget_text = widget.text().lower()
                        elif hasattr(widget, 'title'):
                            widget_text = widget.title().lower()
                        
                        tooltip = widget.toolTip().lower()
                        
                        if text in widget_text or text in tooltip:
                            matches = True
                            category_has_match = True
                            if first_match_category is None:
                                first_match_category = category
                    
                    # Apply highlighting
                    if matches:
                        widget.setStyleSheet("border: 2px solid orange; border-radius: 3px;")
                    else:
                        widget.setStyleSheet("")
            
            # Highlight category tab if it has matches
            tab_bar = self._titleWidget.tabBar()
            index = tab_bar.indexOf(category_name)
            if index >= 0:
                if text and category_has_match:
                    tab_bar.setTabTextColor(index, QtGui.QColor("orange"))
                else:
                    tab_bar.setTabTextColor(index, QtGui.QColor("white"))

        # Switch to the first category with a match and show ribbon if folded
        if first_match_category:
            self.setCurrentCategory(first_match_category)
            if not self.ribbonVisible():
                self.showRibbon()

    def autoHideRibbon(self) -> bool:
        """Return whether the ribbon bar is automatically hidden when the mouse is pressed outside the ribbon bar.

        :return: Whether the ribbon bar is automatically hidden.
        """
        return self._autoHideRibbon

    def setAutoHideRibbon(self, autoHide: bool):
        """Set whether the ribbon bar is automatically hidden when the mouse is pressed outside the ribbon bar.

        :param autoHide: Whether the ribbon bar is automatically hidden.
        """
        self._autoHideRibbon = autoHide

    def eventFilter(self, a0: QtCore.QObject, a1: QtCore.QEvent) -> bool:
        if self._autoHideRibbon and a1.type() == QtCore.QEvent.Type.HoverMove:
            self.setRibbonVisible(self.underMouse())
        return super().eventFilter(a0, a1)

    def actionAt(self, QPoint):
        raise NotImplementedError("RibbonBar.actionAt() is not implemented in the ribbon bar.")

    def actionGeometry(self, QAction):
        raise NotImplementedError("RibbonBar.actionGeometry() is not implemented in the ribbon bar.")

    def activeAction(self):
        raise NotImplementedError("RibbonBar.activeAction() is not implemented in the ribbon bar.")

    def addMenu(self, *__args):
        raise NotImplementedError("RibbonBar.addMenu() is not implemented in the ribbon bar.")

    def addAction(self, *__args):
        raise NotImplementedError("RibbonBar.addAction() is not implemented in the ribbon bar.")

    def addSeparator(self):
        raise NotImplementedError("RibbonBar.addSeparator() is not implemented in the ribbon bar.")

    def clear(self):
        raise NotImplementedError("RibbonBar.clear() is not implemented in the ribbon bar.")

    def cornerWidget(self, corner=None, *args, **kwargs):
        raise NotImplementedError("RibbonBar.cornerWidget() is not implemented in the ribbon bar.")

    def insertMenu(self, QAction, QMenu):
        raise NotImplementedError("RibbonBar.insertMenu() is not implemented in the ribbon bar.")

    def insertSeparator(self, QAction):
        raise NotImplementedError("RibbonBar.insertSeparator() is not implemented in the ribbon bar.")

    def isDefaultUp(self):
        raise NotImplementedError("RibbonBar.isDefaultUp() is not implemented in the ribbon bar.")

    def isNativeMenuBar(self):
        raise NotImplementedError("RibbonBar.isNativeMenuBar() is not implemented in the ribbon bar.")

    def setActiveAction(self, QAction):
        raise NotImplementedError("RibbonBar.setActiveAction() is not implemented in the ribbon bar.")

    def setCornerWidget(self, QWidget, corner=None, *args, **kwargs):
        raise NotImplementedError("RibbonBar.setCornerWidget() is not implemented in the ribbon bar.")

    def setDefaultUp(self, up):
        raise NotImplementedError("RibbonBar.setDefaultUp() is not implemented in the ribbon bar.")

    def setNativeMenuBar(self, bar):
        raise NotImplementedError("RibbonBar.setNativeMenuBar() is not implemented in the ribbon bar.")

    def setRibbonStyle(self, style: RibbonStyle):
        """Set the style of the ribbon.

        :param style: The style to set.
        """
        # The consolidated ribbon.qss is now loaded by the main GUI style system
        # from gui/styles/widgets/ribbon.qss, so we don't need to load separate files
        pass

    def applicationOptionButton(self) -> RibbonApplicationButton:
        """Return the application button."""
        return self._titleWidget.applicationButton()

    def setApplicationIcon(self, icon: QtGui.QIcon):
        """Set the application icon.

        :param icon: The icon to set.
        """
        self._titleWidget.applicationButton().setIcon(icon)

    def addTitleWidget(self, widget: QtWidgets.QWidget):
        """Add a widget to the title widget.

        :param widget: The widget to add.
        """
        self._titleWidget.addTitleWidget(widget)

    def removeTitleWidget(self, widget: QtWidgets.QWidget):
        """Remove a widget from the title widget.

        :param widget: The widget to remove.
        """
        self._titleWidget.removeTitleWidget(widget)

    def insertTitleWidget(self, index: int, widget: QtWidgets.QWidget):
        """Insert a widget to the title widget.

        :param index: The index to insert the widget.
        :param widget: The widget to insert.
        """
        self._titleWidget.insertTitleWidget(index, widget)

    def addFileMenu(self) -> RibbonMenu:
        """Add a file menu to the ribbon."""
        return self.applicationOptionButton().addFileMenu()

    def ribbonHeight(self) -> int:
        """Get the total height of the ribbon.

        :return: The height of the ribbon.
        """
        return self._ribbonHeight

    def setRibbonHeight(self, height: int):
        """Set the total height of the ribbon.

        :param height: The height to set.
        """
        self._ribbonHeight = height
        self.setFixedHeight(height)

    def tabBar(self) -> RibbonTabBar:
        """Return the tab bar of the ribbon.

        :return: The tab bar of the ribbon.
        """
        return self._titleWidget.tabBar()

    def quickAccessToolBar(self) -> QtWidgets.QToolBar:
        """Return the quick access toolbar of the ribbon.

        :return: The quick access toolbar of the ribbon.
        """
        return self._titleWidget.quickAccessToolBar()

    def addQuickAccessButton(self, button: QtWidgets.QToolButton):
        """Add a button to the quick access bar.

        :param button: The button to add.
        """
        button.setAutoRaise(True)
        self._titleWidget.quickAccessToolBar().addWidget(button)

    def setQuickAccessButtonHeight(self, height: int):
        """Set the height of the quick access buttons.

        :param height: The height to set.
        """
        self._titleWidget.setQuickAccessButtonHeight(height)

    def title(self) -> str:
        """Return the title of the ribbon.

        :return: The title of the ribbon.
        """
        return self._titleWidget.title()

    def setTitle(self, title: str):
        """Set the title of the ribbon.

        :param title: The title to set.
        """
        self._titleWidget.setTitle(title)

    def setTitleWidgetHeight(self, height: int):
        """Set the height of the title widget.

        :param height: The height to set.
        """
        self._titleWidget.setTitleWidgetHeight(height)

    def rightToolBar(self) -> QtWidgets.QToolBar:
        """Return the right toolbar of the ribbon.

        :return: The right toolbar of the ribbon.
        """
        return self._titleWidget.rightToolBar()

    def addRightToolButton(self, button: QtWidgets.QToolButton):
        """Add a widget to the right button bar.

        :param button: The button to add.
        """
        button.setAutoRaise(True)
        self._titleWidget.addRightToolButton(button)

    def setRightToolBarHeight(self, height: int):
        """Set the height of the right buttons.

        :param height: The height to set.
        """
        self._titleWidget.setRightToolBarHeight(height)

    def helpRibbonButton(self) -> QtWidgets.QToolButton:
        """Return the help button of the ribbon.

        :return: The help button of the ribbon.
        """
        return self._titleWidget.helpRibbonButton()

    def setHelpButtonIcon(self, icon: QtGui.QIcon):
        """Set the icon of the help button.

        :param icon: The icon to set.
        """
        self._titleWidget.setHelpButtonIcon(icon)

    def removeHelpButton(self):
        """Remove the help button from the ribbon."""
        self._titleWidget.removeHelpButton()

    def collapseRibbonButton(self) -> QtWidgets.QToolButton:
        """Return the collapse ribbon button.

        :return: The collapse ribbon button.
        """
        return self._titleWidget.collapseRibbonButton()

    def setCollapseButtonIcon(self, icon: QtGui.QIcon):
        """Set the icon of the min button.

        :param icon: The icon to set.
        """
        self._titleWidget.setCollapseButtonIcon(icon)

    def removeCollapseButton(self):
        """Remove the min button from the ribbon."""
        self._titleWidget.removeCollapseButton()

    def category(self, name: str) -> RibbonCategory:
        """Return the category with the given name.

        :param name: The name of the category.
        :return: The category with the given name.
        """
        return self._categories[name]

    def categories(self) -> typing.Dict[str, RibbonCategory]:
        """Return a list of categories of the ribbon.

        :return: A dict of categories of the ribbon.
        """
        return self._categories

    def addCategoriesBy(
        self,
        data: typing.Dict[
            str,  # title of the category
            typing.Dict,  # data of the category
        ],
    ) -> typing.Dict[str, RibbonCategory]:
        """Add categories from a dict.

        :param data: The dict of categories. The dict is of the form:

            .. code-block:: python

                {
                    "category-title": {
                        "style": RibbonCategoryStyle.Normal,
                        "color": QtCore.Qt.red,
                        "panels": {
                            "panel-title": {
                                "showPanelOptionButton": True,
                                "widgets": {
                                    "widget-name": {
                                        "type": "Button",
                                        "args": (),
                                        "kwargs": {  # or "arguments" for backward compatibility
                                            "key1": "value1",
                                            "key2": "value2"
                                        }
                                    },
                                }
                            },
                        },
                    }
                }
        :return: A dict of categories of the ribbon.
        """
        categories = {}
        for title, category_data in data.items():
            style = category_data.get("style", RibbonCategoryStyle.Normal)
            color = category_data.get("color", None)
            categories[title] = self.addCategory(title, style, color)
            categories[title].addPanelsBy(category_data.get("panels", {}))
        return categories

    def addCategory(
        self,
        title: str,
        style=RibbonCategoryStyle.Normal,
        color: QtGui.QColor = None,
    ) -> typing.Union[RibbonNormalCategory, RibbonContextCategory]:
        """Add a new category to the ribbon.

        :param title: The title of the category.
        :param style: The button style of the category.
        :param color: The color of the context category, only used if style is Context, if None, the default color
                      will be used.
        :return: The newly created category.
        """
        if title in self._categories:
            raise ValueError(f"Category with title {title} already exists.")
        if style == RibbonCategoryStyle.Context:
            if color is None:
                color = contextColors[self._contextCategoryCount % len(contextColors)]
                self._contextCategoryCount += 1
        category = (
            RibbonContextCategory(title, color, self)  # noqa
            if style == RibbonCategoryStyle.Context
            else RibbonNormalCategory(title, self)  # noqa
        )
        category.setMaximumRows(self._maxRows)
        category.setFixedHeight(
            self._ribbonHeight
            - self._mainLayout.spacing() * 2
            - self._mainLayout.contentsMargins().top()
            - self._mainLayout.contentsMargins().bottom()
            - self._titleWidget.height()
        )  # 4: extra space for drawing lines when debugging
        self._categories[title] = category
        self._stackedWidget.addWidget(category)
        if style == RibbonCategoryStyle.Normal:
            self._titleWidget.tabBar().addTab(title, color)
        elif style == RibbonCategoryStyle.Context:
            category.hide()
        if len(self._categories) == 1:
            self._titleWidget.tabBar().setCurrentIndex(1)
            self.showCategoryByIndex(1)
        return category

    def addNormalCategory(self, title: str) -> RibbonNormalCategory:
        """Add a new category to the ribbon.

        :param title: The title of the category.
        :return: The newly created category.
        """
        return self.addCategory(title, RibbonCategoryStyle.Normal)

    def addContextCategory(
        self,
        title: str,
        color: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor] = QtCore.Qt.GlobalColor.blue,
    ) -> RibbonContextCategory:
        """Add a new context category to the ribbon.

        :param title: The title of the category.
        :param color: The color of the context category, if None, the default color will be used.
        :return: The newly created category.
        """
        return self.addCategory(title, RibbonCategoryStyle.Context, color)

    def addContextCategories(
        self,
        name: str,
        titles: typing.List[str],
        color: typing.Union[QtGui.QColor, QtCore.Qt.GlobalColor] = QtCore.Qt.GlobalColor.blue,
    ) -> RibbonContextCategories:
        """Add a group of context categories with the same tab color to the ribbon.

        :param name: The name of the context categories.
        :param titles: The title of the category.
        :param color: The color of the context category, if None, the default color will be used.
        :return: The newly created category.
        """
        if color is None:
            color = contextColors[self._contextCategoryCount % len(contextColors)]
            self._contextCategoryCount += 1
        categories = RibbonContextCategories(
            name,
            color,
            {title: self.addContextCategory(title, color) for title in titles},
            self,
        )
        return categories

    def showCategoryByIndex(self, index: int):
        """Show category by tab index

        :param index: tab index
        """
        self._currentTabIndex = index
        title = self._titleWidget.tabBar().tabText(index)  # 0 is the file tab
        if title in self._categories:
            self._stackedWidget.setCurrentWidget(self._categories[title])

    def showContextCategory(self, category: typing.Union[RibbonContextCategory, RibbonContextCategories]):
        """Show the given category or categories, if it is not a context category, nothing happens.

        :param category: The category to show.
        """
        if isinstance(category, RibbonContextCategory):
            self._titleWidget.tabBar().addTab(category.title(), category.color())
            self._titleWidget.tabBar().setCurrentIndex(self._titleWidget.tabBar().count() - 1)
            self._stackedWidget.setCurrentWidget(category)
        elif isinstance(category, RibbonContextCategories):
            categories = category
            titles = list(categories.keys())
            self._titleWidget.tabBar().addAssociatedTabs(categories.name(), titles, categories.color())
            self._titleWidget.tabBar().setCurrentIndex(self._titleWidget.tabBar().count() - len(titles))
            self._stackedWidget.setCurrentWidget(categories[titles[0]])

    def hideContextCategory(self, category: typing.Union[RibbonContextCategory, RibbonContextCategories]):
        """Hide the given category or categories, if it is not a context category, nothing happens.

        :param category: The category to hide.
        """
        if isinstance(category, RibbonContextCategory):
            self.tabBar().removeTab(self.tabBar().indexOf(category.title()))
        elif isinstance(category, RibbonContextCategories):
            categories = category
            for c in categories:
                self.tabBar().removeTab(self.tabBar().indexOf(c.title()))

    def categoryVisible(self, category: RibbonCategory) -> bool:
        """Return whether the category is shown.

        :param category: The category to check.

        :return: Whether the category is shown.
        """
        return category.title() in self._titleWidget.tabBar().tabTitles()

    def removeCategory(self, category: RibbonCategory):
        """Remove a category from the ribbon.

        :param category: The category to remove.
        """
        self.tabBar().removeTab(self._titleWidget.tabBar().indexOf(category.title()))
        self._stackedWidget.removeWidget(category)

    def removeCategories(self, categories: RibbonContextCategories):
        """Remove a list of categories from the ribbon.

        :param categories: The categories to remove.
        """
        for category in categories.values():
            self.removeCategory(category)

    def setCurrentCategory(self, category: RibbonCategory):
        """Set the current category.

        :param category: The category to set.
        """
        self._stackedWidget.setCurrentWidget(category)
        if category.title() in self._titleWidget.tabBar().tabTitles():
            self._titleWidget.tabBar().setCurrentIndex(self._titleWidget.tabBar().indexOf(category.title()))
        else:
            raise ValueError(
                f"Category {category.title()} is not in the ribbon, "
                f"please show the context category/categories first."
            )

    def currentCategory(self) -> RibbonCategory:
        """Return the current category.

        :return: The current category.
        """
        return self._categories[self._titleWidget.tabBar().tabText(self._titleWidget.tabBar().currentIndex())]

    def minimumSizeHint(self) -> QtCore.QSize:
        """Return the minimum size hint of the widget.

        :return: The minimum size hint.
        """
        return QtCore.QSize(super().minimumSizeHint().width(), self._ribbonHeight)

    def _onTabBarDoubleClicked(self):
        """Handle double-click on tab bar to toggle ribbon visibility."""
        if self._ribbonVisible:
            self.hideRibbon()
        else:
            self.showRibbon()
            
    def _collapseButtonClicked(self):
        self.tabBar().currentChanged.connect(self.showRibbon)  # type: ignore
        self.hideRibbon() if self._stackedWidget.isVisible() else self.showRibbon()

    def showRibbon(self):
        """Show the ribbon."""
        if not self._ribbonVisible:
            self._ribbonVisible = True
            self.collapseRibbonButton().setToolTip("Collapse Ribbon")
            # Use Unicode icon instead of PNG
            self.collapseRibbonButton().setText("^")
            self._stackedWidget.setVisible(True)
            self.setFixedSize(self.sizeHint())

    def hideRibbon(self):
        """Hide the ribbon."""
        if self._ribbonVisible:
            self._ribbonVisible = False
            self.collapseRibbonButton().setToolTip("Expand Ribbon")
            # Use Unicode icon instead of PNG
            self.collapseRibbonButton().setText("v")
            self._stackedWidget.setVisible(False)
            self.setFixedSize(self.sizeHint().width(), self._titleWidget.size().height() + 5)  # type: ignore

    def ribbonVisible(self) -> bool:
        """Get the visibility of the ribbon.

        :return: True if the ribbon is visible, False otherwise.
        """
        return self._ribbonVisible

    def setRibbonVisible(self, visible: bool):
        """Set the visibility of the ribbon.

        :param visible: True to show the ribbon, False to hide it.
        """
        self.showRibbon() if visible else self.hideRibbon()
