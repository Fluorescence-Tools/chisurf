from enum import IntEnum

from qtpy import QtGui


class RibbonCategoryStyle(IntEnum):
    """The button style of a category."""

    Normal = 0
    Context = 1


Normal = RibbonCategoryStyle.Normal
Context = RibbonCategoryStyle.Context


#: A list of context category colors - ChiSurf color scheme
contextColors = [
    QtGui.QColor(42, 130, 218),  # Blue
    QtGui.QColor(46, 204, 113),  # Green
    QtGui.QColor(231, 76, 60),   # Red
    QtGui.QColor(241, 196, 15),  # Yellow
    QtGui.QColor(155, 89, 182),  # Purple
    QtGui.QColor(52, 152, 219),  # Light blue
]


class RibbonSpaceFindMode(IntEnum):
    """Mode to find available space in a grid layout, ColumnWise or RowWise."""

    ColumnWise = 0
    RowWise = 1


ColumnWise = RibbonSpaceFindMode.ColumnWise
RowWise = RibbonSpaceFindMode.RowWise


class RibbonStyle(IntEnum):
    Default = 0
    Debug = 1


Debug = RibbonStyle.Debug
Default = RibbonStyle.Default


class RibbonButtonStyle(IntEnum):
    """Button style, Small, Medium, or Large."""

    Small = 0
    Medium = 1
    Large = 2


Small = RibbonButtonStyle.Small
Medium = RibbonButtonStyle.Medium
Large = RibbonButtonStyle.Large
