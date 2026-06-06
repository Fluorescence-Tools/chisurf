class TestCodeEditor:
    def test_creation(self, qapp):
        from chisurf.plugins.misc.code_editor import CodeEditor
        widget = CodeEditor()
        assert widget is not None

    def test_creation_with_filename(self, qapp):
        from chisurf.plugins.misc.code_editor import CodeEditor
        widget = CodeEditor(filename="test.py", language="Python")
        assert widget is not None
