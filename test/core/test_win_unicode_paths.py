import os
import sys
import pytest
import shutil
import tttrlib

@pytest.mark.skipif(sys.platform != 'win32', reason="Windows-specific path tests")
class TestWindowsUnicodePaths:
    
    @pytest.fixture
    def unicode_file(self, tmp_path):
        filename = tmp_path / "test_µ_file.ptu"
        filename.write_text("mock data")
        return str(filename)

    def test_unicode_path_opening(self, unicode_file):
        """Test if tttrlib can open a file with unicode characters in its name."""
        try:
            tt = tttrlib.TTTR(unicode_file)
            assert tt is not None
        except Exception as e:
            pytest.fail(f"Could not open unicode path: {e}")

    def test_win_file_handle_release(self, unicode_file, tmp_path):
        """Test if tttrlib releases the file handle, allowing rename/delete."""
        # Using a real PTU file if possible, or just the mock
        tt = tttrlib.TTTR(unicode_file)
        
        temp_name = str(tmp_path / "test_m_file.ptu")
        
        # On Windows, if tttrlib holds the handle, this rename will fail
        try:
            # We must delete the object to release the handle if it's held by the wrapper
            del tt
            os.rename(unicode_file, temp_name)
            assert os.path.exists(temp_name)
            assert not os.path.exists(unicode_file)
        except Exception as e:
            pytest.fail(f"Rename failed (handle likely held): {e}")

    def test_short_path_name(self, unicode_file):
        """Test if short path names (DOS 8.3) work as a fallback for unicode issues."""
        def get_short_path_name(long_name):
            import ctypes
            from ctypes import wintypes
            _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
            _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            _GetShortPathNameW.restype = wintypes.DWORD
            
            long_name = os.path.abspath(long_name)
            output_buf_size = _GetShortPathNameW(long_name, None, 0)
            if output_buf_size == 0:
                return long_name
            output_buf = ctypes.create_unicode_buffer(output_buf_size)
            needed = _GetShortPathNameW(long_name, output_buf, output_buf_size)
            if output_buf_size >= needed:
                return output_buf.value
            else:
                return long_name

        short_path = get_short_path_name(unicode_file)
        try:
            tt = tttrlib.TTTR(short_path)
            assert tt is not None
        except Exception as e:
            pytest.fail(f"Short path opening failed: {e}")
