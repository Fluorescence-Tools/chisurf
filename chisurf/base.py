from __future__ import annotations
from chisurf import typing

import os
import uuid
import json
import os.path
import zlib
import copy
import yaml
import pickle
import logging
import weakref

import numpy as np
import chisurf

import re
import unicodedata
from collections.abc import Iterable


def slugify(text, separator='_', regex_pattern=r'[^-a-z0-9_]+'):
    """
    Convert a string to a slug.

    Parameters
    ----------
    text : str
        The string to convert
    separator : str
        The separator to use (default is '_')
    regex_pattern : str
        The regex pattern used to identify characters to replace

    Returns
    -------
    str
        A slugified string
    """
    # Convert to lowercase
    text = str(text).lower()

    # Convert accented characters to their ASCII equivalents
    text = unicodedata.normalize('NFKD', text)
    text = ''.join([c for c in text if not unicodedata.combining(c)])

    # Replace characters matching the regex pattern with the separator
    text = re.sub(regex_pattern, separator, text)

    # Replace multiple consecutive separators with a single one
    text = re.sub(f'{separator}+', separator, text)

    # Remove leading/trailing separators
    text = text.strip(separator)

    return text


def to_elementary(
    obj: typing.Dict,
    verbose: bool = False,
    remove_protected: bool = True,
    skip_qt_widgets: bool = False
) -> typing.Dict:
    """Creates a dictionary containing only elements of (basic) elementary types.

    The function recurse into the passed dictionary and creates returns a
    new dictionary where all elements are represented by stings, floats, int,
    and booleans. Numpy arrays ware converted into lists. Iterable types are
    converted into lists containing elementary types.

    Parameters
    ----------
    obj : dict
        The dictionary that is converted
    verbose : bool
        Display additional information during conversion
    remove_protected : bool
        If set to True (the default value is True) protected dictonary
        items, i.e., items with a key that start with an underscore, are not
        copied to the target dictionary.
    skip_qt_widgets : bool
        If set to True (the default value is False) Qt widgets from PyQt5
        will be skipped during conversion to avoid serialization issues.

    Returns
    -------
    dict
        Dictionary that only contains objects of the type stings,
        floats, int, or boolean.
    """
    logging.debug(f"to_elementary: type={type(obj)}")
    if verbose:
        print(type(obj))
    if isinstance(obj, dict):
        logging.debug("to_elementary: Converting elements of dict.")
        if verbose:
            print("Converting elements of dict.")
        re = dict()
        for k in obj:
            if (k[0] == "_") and remove_protected:
                logging.debug(f"to_elementary: Skipping protected key: {k}")
                continue
            logging.debug(f"to_elementary: Converting key: {k}")
            if verbose:
                print("Converting key:", k)
            re[k] = to_elementary(
                obj=obj[k],
                verbose=verbose,
                remove_protected=remove_protected,
                skip_qt_widgets=skip_qt_widgets
            )
        return re
    # Check numpy types first, as np.float also is a python float instance
    elif isinstance(obj, np.floating):
        logging.debug("to_elementary: Converting numpy float to python.")
        if verbose:
            print("Converting numpy float to python.")
        return float(obj)
    elif isinstance(obj, (str, float, int, bool)) or obj is None:
        logging.debug(f"to_elementary: Passing through elementary type: {type(obj)}")
        return obj
    elif isinstance(obj, np.ndarray):
        logging.debug(f"to_elementary: Converting np.ndarray to list. Shape: {obj.shape}")
        if verbose:
            print("Converting np.ndarray to list.")
        return obj.tolist()
    elif isinstance(obj, Iterable):
        logging.debug("to_elementary: Converting Iterable to list.")
        if verbose:
            print("Converting Iterable list.")
        return [
            to_elementary(
                obj=e,
                verbose=verbose,
                remove_protected=remove_protected,
                skip_qt_widgets=skip_qt_widgets
            ) for e in obj
        ]
    elif isinstance(obj, np.integer):
        logging.debug("to_elementary: Converting numpy integer to python.")
        if verbose:
            print("Converting numpy integer to python.")
        return int(obj)
    # Check if it's a Qt widget and skip if requested
    elif skip_qt_widgets and hasattr(obj, '__module__') and obj.__module__.startswith('PyQt5'):
        logging.warning(f"Skipping element {obj.__class__.__name__}")
        return None
    elif isinstance(obj, chisurf.base.Base):
        logging.debug(f"to_elementary: Converting chisurf.base.Base of type {obj.__class__.__name__}.")
        if verbose:
            print("Converting chisurf.base.Base.")
        return to_elementary(
            obj.to_dict(
                convert_values_to_elementary=True,
                copy_values=True,
                remove_protected=remove_protected,
                skip_qt_widgets=skip_qt_widgets
            ),
            verbose=verbose,
            remove_protected=remove_protected,
            skip_qt_widgets=skip_qt_widgets
        )
    else:
        logging.warning(f"to_elementary: Object of type {type(obj)} was not converted to basic type")
        print("WARNING object was not converted to basic type")
        return str(obj)


def clean_string(
        s: str,
        regex_pattern: str = r'[^-a-z0-9_]+'
) -> str:
    """Get a slugified a string.

    Special characters to clean up string. The slugified string can be used
    as a Python variable name.

    Parameters
    ----------
    s : str
        The string that is slugified
    regex_pattern : str
        The regex pattern that is used to

    Returns
    -------
    str
        A slugified string

    Examples
    --------

    >>> import chisurf.base
    >>> chisurf.base.clean_string("kkl ss ##")
    'kkl_ss'

    """
    r = slugify(s, separator='_', regex_pattern=regex_pattern)
    return r


def find_objects(
        search_iterable: Iterable,
        searched_object_type: typing.Type,
        remove_doublets: bool = True
) -> typing.List[object]:
    """Traverse a list recursively to return all objects of type
    `searched_object_type` as a list

    :param search_iterable: list
    :param searched_object_type: an object type
    :param remove_doublets: boolean
    :return: list of objects with certain object type
    """
    re = list()
    for value in search_iterable:
        # Standard isinstance check for speed and correctness
        if isinstance(value, searched_object_type):
            re.append(value)
        # Fallback to class name check to handle reloaded environments (e.g. pytest)
        elif type(value).__name__ == getattr(searched_object_type, '__name__', None):
            re.append(value)
        elif isinstance(value, list):
            re += find_objects(value, searched_object_type, remove_doublets)
    if remove_doublets:
        seen = set()
        return [x for x in re if not (x in seen or seen.add(x))]
    else:
        return re


class Base(object):

    _verbose = chisurf.settings.cs_settings['verbose']
    supported_save_file_types: typing.List[str] = ["yaml", "json", "pkl"]
    meta_data: typing.Dict = dict()
    # Global index of all live Base instances keyed by unique_identifier.
    # Entries are weak references; dead instances are evicted automatically.
    _uuid_index: "weakref.WeakValueDictionary[str, Base]" = weakref.WeakValueDictionary()

    @property
    def unique_identifier(self):
        """Return the UUID that uniquely identifies this instance."""
        return self.meta_data['unique_identifier']

    @unique_identifier.setter
    def unique_identifier(self, v):
        """Set the UUID that uniquely identifies this instance."""
        self.meta_data['unique_identifier'] = v

    def __eq__(self, other: object) -> bool:
        """Compare two Base instances by their unique identifier."""
        if not isinstance(other, Base):
            return NotImplemented
        return self.unique_identifier == other.unique_identifier

    def __ne__(self, other: object) -> bool:
        """Inequality comparison, inverse of :meth:`__eq__`."""
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result

    def __hash__(self):
        """Hash based on the unique identifier."""
        return hash(self.unique_identifier)

    @classmethod
    def find_by_uuid(cls, uid: str) -> typing.Optional["Base"]:
        """Look up a live :class:`Base` instance by its ``unique_identifier``.

        Returns ``None`` if no live instance with that UID exists.

        If multiple live instances share the same UID (e.g. a copy was made
        with :func:`copy.copy`), the most recently created one is returned.
        """
        if not uid:
            return None
        return cls._uuid_index.get(str(uid))

    @classmethod
    def all_uuids(cls) -> typing.List[str]:
        """Return the UIDs of all currently live :class:`Base` instances."""
        return list(cls._uuid_index.keys())

    @property
    def name(self) -> str:
        """Return the name of this object (falls back to class name)."""
        # try:
        name = self.__dict__.get('name', self.__class__.name)
        name = name() if callable(name) else name
        return name
        # except (KeyError, AttributeError):
        #     return self.__class__.__name__

    @name.setter
    def name(self, v: str):
        """Set the name of this object."""
        self.__dict__['name'] = v

    @property
    def verbose(self):
        """Return the verbosity flag."""
        return self.meta_data['verbose']

    @verbose.setter
    def verbose(self, v: bool):
        """Set the verbosity flag."""
        self.meta_data['verbose'] = v

    def save(
            self,
            filename: str,
            file_type: str = 'yaml',
            verbose: bool = False,
            skip_qt_widgets: bool = False
    ) -> None:
        """Serialize and save the object to a file.

        Parameters
        ----------
        filename : str
            Path to the output file.
        file_type : str
            Output format (``'yaml'``, ``'json'``, or ``'pkl'``).
        verbose : bool
            If True, print the serialized content.
        skip_qt_widgets : bool
            If True, skip Qt widgets during serialization.
        """
        chisurf.logging.info(
            "%s of type %s is saving filename %s as file type %s" % (
                self.name,
                self.__class__.__name__,
                filename,
                file_type
            )
        )
        if file_type in self.supported_save_file_types:
            txt = ""
            mode = "w"
            # check for filename extension
            root, ext = os.path.splitext(filename)
            filename = root + "." + file_type
            if file_type == "yaml":
                txt = self.to_yaml(skip_qt_widgets=skip_qt_widgets)
            elif file_type == "json":
                txt = self.to_json(skip_qt_widgets=skip_qt_widgets)
            elif file_type == "pkl":
                txt = pickle.dumps(self)
                mode = 'wb'
            if verbose:
                print(txt)
            with io.open_maybe_zipped(filename, mode) as fp:
                fp.write(txt)

    def load(
            self,
            filename: str,
            file_type: str = 'yaml',
            verbose: bool = False,
            **kwargs
    ) -> None:
        """Load and restore the object's state from a file.

        Parameters
        ----------
        filename : str
            Path to the input file.
        file_type : str
            File format (``'json'``, ``'p'``, or ``'yaml'``).
        verbose : bool
            If True, print the loaded content.
        """
        if file_type == "json":
            self.from_json(
                filename=filename,
                verbose=verbose
            )
        elif file_type == "p":
            with open(filename, 'rb') as file:
                data = file.read()
                obj = pickle.loads(data)
                self.__dict__.update(obj.__dict__)
        else:
            self.from_yaml(
                filename=filename,
                verbose=verbose
            )

    def to_dict(
            self,
            remove_protected: bool = False,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False,
            skip_qt_widgets: bool = False
    ) -> dict:
        """

        Parameters
        ----------
        remove_protected : bool
            If this is set to True (default False), protected attributes of the
            class are not part of the returned dictionary
        copy_values : bool
            If this is set to True (default True) the values of *__dict__* are
            copied otherwise the content of *__dict__* is returned as is.
        convert_values_to_elementary: bool
            If this parameter is set to True (default False) the values of
            __dict__ are copied to a new dictionary. The copied values will be
            converted using the function *chisurf.base.to_elementary* to an
            elementary data type, i.e., float, int, bool, str and list of these
            types.
        skip_qt_widgets: bool
            If this parameter is set to True (default False), Qt widgets from PyQt5
            will be skipped during dictionary creation to avoid serialization issues.

        Returns
        -------
        dict
            A dictionary containing all class attributes. This corresponds to
            the attribute *__dict__*.

        """
        if convert_values_to_elementary:
            copy_values = True
        if remove_protected:
            d = dict()
            for key in self.__dict__:
                if key[0] != '_':
                    try:
                        # Skip Qt widgets if requested
                        if skip_qt_widgets:
                            value = self.__dict__[key]
                            # Check if it's a Qt widget
                            if hasattr(value, '__module__') and value.__module__.startswith('PyQt5'):
                                chisurf.logging.warning(f"Skipping element {key}")
                                continue
                        
                        if copy_values:
                            d[key] = copy.copy(self.__dict__[key])
                        else:
                            d[key] = self.__dict__[key]
                    except TypeError:
                        chisurf.logging.warning(f"Skipping element {key}")
        else:
            if copy_values:
                d = copy.copy(self.__dict__)
                # Skip Qt widgets if requested
                if skip_qt_widgets:
                    keys_to_remove = []
                    for key, value in d.items():
                        if hasattr(value, '__module__') and value.__module__.startswith('PyQt5'):
                            keys_to_remove.append(key)
                    
                    for key in keys_to_remove:
                        chisurf.logging.warning(f"Skipping element {key}")
                        d.pop(key, None)
                
                d["meta_data"] = copy.deepcopy(self.meta_data)
                return d
            else:
                d = self.__dict__
        if convert_values_to_elementary:
            return to_elementary(d)
        else:
            return d

    def from_dict(
            self,
            v: dict
    ) -> None:
        """Restore the object's state from a dictionary.

        Parameters
        ----------
        v : dict
            Dictionary containing the attributes to restore.
        """
        self.__dict__.update(v)

    def to_json(
            self,
            indent: int = 4,
            sort_keys: bool = True,
            d: typing.Dict = None,
            remove_protected: bool = False,
            skip_qt_widgets: bool = False
    ) -> str:
        """Serialize the object to a JSON string.

        Parameters
        ----------
        indent : int
            Indentation level for pretty-printing.
        sort_keys : bool
            Whether to sort dictionary keys.
        d : dict, optional
            Pre-built dictionary to serialize (if None, built from self).
        remove_protected : bool
            Whether to omit protected (underscore-prefixed) attributes.
        skip_qt_widgets : bool
            If True, skip Qt widgets during serialization.

        Returns
        -------
        str
            JSON-formatted string.
        """
        if d is None:
            d = self.to_dict(
                remove_protected=remove_protected,
                skip_qt_widgets=skip_qt_widgets
            )
        return json.dumps(
            obj=to_elementary(
                d,
                remove_protected=remove_protected,
                skip_qt_widgets=skip_qt_widgets
            ),
            indent=indent,
            sort_keys=sort_keys
        )

    def to_yaml(
            self,
            remove_protected: bool = True,
            convert_values_to_elementary: bool = True,
            skip_qt_widgets: bool = False
    ) -> str:
        """Serialize the object to a YAML string.

        Parameters
        ----------
        remove_protected : bool
            Whether to omit protected attributes.
        convert_values_to_elementary : bool
            Whether to convert compound types to elementary types.
        skip_qt_widgets : bool
            If True, skip Qt widgets during serialization.

        Returns
        -------
        str
            YAML-formatted string.
        """
        return yaml.dump(
            data=to_elementary(
                self.to_dict(
                    remove_protected=remove_protected,
                    convert_values_to_elementary=convert_values_to_elementary,
                    skip_qt_widgets=skip_qt_widgets
                ),
                remove_protected=remove_protected,
                skip_qt_widgets=skip_qt_widgets
            )
        )

    def from_yaml(
            self,
            yaml_string: str = None,
            filename: str = None,
            verbose: bool = False
    ) -> None:
        """Restore the object's state from a YAML file

        Parameters
        ----------
        yaml_string : str
        filename : str
        verbose : bool

        Returns
        -------
        None

        """
        j = dict()
        if isinstance(filename, str):
            if os.path.isfile(filename):
                with io.open_maybe_zipped(filename, 'r') as fp:
                    j = yaml.safe_load(fp)
        if isinstance(yaml_string, str):
            j = yaml.safe_load(
                yaml_string
            )
        if verbose:
            print(j)
        self.from_dict(j)

    def from_json(
            self,
            json_string: str = None,
            filename: str = None,
            verbose: bool = False
    ) -> None:
        """Restore the object's state from a JSON file

        Parameters
        ----------
        Parameters
        ----------
        json_string : str
            A string containing the JSON file
        filename: str
            The filename to be opened
        verbose: bool
            If True additional output is printed to stdout

        Returns
        -------
        None

        Examples
        --------
        >>> import chisurf.data
        >>> dc = chisurf.data.DataCurve()
        >>> dc.from_json(filename='./test/data/internal_types/datacurve.json')
        """
        j = dict()
        if isinstance(filename, str):
            if os.path.isfile(filename):
                with io.open_maybe_zipped(filename, 'r') as fp:
                    j = json.load(fp)
        if isinstance(json_string, str):
            j = json.loads(json_string)
        if verbose:
            print(j)
        self.from_dict(j)

    def __setattr__(self, key: str, value: object):
        """Route property assignments through their setter; store others in ``__dict__``."""
        propobj = getattr(self.__class__, key, None)
        if isinstance(propobj, property):
            if propobj.fset is None:
                raise AttributeError("can't set attribute")
            propobj.fset(self, value)
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key: str):
        """Fallback attribute lookup that checks for properties on the class."""
        import logging
        propobj = getattr(self.__class__, key, None)
        # the key refers to a property
        if isinstance(propobj, property):
            if propobj.fget is None:
                logging.debug(f"Property '{key}' has no getter")
                raise AttributeError("can't get attribute")
            return propobj.fget(self)
        if propobj is None:
            logging.debug(f"Attribute '{key}' not found in {self.__class__.__name__}")
            raise AttributeError(f"{self.__class__.__name__} object has no attribute '{key}'")
        return propobj

    def __getstate__(self):
        """Return a minimal dict for pickling (metadata + name)."""
        d = {
            'meta_data': self.meta_data,
            'name': self.name
        }
        return d

    def __setstate__(self, state):
        """Restore pickled state from :meth:`__getstate__`."""
        self.__dict__.update(state)
        try:
            Base._uuid_index[str(self.unique_identifier)] = self
        except Exception:
            pass

    def __str__(self):
        """Return a one-line summary showing the class name."""
        s = 'Class: %s\n' % self.__class__.__name__
        return s

    def __init__(
            self,
            name: object = None,
            verbose: bool = False,
            unique_identifier: str = None,
            meta_data: typing.Dict = None,
            *args,
            **kwargs
    ):
        """The class saves all passed keyword arguments in dictionary and makes
        these keywords accessible as attributes. Moreover, this class saves these
        keywords in a JSON or YAML file. These files can be also loaded.

        :param name:
        :param args:
        :param kwargs:

        Example
        -------

        >>> import chisurf.base
        >>> bc = chisurf.base.Base(parameter="ala", lol=1)
        >>> bc.lol
        1
        >>> bc.parameter
        'ala'
        >>> sorted(bc.to_dict().keys())
        ['lol', 'meta_data', 'name', 'parameter']
        >>> bc.from_dict({'jj': 22, 'zu': "auf"})
        >>> bc.jj
        22
        >>> bc.zu
        'auf'
        """
        super().__init__()
        if len(args) > 0 and isinstance(args[0], dict):
            kwargs = args[0]

        if meta_data is None:
            meta_data = dict()
        self.meta_data = meta_data

        self.verbose = verbose
        if len(args) > 0 and isinstance(args[0], dict):
            kwargs = args[0]

        if unique_identifier is None:
            unique_identifier = str(uuid.uuid4())
        self.meta_data['unique_identifier'] = unique_identifier
        self.meta_data['verbose'] = verbose

        # clean up the keys (no spaces etc.)
        d = dict()
        for key in kwargs:
            d[clean_string(key)] = kwargs[key]

        # Assign names and set standard values
        if name is None:
            name = self.__class__.__name__

        d['name'] = name
        kwargs.update(d)
        self.__dict__.update(**kwargs)
        Base._uuid_index[str(self.unique_identifier)] = self

    def __copy__(self) -> typing.Type[Base]:
        """Return a shallow copy with a deep copy of metadata."""
        c = self.__class__.__new__(self.__class__)
        c.__dict__ = copy.copy(self.__dict__)
        c.__dict__['meta_data'] = copy.deepcopy(self.__dict__.get('meta_data', {}))
        Base._uuid_index[str(c.unique_identifier)] = c
        return c

    def __deepcopy__(self, memodict=None):
        """Return a deep copy of this instance."""
        if memodict is None:
            memodict = {}
        c = self.__class__.__new__(self.__class__)
        c.__dict__ = copy.deepcopy(self.__dict__, memodict)
        Base._uuid_index[str(c.unique_identifier)] = c
        return c


def find_by_uuid(uid: str) -> typing.Optional["Base"]:
    """Convenience: look up a live :class:`Base` instance by its UID.

    Equivalent to ``Base.find_by_uuid(uid)``.
    """
    return Base.find_by_uuid(uid)


def all_uuids() -> typing.List[str]:
    """Convenience: return UIDs of all live :class:`Base` instances.

    Equivalent to ``Base.all_uuids()``.
    """
    return Base.all_uuids()


class Data(Base):
    """Base class for data objects with file-binding and optional data embedding.

    Extends :class:`Base` with file-path tracking, binary data embedding,
    and configurable read-size limits.
    """

    def __init__(
            self,
            filename: str = "None",
            data: bytes = None,
            embed_data: bool = None,
            read_file_size_limit: int = None,
            name: object = None,
            verbose: bool = False,
            unique_identifier: str = None,
            meta_data: typing.Dict = None,
            **kwargs
    ):
        super().__init__(
            name=name,
            verbose=verbose,
            unique_identifier=unique_identifier,
            meta_data=meta_data,
            **kwargs
        )
        self._data = data
        self._filename = None

        if embed_data is None:
            embed_data = chisurf.settings.database['embed_data']
        if read_file_size_limit is None:
            read_file_size_limit = chisurf.settings.database['read_file_size_limit']

        self._embed_data = embed_data
        self._max_file_size = read_file_size_limit

        self.filename = filename

    @property
    def embed_data(self) -> bool:
        """Whether binary data is embedded in the serialized output."""
        return self._embed_data

    @embed_data.setter
    def embed_data(self, v: bool) -> None:
        """Control whether binary data is embedded in serialized output."""
        self._embed_data = v
        if v is False:
            self._data = None

    @property
    def data(self) -> bytes:
        """Return the embedded binary data (or None)."""
        return self._data

    @data.setter
    def data(self, v: Data):
        """Set the embedded binary data."""
        self._data = v

    @property
    def name(self) -> str:
        """Return the object name, falling back to the filename."""
        try:
            return self.__dict__['name']
        except KeyError:
            return self.filename

    @name.setter
    def name(self, v: str):
        """Set the object name."""
        self.__dict__['name'] = v

    @property
    def filename(self) -> str:
        """Return the associated file path, or ``'No file'``."""
        try:
            return self._filename
        except (AttributeError, TypeError):
            return 'No file'

    @filename.setter
    def filename(
            self,
            v: str
    ) -> None:
        """Set the file path and optionally embed its binary content.

        The file content is read and optionally compressed/embedded
        according to the instance's ``embed_data`` and ``_max_file_size``
        settings.
        """
        try:
            self._filename = os.path.normpath(v)
            file_size = os.path.getsize(self._filename)
            self._data = b""
            if file_size < self._max_file_size and self._embed_data:
                with open(self._filename, "rb") as fp:
                    data = fp.read()
                    if len(data) > chisurf.settings.database['compression_data_limit']:
                        data = zlib.compress(data)
                    if len(data) < chisurf.settings.database['embed_data_limit']:
                        self._data = data
            if self.verbose:
                print("Filename: %s" % self._filename)
                print("File size [byte]: %s" % file_size)
        except FileNotFoundError:
            if self.verbose:
                chisurf.logging.warning("Filename: %s not found" % v)

    def __str__(self):
        """Return a summary including class name and filename."""
        s = super().__str__()
        s += "\nfilename: %s" % self.filename
        return s


def safe_import(module_name: str, package_name: str = None, parent=None):
    """
    Safely import a module with user-friendly error handling for conda packages.
    
    Args:
        module_name: The module to import (e.g., 'numpy', 'scipy.optimize')
        package_name: The conda package name if different from module_name (e.g., 'scipy' for 'scipy.optimize')
        parent: Parent widget for error dialogs
    
    Returns:
        The imported module, or None if import failed
        
    Example:
        >>> np = safe_import('numpy')
        >>> optimize = safe_import('scipy.optimize', 'scipy')
    """
    if package_name is None:
        package_name = module_name.split('.')[0]
    
    try:
        return __import__(module_name)
    except ImportError:
        # Try to show Package Manager redirect dialog
        try:
            from chisurf.gui import QtWidgets
            if QtWidgets is not None:
                title = f"Package {package_name} not available"
                text = (
                    f"The package '{package_name}' is required but not installed.\n\n"
                    f"Please use ChiSurf's Package Manager (available in Help > Updates and Packages > Package Manager) "
                    f"to install the '{package_name}' package."
                )
                
                msg = QtWidgets.QMessageBox(parent)
                msg.setIcon(QtWidgets.QMessageBox.Information)
                msg.setWindowTitle(title)
                msg.setText(text)
                msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
                msg.exec_()
        except Exception:
            # Fallback to logging if Qt not available
            chisurf.logging.warning(f"Package '{package_name}' not available. Please use ChiSurf's Package Manager to install it.")
        return None


import chisurf.fio as io
