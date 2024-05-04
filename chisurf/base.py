from __future__ import annotations
from chisurf import typing

import os
import uuid
import json
import os.path
import zlib
import copy
import yaml
import numpy as np

from slugify import slugify
from collections.abc import Iterable

import chisurf
import scikit_fluorescence.io.zipped


def to_elementary(
    obj: typing.Dict,
    verbose: bool = False,
    remove_protected: bool = True
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

    Returns
    -------
    dict
        Dictionary that only contains objects of the type stings,
        floats, int, or boolean.
    """
    if verbose:
        print(type(obj))
    if isinstance(obj, dict):
        if verbose:
            print("Converting elements of dict.")
        re = dict()
        for k in obj:
            if (k[0] == "_") and remove_protected:
                continue
            if verbose:
                print("Converting key:", k)
            re[k] = to_elementary(
                obj=obj[k],
                verbose=verbose,
                remove_protected=remove_protected
            )
        return re
    # Check numpy types first, as np.float also is a python float instance
    elif isinstance(obj, np.floating):
        if verbose:
            print("Converting numpy float to python.")
        return float(obj)
    elif isinstance(obj, (str, float, int, bool)) or obj is None:
        return obj
    elif isinstance(obj, np.ndarray):
        if verbose:
            print("Converting np.ndarray to list.")
        return obj.tolist()
    elif isinstance(obj, Iterable):
        if verbose:
            print("Converting Iterable list.")
        return [
            to_elementary(
                obj=e,
                verbose=verbose,
                remove_protected=remove_protected
            ) for e in obj
        ]
    elif isinstance(obj, np.integer):
        if verbose:
            print("Converting numpy integer to python.")
        return int(obj)
    elif isinstance(obj, chisurf.base.Base):
        if verbose:
            print("Converting chisurf.base.Base.")
        return to_elementary(
            obj.to_dict(
                convert_values_to_elementary=True,
                copy_values=True,
                remove_protected=remove_protected
            )
        )
    else:
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
    kkl_ss

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
        if isinstance(value, searched_object_type):
            re.append(value)
        elif isinstance(value, list):
            re += find_objects(value, searched_object_type)
    if remove_doublets:
        return list(set(re))
    else:
        return re


class Base(object):

    _verbose = chisurf.verbose
    supported_save_file_types: typing.List[str] = ["yaml", "json"]
    meta_data: typing.Dict = dict()

    @property
    def unique_identifier(self):
        return self.meta_data['unique_identifier']

    @unique_identifier.setter
    def unique_identifier(self, v):
        self.meta_data['unique_identifier'] = v

    @property
    def name(self) -> str:
        # try:
        name = self.__dict__.get('name', self.__class__.name)
        name = name() if callable(name) else name
        return name
        # except (KeyError, AttributeError):
        #     return self.__class__.__name__

    @name.setter
    def name(self, v: str):
        self.__dict__['name'] = v

    @property
    def verbose(self):
        return self.meta_data['verbose']

    @verbose.setter
    def verbose(self, v: bool):
        self.meta_data['verbose'] = v

    def save(
            self,
            filename: str,
            file_type: str = 'yaml',
            verbose: bool = False
    ) -> None:
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
            # check for filename extension
            root, ext = os.path.splitext(filename)
            filename = root + "." + file_type
            if file_type == "yaml":
                txt = self.to_yaml()
            elif file_type == "json":
                txt = self.to_json()
            if verbose:
                print(txt)
            with scikit_fluorescence.io.zipped.open_maybe_zipped(
                    filename=filename,
                    mode='w'
            ) as fp:
                fp.write(txt)

    def load(
            self,
            filename: str,
            file_type: str = 'yaml',
            verbose: bool = False,
            **kwargs
    ) -> None:
        if file_type == "json":
            self.from_json(
                filename=filename,
                verbose=verbose
            )
        else:
            self.from_yaml(
                filename=filename,
                verbose=verbose
            )

    def to_dict(
            self,
            remove_protected: bool = False,
            copy_values: bool = True,
            convert_values_to_elementary: bool = False
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

        Returns
        -------
        dict
            A dictionary containing all class attributes. This corresponds to
            the attribute *__dict__*.

        """
        if to_elementary:
            copy_values = True
        if remove_protected:
            d = dict()
            for key in self.__dict__:
                if key[0] != '_':
                    try:
                        if copy_values:
                            d[key] = copy.copy(self.__dict__[key])
                        else:
                            d[key] = self.__dict__[key]
                    except TypeError:
                        chisurf.logging.warning(f"Skipping element {key}")
        else:
            if copy_values:
                d = copy.copy(self.__dict__)
                d["meta_data"] = copy.deepcopy(self.meta_data)
                return d
            else:
                d = self.__dict__
        if convert_values_to_elementary:
            return to_elementary(
                obj=d
            )
        else:
            return d

    def from_dict(
            self,
            v: dict
    ) -> None:
        self.__dict__.update(v)

    def to_json(
            self,
            indent: int = 4,
            sort_keys: bool = True,
            d: typing.Dict = None,
            remove_protected: bool = False
    ) -> str:
        if d is None:
            d = self.to_dict(
                remove_protected=remove_protected
            )
        return json.dumps(
            obj=to_elementary(
                d
            ),
            indent=indent,
            sort_keys=sort_keys
        )

    def to_yaml(
            self,
            remove_protected: bool = True,
            convert_values_to_elementary: bool = True
    ) -> str:
        return yaml.dump(
            data=to_elementary(
                self.to_dict(
                    remove_protected=remove_protected,
                    convert_values_to_elementary=convert_values_to_elementary
                )
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
                with scikit_fluorescence.io.zipped.open_maybe_zipped(filename, 'r') as fp:
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
        >>> import chisurf.experiments
        >>> dc = chisurf.experiments.data.DataCurve()
        >>> dc.from_json(filename='./test/data/internal_types/datacurve.json')
        """
        j = dict()
        if isinstance(filename, str):
            if os.path.isfile(filename):
                with scikit_fluorescence.io.zipped.open_maybe_zipped(filename, 'r') as fp:
                    j = json.load(fp)
        if isinstance(json_string, str):
            j = json.loads(json_string)
        if verbose:
            print(j)
        self.from_dict(j)

    def __setattr__(
            self,
            key: str,
            value: object
    ):
        propobj = getattr(self.__class__, key, None)
        if isinstance(propobj, property):
            if propobj.fset is None:
                raise AttributeError("can't set attribute")
            propobj.fset(self, value)
        else:
            super().__setattr__(key, value)

    def __getattr__(
            self,
            key: str
    ):
        propobj = getattr(self.__class__, key, None)
        # the key refers to a property
        if isinstance(propobj, property):
            if propobj.fget is None:
                raise AttributeError("can't get attribute")
            return propobj.fget(self)
        return propobj

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.clear()
        self.__dict__.update(state)

    def __str__(self):
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
        these keywords accessible as attributes. Moreover, this class may saves these
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
        ala
        >>> bc.to_dict()
        {'lol': 1, 'parameter': 'ala', 'verbose': False}
        >>> bc.from_dict({'jj': 22, 'zu': "auf"})
        >>> bc.jj
        22
        >>> bc.zu
        auf
        """
        super().__init__()
        if len(args) > 0 and isinstance(args[0], dict):
            kwargs = args[0]

        if meta_data is None:
            meta_data = dict()
        self.meta_data = meta_data

        if unique_identifier is None:
            unique_identifier = str(uuid.uuid4())
        self.meta_data['unique_identifier'] = unique_identifier
        self.meta_data['verbose'] = verbose

        # clean up the keys (no spaces etc)
        d = dict()
        for key in kwargs:
            d[clean_string(key)] = kwargs[key]

        # Assign the the names and set standard values
        if name is None:
            name = self.__class__.__name__

        d['name'] = name
        kwargs.update(d)
        self.__dict__.update(**kwargs)

    def __copy__(self) -> typing.Type[Base]:
        c = self.__class__()
        c.from_dict(
            copy.copy(
                self.to_dict(
                    copy_values=True
                )
            )
        )
        # make sure that the copy gets a new uuid
        c.unique_identifier = str(uuid.uuid4())
        return c

    def __deepcopy__(self, memodict={}):
        c = self.__class__()
        c.from_dict(
            copy.deepcopy(self.to_dict())
        )
        # make sure that the copy gets a new uuid
        c.unique_identifier = str(uuid.uuid4())
        return c


class Data(Base):
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
        return self._embed_data

    @embed_data.setter
    def embed_data(self, v: bool) -> None:
        self._embed_data = v
        if v is False:
            self._data = None

    @property
    def data(self) -> bytes:
        return self._data

    @data.setter
    def data(self, v: Data):
        self._data = v

    @property
    def name(self) -> str:
        try:
            return self.__dict__['name']
        except KeyError:
            return self.filename

    @name.setter
    def name(self, v: str):
        self.__dict__['name'] = v

    @property
    def filename(self) -> str:
        try:
            return self._filename
        except (AttributeError, TypeError):
            return 'No file'

    @filename.setter
    def filename(
            self,
            v: str
    ) -> None:
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
        s = super().__str__()
        s += "\nfilename: %s" % self.filename
        return s

