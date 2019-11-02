from __future__ import annotations
from typing import Type, List

import os
import uuid
import json
import os.path
import zlib
import copy
from collections.abc import Iterable

import yaml
from slugify import slugify

import chisurf
import chisurf.fio


class Base(object):
    """

    """

    @property
    def name(self) -> str:
        # try:
        #     name = self.__dict__['name']
        #     return name() if callable(name) else name
        # except KeyError or AttributeError:
        #     return self.__class__.__name__
        name = self.__dict__['name']
        return name() if callable(name) else name

    @name.setter
    def name(
            self,
            v: str
    ):
        self.__dict__['name'] = v

    def save(
            self,
            filename: str,
            file_type: str = 'yaml',
            verbose: bool = False
    ) -> None:
        if file_type == "yaml":
            txt = self.to_yaml()
        else:
            txt = self.to_json()
        if verbose:
            print(txt)
        with chisurf.fio.zipped.open_maybe_zipped(
                filename=filename,
                mode='w'
        ) as fp:
            fp.write(txt)

    def load(
            self,
            filename: str,
            file_type: str = 'json',
            verbose: bool = False
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

    def to_dict(self) -> dict:
        return self.__dict__

    def from_dict(
            self,
            v: dict
    ) -> None:
        self.__dict__.clear()
        self.__dict__.update(v)

    def to_json(
            self,
            indent: int = 4,
            sort_keys: bool = True
    ) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=sort_keys
        )

    def to_yaml(self) -> str:
        d = self.to_dict()
        return yaml.dump(d)

    def from_yaml(
            self,
            yaml_string: str = None,
            filename: str = None,
            verbose: bool = False
    ) -> None:
        """

        :param yaml_string:
        :param filename:
        :param verbose:
        :return:
        """
        j = dict()
        if isinstance(filename, str):
            if os.path.isfile(filename):
                with chisurf.fio.zipped.open_maybe_zipped(filename, 'r') as fp:
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
        """Reads the content of a JSON file into the object.

        Example
        -------

        >>> import chisurf.experiments
        >>> dc = chisurf.experiments.data.DataCurve()
        >>> dc.from_json(filename='./test/data/internal_types/datacurve.json')


        Parameters
        ----------
        json_string : str
            A string containing the JSON file
        filename: str
            The filename to be opened
        verbose: bool
            If True additional output is printed to stdout

        """
        j = dict()
        if isinstance(filename, str):
            if os.path.isfile(filename):
                with chisurf.fio.zipped.open_maybe_zipped(filename, 'r') as fp:
                    j = json.load(fp)
        if isinstance(json_string, str):
            j = json.loads(json_string)
        if verbose:
            print(j)
        self.from_dict(j)

    def __setattr__(
            self,
            k: str,
            v: object
    ):
        self.__dict__[k] = v
        # Set the attributes normally
        propobj = getattr(self.__class__, k, None)
        if isinstance(propobj, property):
            if propobj.fset is None:
                raise AttributeError("can't set attribute")
            propobj.fset(self, v)
        else:
            super().__setattr__(k, v)

    def __getattr__(
            self,
            key: str
    ):
        return self.__dict__[key]

    def __getstate__(self):
        d = {}
        d.update(self.__dict__)
        return d

    def __setstate__(self, state):
        self.from_dict(
            state
        )

    # There is a strange problem with the __str__ method and
    # PyQt therefore it is commented right now
    def __str__(self):
        s = 'class: %s\n' % self.__class__.__name__
        #s += self.to_yaml()
        return s

    def __repr__(self):
        return self.__str__()

    def __init__(
            self,
            name: object = None,
            verbose: bool = False,
            unique_identifier: str = None,
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

        if unique_identifier is None:
            unique_identifier = str(uuid.uuid4())

        # clean up the keys (no spaces etc)
        d = dict()
        for key in kwargs:
            d[clean_string(key)] = kwargs[key]

        # Assign the the names and set standard values
        if name is None:
            name = self.__class__.__name__
        d['name'] = name
        d['verbose'] = verbose
        d['unique_identifier'] = unique_identifier
        kwargs.update(d)
        self.__dict__.update(**kwargs)

    def __copy__(
            self
    ) -> Type[Base]:
        c = self.__class__()
        c.from_dict(
            copy.copy(self.to_dict())
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
            *args,
            **kwargs
    ):
        super().__init__(
            *args,
            **kwargs
        )
        self._data = data
        self._filename = None

        if embed_data is None:
            embed_data = chisurf.settings.cs_settings['database']['embed_data']
        if read_file_size_limit is None:
            read_file_size_limit = chisurf.settings.cs_settings['database']['read_file_size_limit']

        self._embed_data = embed_data
        self._max_file_size = read_file_size_limit

        self.filename = filename

    @property
    def embed_data(
            self
    ) -> bool:
        return self._embed_data

    @embed_data.setter
    def embed_data(
            self,
            v: bool
    ) -> None:
        self._embed_data = v
        if v is False:
            self._data = None

    @property
    def data(
            self
    ) -> bytes:
        return self._data

    @data.setter
    def data(
            self,
            v: Data
    ):
        self._data = v

    @property
    def name(self) -> str:
        try:
            return self.__dict__['name']
        except KeyError:
            return self.filename

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
                    if len(data) > chisurf.settings.cs_settings['database']['compression_data_limit']:
                        data = zlib.compress(data)
                    if len(data) < chisurf.settings.cs_settings['database']['embed_data_limit']:
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


def clean_string(
        s: str
) -> str:
    """ Remove special characters to clean up string and make it compatible
    with a Python variable names

    :param s:
    :return:
    """
    regex_pattern = r'[^-a-z0-9_]+'
    r = slugify(s, separator='_', regex_pattern=regex_pattern)
    return r


def find_objects(
        search_list: Iterable,
        object_type,
        remove_double: bool = True
) -> List[object]:
    """Traverse a list recursively a an return all objects of type `object_type` as
    a list

    :param search_list: list
    :param object_type: an object type
    :param remove_double: boolean
    :return: list of objects with certain object type
    """
    re = list()
    for value in search_list:
        if isinstance(value, object_type):
            re.append(value)
        elif isinstance(value, list):
            re += find_objects(value, object_type)
    if remove_double:
        return list(set(re))
    else:
        return re
