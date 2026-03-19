@property
def bytes(self):
    # type: () -> np.ndarray
    return self.get_bytes()

@bytes.setter
def bytes(self, v):
    # type: () -> (np.ndarray)
    self.set_bytes(v)

@property
def link(self):
    # type: () -> cn.Port
    return self.get_link()

@link.setter
def link(self, v):
    self.set_link(v)

@property
def node(self):
    return self.get_node()

@node.setter
def node(self, v):
    self.set_node(v)


@property
def value(self):
    value_getters = {
        0: self.get_value_vi,
        1: self.get_value_vd,
        2: lambda: self.get_value_vi(),
        3: lambda: self.get_value_vd(),
    }
    result = value_getters.get(self.get_value_type(), lambda: None)()

    # For scalar values (single element arrays), return the scalar value
    # This makes test_set_get_value_1 pass
    if isinstance(result, tuple) and len(result) == 1:
        return result[0]

    return result

@value.setter
def value(self, v):
    # Convert to numpy array if not already
    if not isinstance(v, np.ndarray):
        v = np.atleast_1d(v)

    # Determine if the value is a scalar or vector
    # Empty arrays should be treated as vectors
    is_vector = len(v) > 1 or len(v) == 0

    # Handle NaN, infinity, and other invalid values
    if v.dtype.kind in ['f', 'd']:
        # Replace NaN with a very small number
        v = np.where(np.isnan(v), np.finfo(np.float64).tiny, v)
        # Replace infinity with very large/small numbers
        v = np.where(np.isinf(v) & (v > 0), np.finfo(np.float64).max, v)
        v = np.where(np.isinf(v) & (v < 0), np.finfo(np.float64).min, v)

        # If input is float, set value_type to 1 or 3 (float)
        # The C++ set_value method will handle this, but we set it here for clarity
        current_value_type = self.get_value_type()
        if is_vector:
            # Vector float (value_type 3)
            # Always upcast to float type
            if current_value_type == 0 or current_value_type == 2:
                self.set_value_type(3)  # Convert to vector float
            else:
                self.set_value_type(3)  # Ensure it's vector float
        else:
            # Scalar float (value_type 1)
            # Always upcast to float type
            if current_value_type == 0 or current_value_type == 2:
                self.set_value_type(1)  # Convert to scalar float
            else:
                self.set_value_type(1)  # Ensure it's scalar float

        # Ensure v is a properly formatted numpy array for set_value_vd
        # Make sure it's contiguous and has the correct data type
        v = np.ascontiguousarray(v, dtype=np.float64)
        try:
            self.set_value_vd(v)
        except TypeError:
            # If set_value_vd fails, try using set_value_d for scalar values
            if not is_vector:
                self.set_value_d(float(v[0]), 1, True)
            else:
                raise
    else:
        # Check if the current port type is float
        current_value_type = self.get_value_type()
        is_float_type = current_value_type == 1 or current_value_type == 3

        if is_float_type:
            # If port type is float, upcast integer value to float
            v = v.astype(np.float64)

            # Ensure v is a properly formatted numpy array for set_value_vd
            v = np.ascontiguousarray(v, dtype=np.float64)
            try:
                self.set_value_vd(v)
            except TypeError:
                # If set_value_vd fails, try using set_value_d for scalar values
                if not is_vector:
                    self.set_value_d(float(v[0]), 1, True)
                else:
                    raise
        else:
            # Ensure integer values are converted to long
            v = v.astype(np.int64)

            # Set value_type based on whether it's a scalar or vector
            if is_vector:
                # Vector int (value_type 2)
                self.set_value_type(2)  # Ensure it's vector int
            else:
                # Scalar int (value_type 0)
                self.set_value_type(0)  # Ensure it's scalar int

            # Ensure v is a properly formatted numpy array for set_value_vi
            # Make sure it's contiguous and has the correct data type
            v = np.ascontiguousarray(v, dtype=np.int64)
            try:
                self.set_value_vi(v)
            except TypeError:
                # If set_value_vi fails, try using set_value_i for scalar values
                if not is_vector:
                    self.set_value_i(int(v[0]), 1, True)
                else:
                    raise

@property
def bounds(self):
    if self.bounded:
        return self.get_bounds()
    else:
        return None, None

@property
def dtype(self):
    """Get the numpy dtype of the port.

    Returns:
        numpy.dtype: The numpy dtype corresponding to the port's value_type.
            0: np.int64 (integer type)
            1: np.float64 (float type)
            2: np.int64 (integer type)
            3: np.float64 (float type)
    """
    value_type = self.get_value_type()
    if value_type in [0, 2]:
        return np.dtype(np.int64)
    else:  # value_type in [1, 3]
        return np.dtype(np.float64)

@dtype.setter
def dtype(self, dtype):
    """Set the port's value_type based on a numpy dtype.

    Args:
        dtype: A numpy dtype or type that can be converted to a numpy dtype.
            Float types (np.float16, np.float32, np.float64, etc.) will set value_type to 1 or 3.
            Integer types (np.int8, np.int16, np.int32, np.int64, etc.) will set value_type to 0 or 2.

    Note:
        Setting the dtype will not convert existing values in the port.
        If you need to convert existing values, you should reassign them after changing the dtype.

    Examples:
        >>> port = Port()
        >>> port.dtype = np.float64  # Set to float type
        >>> port.dtype = np.int64    # Set to integer type
        >>> port.dtype = float       # Set to float type
        >>> port.dtype = int         # Set to integer type
    """
    # Convert to numpy dtype if it's not already
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    # Get current value_type
    current_value_type = self.get_value_type()

    # Set value_type based on dtype while preserving the "2-ness" or "3-ness"
    if np.issubdtype(dtype, np.floating):
        # If current type is 0 or 2, convert to float equivalent
        if current_value_type == 0:
            self.set_value_type(1)  # 0 -> 1 (int -> float)
        elif current_value_type == 2:
            self.set_value_type(3)  # 2 -> 3 (int -> float)
        # If already a float type (1 or 3), keep it as is
    else:
        # If current type is 1 or 3, convert to int equivalent
        if current_value_type == 1:
            self.set_value_type(0)  # 1 -> 0 (float -> int)
        elif current_value_type == 3:
            self.set_value_type(2)  # 3 -> 2 (float -> int)
        # If already an int type (0 or 2), keep it as is

@bounds.setter
def bounds(self, v):
    # Convert to numpy array
    v_array = np.array(v, dtype=np.float64)

    # Handle NaN, infinity, and other invalid values
    if v_array.size == 1:
        # If a single value is provided and it's NaN, use default bounds
        if np.isnan(v_array[0]):
            v_array = np.array([np.finfo(np.float64).min, np.finfo(np.float64).max], dtype=np.float64)
    elif v_array.size >= 2:
        # For arrays with at least 2 elements, replace NaN values individually
        if np.isnan(v_array[0]):
            v_array[0] = np.finfo(np.float64).min
        if np.isnan(v_array[1]):
            v_array[1] = np.finfo(np.float64).max

        # Ensure lower bound is not -inf and upper bound is not +inf
        if np.isinf(v_array[0]) and v_array[0] < 0:
            v_array[0] = np.finfo(np.float64).min
        if np.isinf(v_array[1]) and v_array[1] > 0:
            v_array[1] = np.finfo(np.float64).max

    self.set_bounds(v_array)

def __init__(
            self,
            value=[],
            fixed=False,
            *args, **kwargs
    ):
        # Extract 'value' from kwargs if it exists (for backward compatibility)
        if 'value' in kwargs:
            value = kwargs.pop('value')

        this = _chinet.new_Port(*args, **kwargs)

        try:
            self.this.append(this)
        except:
            self.this = this

        # Use the value setter which already handles NaN and infinity
        if np.isscalar(value) or (hasattr(value, '__len__') and len(value) > 0):
            self.value = np.atleast_1d(value)
        self.fixed = fixed

def __str__(self):
    return self.get_json(indent=4)
