from __future__ import annotations
from chisurf import typing

import numpy as np
import chinet as cn
import types
import inspect

import chisurf.decorators
import chisurf.parameter
import chisurf.fitting.fit
import chisurf.models

from chisurf.curve import Curve
from chisurf.models import model
from chisurf.fitting.parameter import GlobalFittingParameter


class ParameterTransformModel(model.Model):

    name = "Parameter Transform"

    def finalize(self):
        self.update_model()
        for i, p in enumerate(self.parameters_all):
            if hasattr(p, 'controller') and p.controller is not None:
                p.controller.finalize()

    def update_model(self, **kwargs):
        # Temporarily unlock outputs for evaluation
        for output in self._model._node.outputs.values():
            output.fixed = False

        # Evaluate the model node with error handling
        try:
            self._model._node.evaluate()
        except Exception as e:
            # Log the error but continue
            import logging
            logging.warning(f"Error during model node evaluation: {str(e)}")
            # Don't re-raise the exception to allow the UI to continue functioning

        # Lock the outputs again after evaluation
        for output in self._model._node.outputs.values():
            output.fixed = True

    @property
    def n_points(self):
        return 1

    @property
    def n_free(self):
        return 0

    @property
    def weighted_residuals(self) -> np.ndarray:
        return np.array([], dtype=np.float64)

    @property
    def function(self) -> str:
        return self._function

    @function.setter
    def function(self, fun: str):
        # Validate the function string before using it
        # Try to compile the function to check for syntax errors
        code_obj = compile(fun, '<string>', 'exec')
        function_obj = None

        # Extract the function object from the compiled code
        for o in code_obj.co_consts:
            if isinstance(o, types.CodeType):
                function_obj = types.FunctionType(o, globals())
                break

        if function_obj is None:
            raise ValueError("No function found in the provided code")

        # Get the function signature
        sig = inspect.signature(function_obj)

        # Create default arguments for all parameters
        default_args = {}
        for param_name, param in sig.parameters.items():
            if param.default is not inspect.Parameter.empty:
                default_args[param_name] = param.default
            elif param.annotation == 'int':
                default_args[param_name] = 1
            elif param.annotation == 'float':
                default_args[param_name] = 1.0
            else:
                default_args[param_name] = 0.0

        # Call the function with default arguments, but catch any errors
        try:
            result = function_obj(**default_args)
        except Exception as e:
            # Log the error but continue with model creation
            import logging
            logging.warning(f"Error evaluating function with default parameters: {str(e)}")
            # Create a dummy result with expected output keys
            # This allows the model to be created even if evaluation fails
            result = {}

        self._function = fun
        m = chisurf.models.function_to_model_decorator(name=self.name)

        # Create the model class with the function
        model_class = m(fun)

        if self.fit is None:
            raise ValueError("Fit object cannot be None")

        self._model = model_class(self.fit)

    @property
    def _parameters(self) -> typing.List[chisurf.fitting.parameter.FittingParameter]:
        return self._model.parameters_all

    @_parameters.setter
    def _parameters(self, v):
        pass

    def __init__(
            self,
            fit: chisurf.fitting.fit.Fit,
            function: typing.Callable = None,
            *args,
            **kwargs
    ):
        if function is None:
            function = 'def f(x): return x'
        self.fit = fit
        self.function = function
        super().__init__(fit, *args, **kwargs)

    def __str__(self):
        s = "\n"
        s += "Model: Parameter transform\n"
        s += "\n"
        s += "Function:\n"
        s += str(self._function)
        s += "\n"
        s += "Parameter \t Value \t Bounds \t Output \t Linked\n"
        for p in self.parameters_all:
            s += f"{p.name} \t {p.value:.4f} \t {p.bounds} \t {p.fixed} \t {p.is_linked} \n"
        s += "\n"
        return s

    def __getitem__(self, key):
        start = key.start
        stop = key.stop
        step = 1 if key.step is None else key.step
        return self.x[start:stop:step], self.y[start:stop:step]
