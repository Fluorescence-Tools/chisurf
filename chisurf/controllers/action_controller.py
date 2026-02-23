from __future__ import annotations

from chisurf import typing
from chisurf.controllers.services import dataset_service, fit_service, project_service, setup_service


class ActionController:
    def __init__(self):
        self._handlers: typing.Dict[str, typing.Callable[..., typing.Any]] = {
            "action_catalog_export": self._handle_action_catalog_export,
            "experiment_set": self._handle_experiment_set,
            "setup_select": self._handle_setup_select,
            "setup_params_set": self._handle_setup_params_set,
            "dataset_add": self._handle_dataset_add,
            "dataset_remove": self._handle_dataset_remove,
            "dataset_group": self._handle_dataset_group,
            "dataset_ungroup": self._handle_dataset_ungroup,
            "fit_add": self._handle_fit_add,
            "fit_close": self._handle_fit_close,
            "fit_set_dataset": self._handle_fit_set_dataset,
            "fit_run": self._handle_fit_run,
            "fit_run_execute": self._handle_fit_run_execute,
            "fit_run_start": self._handle_fit_run_event,
            "fit_run_finish": self._handle_fit_run_event,
            "fit_run_abort": self._handle_fit_run_event,
            "project_save": self._handle_project_save,
            "project_load": self._handle_project_load,
            "project_close": self._handle_project_close,
        }

    def _resolve_name(self, name: str) -> str:
        import chisurf

        dispatcher = getattr(chisurf, "action_dispatcher", None)
        registry = getattr(dispatcher, "registry", None)
        if registry is not None and hasattr(registry, "resolve_name"):
            return str(registry.resolve_name(name))
        return str(name).replace(".", "_")

    def execute(
            self,
            name: str,
            payload: typing.Optional[typing.Dict[str, typing.Any]] = None,
            context: typing.Optional[typing.Dict[str, typing.Any]] = None,
    ) -> typing.Dict[str, typing.Any]:
        canonical = self._resolve_name(name)
        data = dict(payload or {})
        ctx = dict(context or {})

        handler = self._handlers.get(canonical)
        if handler is None:
            raise KeyError(f"No handler registered for action '{canonical}'")

        result = handler(data, ctx)
        return {
            "ok": True,
            "action": canonical,
            "result": result,
        }

    @staticmethod
    def list_actions() -> typing.List[typing.Dict[str, typing.Any]]:
        import chisurf

        provider = getattr(chisurf, "action_catalog", None)
        if callable(provider):
            value = provider()
            if isinstance(value, list):
                return value
        return []

    @staticmethod
    def _handle_project_save(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        project_service.save_project(
            target_path=str(payload.get("target_path") or ""),
            project_name=str(payload.get("project_name") or "chisurf_project"),
        )

    @staticmethod
    def _handle_experiment_set(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        setup_service.set_experiment(name=str(payload.get("name") or ""))

    @staticmethod
    def _handle_setup_select(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        setup_service.select_setup(name=str(payload.get("name") or ""))

    @staticmethod
    def _handle_setup_params_set(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        params = payload.get("params") or {}
        setup_service.set_setup_params(params=params)

    @staticmethod
    def _handle_fit_add(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        dataset_indices = payload.get("dataset_indices") or []
        model_name = payload.get("model_name")
        model_kw = payload.get("model_kw")
        fit_service.add_fit(dataset_indices=list(dataset_indices), model_name=model_name, model_kw=model_kw)

    @staticmethod
    def _handle_fit_close(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        idx = payload.get("idx")
        fit_service.close_fit(idx=None if idx is None else int(idx))

    @staticmethod
    def _handle_fit_set_dataset(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        fit_service.set_fit_dataset(
            fit_index=int(payload.get("fit_index", 0)),
            dataset_index=int(payload.get("dataset_index", -1)),
        )

    @staticmethod
    def _handle_fit_run(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        fit_service.run_fit(fit_index=int(payload.get("fit_index", 0)))

    @staticmethod
    def _handle_dataset_add(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        dataset_service.add_dataset(payload=payload)

    @staticmethod
    def _handle_dataset_remove(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        dataset_indices = payload.get("dataset_indices") or []
        dataset_service.remove_datasets(dataset_indices=list(dataset_indices))

    @staticmethod
    def _handle_dataset_group(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        dataset_indices = payload.get("dataset_indices") or []
        dataset_service.group_datasets(dataset_indices=list(dataset_indices))

    @staticmethod
    def _handle_dataset_ungroup(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        dataset_indices = payload.get("dataset_indices") or []
        dataset_service.ungroup_datasets(dataset_indices=list(dataset_indices))

    @staticmethod
    def _handle_fit_run_execute(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        fit_controller = payload.get("fit_controller")
        if fit_controller is None:
            fit_controller = context.get("fit_controller")
        fit_service.run_fit_controller_impl(fit_controller=fit_controller)

    @staticmethod
    def _handle_fit_run_event(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        action_type = str(context.get("action_type") or "fit_run_start")
        summary = str(context.get("summary") or action_type)
        source_uid = context.get("source_uid")
        fit_service.record_fit_run_event(action_type=action_type, summary=summary, payload=payload, source_uid=source_uid)

    @staticmethod
    def _handle_project_load(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        project_service.load_project(project_path=str(payload.get("project_path") or ""))

    @staticmethod
    def _handle_project_close(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> None:
        project_service.close_project(
            main_window=payload.get("main_window"),
            current_project_dir=str(payload.get("current_project_dir") or ""),
        )

    @staticmethod
    def _handle_action_catalog_export(
            payload: typing.Dict[str, typing.Any],
            context: typing.Dict[str, typing.Any],
    ) -> str:
        return project_service.export_action_catalog(
            target_path=str(payload.get("target_path") or ""),
            file_type=str(payload.get("file_type") or "yaml"),
        )
