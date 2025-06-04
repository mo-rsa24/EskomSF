from uuid import uuid4

from db.utilities import load_yaml_config


from profiler.timer.ProfilerTimer import get_logger, ProfilerTimer
from profiler.profiler_run import run_context


class NoOpProfiler:
    pass


def conditional_timer(module, function, message="", category="general", context=None):
    config = load_yaml_config()
    profiling_cfg = config.get("profiling", {})
    enabled = profiling_cfg.get("enabled", False)

    if enabled:
        logger = get_logger(
            engine=profiling_cfg.get("engine", "mariadb"),
            env=profiling_cfg.get("environment", "dev")
        )
        return ProfilerTimer(
            module=module,
            function=function,
            logger_backend=logger,
            message=message,
            category=category,
            run_id=run_context.run_id,
            app_name=run_context.app_name,
            context=context
        )
    else:
        return NoOpProfiler()
