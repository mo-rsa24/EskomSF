# profiler_utils.py
"""
Refactored to use logger_factory.get_logger() for profiling logger selection.
Avoids repetitive engine/config resolution.
"""

from uuid import uuid4
from config.config_service import  is_profiling_enabled
from profiler.profiler_run import run_context
from utils.logger_factory import get_logger
from profiler.timer.ProfilerTimer import ProfilerTimer


class NoOpProfiler:
    pass

def conditional_timer(module, function, message="", category="general", context=None):
    profiling_enabled = is_profiling_enabled()

    if profiling_enabled:
        logger = get_logger()  # DB logger (MariaDB or SQLServer)
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
