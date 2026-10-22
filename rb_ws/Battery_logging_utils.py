"""Run logging helpers for Battery_rb.py."""

import atexit
import builtins
import inspect
import os
import sys
from datetime import datetime


class TeeStream:
    def __init__(
        self,
        stream,
        log_file,
    ):
        self.stream = stream
        self.log_file = log_file

    def write(
        self,
        message,
    ):
        self.stream.write(message)
        self.log_file.write(message)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def __getattr__(
        self,
        name,
    ):
        return getattr(
            self.stream,
            name,
        )


class ContextualPrint:
    def __init__(
        self,
        original_print,
        workspace_root,
    ):
        self.original_print = original_print
        self.workspace_root = workspace_root

    def format_prefix(self):
        caller_frame = inspect.currentframe()

        if caller_frame is not None:
            caller_frame = caller_frame.f_back

        if caller_frame is not None:
            caller_frame = caller_frame.f_back

        if caller_frame is not None:
            code = caller_frame.f_code
            file_path = code.co_filename

            try:
                file_path = os.path.relpath(
                    file_path,
                    self.workspace_root,
                )
            except ValueError:
                pass

            function_name = code.co_name
            line_number = caller_frame.f_lineno
        else:
            file_path = "<unknown>"
            function_name = "<unknown>"
            line_number = 0

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )[:-3]

        return (
            f"[{timestamp}] "
            f"[{file_path}:{function_name}:{line_number}]"
        )

    def __call__(
        self,
        *args,
        sep=" ",
        end="\n",
        file=None,
        flush=False,
    ):
        target_file = file if file is not None else sys.stdout

        if target_file not in (
            sys.stdout,
            sys.stderr,
        ):
            self.original_print(
                *args,
                sep=sep,
                end=end,
                file=target_file,
                flush=flush,
            )
            return

        prefix = self.format_prefix()
        message = sep.join(
            str(arg)
            for arg in args
        )

        if "\n" in message:
            message = "\n".join(
                f"{prefix} {line}"
                for line in message.split("\n")
            )
        else:
            message = f"{prefix} {message}"

        self.original_print(
            message,
            end=end,
            file=target_file,
            flush=flush,
        )


class RunLogContext:
    def __init__(
        self,
        log_file,
        original_stdout,
        original_stderr,
        original_print,
    ):
        self.log_file = log_file
        self.original_stdout = original_stdout
        self.original_stderr = original_stderr
        self.original_print = original_print
        self.closed = False

    def close(self):
        if self.closed:
            return

        builtins.print = self.original_print
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.log_file.flush()
        self.log_file.close()
        self.closed = True


def setup_run_logging(
    log_dir="/home/seungjun/isaac_ws/run_logs",
    workspace_root="/home/seungjun/isaac_ws",
):
    os.makedirs(
        log_dir,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    log_path = os.path.join(
        log_dir,
        f"battery_rb_{timestamp}.log",
    )

    log_file = open(
        log_path,
        "a",
        buffering=1,
    )

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    original_print = builtins.print

    sys.stdout = TeeStream(
        original_stdout,
        log_file,
    )

    sys.stderr = TeeStream(
        original_stderr,
        log_file,
    )

    builtins.print = ContextualPrint(
        original_print=original_print,
        workspace_root=workspace_root,
    )

    log_context = RunLogContext(
        log_file=log_file,
        original_stdout=original_stdout,
        original_stderr=original_stderr,
        original_print=original_print,
    )

    atexit.register(
        log_context.close
    )

    print(
        "[LOG] Run log:",
        log_path,
        flush=True,
    )

    return log_path
