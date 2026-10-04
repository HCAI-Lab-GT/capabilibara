"""Anchored platform primitives for no-replace artifact promotion."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import stat
import sys


EntryIdentity = tuple[int, int, int]
_LINUX_NOREPLACE = 1
_DARWIN_EXCLUSIVE = 4


def open_directory(path: Path) -> int:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Promotion parent must be absolute and canonical")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
    except OSError as error:
        os.close(descriptor)
        raise ValueError(
            "Promotion parent contains a symlink or non-directory component"
        ) from error
    return descriptor


def open_stage(directory: int, name: str, metadata: bool = False) -> int:
    flags = (
        os.O_RDONLY
        if metadata
        else getattr(os, "O_PATH", getattr(os, "O_EVTONLY", os.O_RDONLY))
    )
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(name, flags, dir_fd=directory)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError("Promotion stage cannot be a symlink") from error
        raise


def entry_identity(entry: os.stat_result) -> EntryIdentity:
    return entry.st_dev, entry.st_ino, stat.S_IFMT(entry.st_mode)


def rename_noreplace(directory: int, source: str, target: str) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform.startswith("linux"):
        function = getattr(library, "renameat2", None)
        flag = _LINUX_NOREPLACE
    elif sys.platform == "darwin":
        function = getattr(library, "renameatx_np", None)
        flag = _DARWIN_EXCLUSIVE
    else:
        function = None
        flag = 0
    if function is None:
        raise OSError(errno.ENOSYS, "Atomic no-replace rename is unavailable")
    function.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    result = function(
        directory,
        os.fsencode(source),
        directory,
        os.fsencode(target),
        flag,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error, os.strerror(error), target)
    raise OSError(error, os.strerror(error), target)


def quarantine_entry(
    directory: int,
    source: str,
    target: str,
    expected_identity: EntryIdentity,
) -> None:
    rename_noreplace(directory, source, target)
    observed = os.stat(target, dir_fd=directory, follow_symlinks=False)
    if entry_identity(observed) != expected_identity:
        raise ValueError("Rejected publication entry changed during quarantine")


def unlink_stage(directory: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory)
    except FileNotFoundError:
        pass
