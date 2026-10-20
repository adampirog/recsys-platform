"""
Simple and convenient utilities for managing file archives.

Script allows to create or extract given archive.
"""

import shutil
import subprocess
import tarfile
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory


@dataclass(slots=True, frozen=True)
class ArchiveFormat:
    name: str
    extensions: tuple
    cmd_options: str

    @property
    def open_options(self) -> str:
        return self.name.removesuffix("tar")

    @property
    def suffix(self) -> str:
        return self.extensions[0]

    @property
    def open_mode(self) -> str:
        if "tar" in self.name:
            return self.name.removesuffix("tar")

        return ""

    def match(self, file: str | Path) -> bool:
        file = str(file)
        for extension in self.extensions:
            if file.endswith(extension):
                return True
        return False


ARCHIVE_FORMATS = (
    ArchiveFormat("gztar", (".tgz", ".tar.gz"), "czf"),
    ArchiveFormat("tar", (".tar",), "cf"),
    ArchiveFormat("xztar", (".tar.xz", ".txz"), "cJf"),
    ArchiveFormat("bz2tar", (".tar.bz2", ".tbz2"), "cjf"),
    ArchiveFormat("zip", (".zip",), ""),
)

DEFAULT_ARCHIVE_FORMAT = ARCHIVE_FORMATS[0]


def get_archive_format(file: str | Path) -> ArchiveFormat | None:
    """
    Infers archive format from a file name
    """
    for archive_format in ARCHIVE_FORMATS:
        if archive_format.match(file):
            return archive_format

    return None


def create_archive(source: str | Path, destination: str | Path = "") -> str:
    """
    Creates an archive from source directory.

    If destination is not given, creates archive with the same name.
    If archive format cannot be inferred from the extension '.tgz' archive is
    created
    """
    if destination:
        archive_format = get_archive_format(destination)
        if not archive_format:
            archive_format = DEFAULT_ARCHIVE_FORMAT
            destination = Path(destination).with_suffix(archive_format.suffix)
    else:
        archive_format = DEFAULT_ARCHIVE_FORMAT
        destination = Path(source).with_suffix(archive_format.suffix)

    if archive_format.name == "zip":
        return create_zip(source, destination)

    return create_tar(source, destination, archive_format=archive_format)


def extract_archive(source: str | Path, destination: str | Path = "") -> None:
    """
    Extracts an archive.

    If destination is not given, it will be unpacked to a folder, next to the
    archive, with the same name minus the extension
    """
    archive_format = get_archive_format(source)
    if not archive_format:
        raise ValueError(f"'{source}' is not a valid archive.")

    if not destination:
        destination = Path(source).with_suffix("")

    shutil.unpack_archive(
        source, destination, filter=None if archive_format.name == "zip" else "data"
    )


def create_zip(source: str | Path, destination: str | Path) -> str:
    return shutil.make_archive(str(destination).removesuffix(".zip"), "zip", root_dir=source)


def _python_tar(source: str | Path, destination: str | Path, archive_format: ArchiveFormat) -> str:
    """
    Create a tar archive with Python tools
    """
    with tarfile.open(destination, f"w:{archive_format.open_mode}") as tar:
        for file in Path(source).glob("*"):
            tar.add(file, arcname=file.name)

    return str(destination)


def _os_tar(source: str | Path, destination: str | Path, archive_format: ArchiveFormat) -> str:
    """
    Create a tar archive with os tools

    C-based os tar is faster than Python
    """
    destination = Path(destination).resolve()

    files = [file.name for file in Path(source).glob("*")]
    subprocess.run(
        ["tar", archive_format.cmd_options, str(destination), *files],
        cwd=str(source),
        check=True,
    )

    return str(destination)


def create_tar(source: str | Path, destination: str | Path, archive_format: ArchiveFormat) -> str:
    if "tar" not in archive_format.name:
        raise ValueError(f"Archive format '{archive_format}' not compatible.")

    if shutil.which("tar"):
        return _os_tar(source, destination, archive_format=archive_format)

    return _python_tar(source, destination, archive_format=archive_format)


@contextmanager
def temporary_extract(archive: str | Path) -> Generator[Path]:
    """
    Extract an archive to a temporary directory
    """
    archive = Path(archive)

    if archive.is_dir():
        yield archive
    elif archive.is_file():
        with TemporaryDirectory() as temp_dir:
            extract_archive(archive, temp_dir)
            yield Path(temp_dir)
    else:
        raise FileNotFoundError(f"Path '{archive}' is not valid.")


@contextmanager
def repack_archive(archive: str | Path) -> Generator[Path]:
    """
    Unpack and then repack archive
    """
    archive = Path(archive)
    if archive.is_dir():
        yield archive
    elif archive.is_file():
        with temporary_extract(archive) as unpacked:
            yield unpacked
            create_archive(unpacked, archive)
    else:
        raise FileNotFoundError(f"Path '{archive}' is not valid")
