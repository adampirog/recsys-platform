import shutil
from argparse import Namespace
from pathlib import Path

from pytest import fixture, raises
from python_utils.archives import create_archive, extract_archive
from python_utils.archives import main as script_main


@fixture(scope="function")
def test_dir(tmp_path) -> tuple[Path, Path]:
    folder = tmp_path / "folder"
    folder.mkdir(exist_ok=True, parents=True)

    return tmp_path, folder


def test_archive_creation(test_dir):
    root_dir, folder = test_dir

    for i in range(10):
        file = folder / f"file_{i}.txt"
        file.touch()

    target = root_dir / "target.tgz"
    create_archive(folder, target)
    assert target.is_file()


def test_archive_creation_default_target(test_dir):
    _, folder = test_dir

    for i in range(10):
        file = folder / f"file_{i}.txt"
        file.touch()

    target = folder.with_suffix(".tgz")
    create_archive(folder)
    assert target.is_file()


def test_archive_extraction(test_dir):
    root_dir, folder = test_dir

    for i in range(10):
        file = folder / f"file_{i}.txt"
        file.touch()

    target = root_dir / "target.tgz"
    create_archive(folder, target)
    assert target.is_file()

    restore = root_dir / "restore"

    extract_archive(target, restore)
    assert restore.is_dir()

    for index, file in enumerate(sorted(restore.glob("*"))):
        assert file.name == f"file_{index}.txt"


def test_archive_extraction_default_target(test_dir):
    root_dir, folder = test_dir

    for i in range(10):
        file = folder / f"file_{i}.txt"
        file.touch()

    target = root_dir / "target.tgz"
    create_archive(folder, target)
    assert target.is_file()

    restore = root_dir / "target"

    extract_archive(target)
    assert restore.is_dir()

    for index, file in enumerate(sorted(restore.glob("*"))):
        assert file.name == f"file_{index}.txt"


def test_script_nofile_error(test_dir):
    args = Namespace(source="dummy")
    with raises(FileNotFoundError):
        script_main(args)


def test_script_wrongfile_error(test_dir):
    root_dir, _ = test_dir
    file = root_dir / "dummy.tgz"
    file.touch()

    args = Namespace(source=str(file), destination=None)
    with raises(shutil.ReadError, match=r"is not a .*?tar file"):
        script_main(args)


def test_script_creation(test_dir):
    root_dir, folder = test_dir

    for i in range(10):
        file = folder / f"file_{i}.txt"
        file.touch()

    target = root_dir / "target.tgz"
    args = Namespace(source=str(folder), destination=str(target))
    script_main(args)
    assert target.is_file()


def test_script_extraction(test_dir):
    root_dir, folder = test_dir

    for i in range(10):
        file = folder / f"file_{i}.txt"
        file.touch()

    target = root_dir / "target.tgz"
    args = Namespace(source=str(folder), destination=str(target))
    script_main(args)
    assert target.is_file()

    restore = root_dir / "restore"

    args = Namespace(source=str(target), destination=str(restore))
    script_main(args)
    assert restore.is_dir()

    for index, file in enumerate(sorted(restore.glob("*"))):
        assert file.name == f"file_{index}.txt"
