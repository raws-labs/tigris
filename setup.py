"""Build a pure wheel or package a staged host library in a platform wheel."""

import hashlib
import json
import os
from pathlib import Path

from setuptools import Distribution, setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build import build
from setuptools.command.build_py import build_py


VARIANT = os.environ.get("TIGRIS_WHEEL_VARIANT", "pure")
if VARIANT not in {"pure", "native"}:
    raise ValueError("TIGRIS_WHEEL_VARIANT must be pure or native")


class VariantBuild(build):
    def finalize_options(self):
        self.build_base = str(Path("build") / VARIANT)
        super().finalize_options()


class VariantBuildPy(build_py):
    def find_data_files(self, package, src_dir):
        files = super().find_data_files(package, src_dir)
        if VARIANT == "pure" and package == "tigris":
            files = [name for name in files if Path(name).relative_to(src_dir).parts[0] != "native"]
        return files


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return VARIANT == "native" and "editable_wheel" not in self.commands


class HostWheel(bdist_wheel):
    def get_tag(self):
        if VARIANT == "pure" or "editable_wheel" in self.distribution.commands:
            return "py3", "none", "any"
        directory = Path(__file__).parent / "src" / "tigris" / "native"
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError("Stage the host library with scripts/runtime_bundle.py before building a wheel")
        manifest = json.loads(manifest_path.read_text())
        name = manifest["library"]
        if name not in {"libtigris_host.so", "libtigris_host.dylib", "tigris_host.dll"}:
            raise RuntimeError("Invalid staged runtime library")
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != manifest["sha256"]:
            raise RuntimeError("Staged runtime checksum mismatch")
        self.root_is_pure = False
        return "py3", "none", manifest["platform"]


setup(distclass=NativeDistribution, cmdclass={"bdist_wheel": HostWheel,
                                            "build": VariantBuild, "build_py": VariantBuildPy})
