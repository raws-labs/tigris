"""Stage a checksummed native runtime archive for wheel construction."""

import argparse
import hashlib
import io
import json
import re
import platform as host_platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from packaging.tags import sys_tags

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "src" / "tigris" / "native"


def build_source(source: Path):
    source = source.resolve()
    build = source / "build-host-ci"
    machine = host_platform.machine().lower()
    if sys.platform == "win32":
        platform = "win_amd64"
    elif sys.platform == "darwin":
        platform = f"macosx_11_0_{machine}"
    else:
        platform = f"linux_{machine}"
    arguments = ["cmake", "-S", str(source), "-B", str(build), "-G", "Ninja",
                 "-DCMAKE_BUILD_TYPE=Release", "-DTIGRIS_BUILD_HOST=ON",
                 f"-DTIGRIS_HOST_PLATFORM={platform}"]
    if sys.platform == "win32":
        arguments.append("-DCMAKE_C_COMPILER=clang")
    elif sys.platform == "darwin":
        arguments.append("-DCMAKE_OSX_DEPLOYMENT_TARGET=11.0")
    subprocess.run(arguments, check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "test_host", "--parallel", "4"], check=True)
    subprocess.run(["ctest", "--test-dir", str(build), "-R", "^host_library$", "--output-on-failure"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "host_archive"], check=True)
    directory = build / "host-artifacts"
    manifest = json.loads((directory / "stage" / "manifest.json").read_text())
    archive = directory / f"tigris-host-{manifest['version']}-{platform}.tar.gz"
    data = archive.read_bytes()
    checksum = archive.with_suffix(archive.suffix + ".sha256").read_text().split()[0]
    return stage(data, checksum, DESTINATION, platform=platform)


def stage(data: bytes, checksum: str, destination: Path, *, release=None, platform=None):
    if not re.fullmatch(r"[0-9a-f]{64}", checksum) or hashlib.sha256(data).hexdigest() != checksum:
        raise ValueError("Runtime archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        libraries = {"libtigris_host.so", "libtigris_host.dylib", "tigris_host.dll"}
        if len(names) != 4 or len(set(names)) != 4 or not {"manifest.json", "LICENSE", "tigris_host.h"} <= set(names):
            raise ValueError("Unexpected runtime archive contents")
        if any(not member.isfile() or member.name not in libraries | {"manifest.json", "LICENSE", "tigris_host.h"}
               for member in members):
            raise ValueError("Unsafe runtime archive member")
        contents = {member.name: archive.extractfile(member).read() for member in members}
    manifest = json.loads(contents["manifest.json"])
    if manifest.get("abi") != 1 or manifest.get("library") not in libraries:
        raise ValueError("Unsupported runtime archive ABI or library")
    if hashlib.sha256(contents[manifest["library"]]).hexdigest() != manifest.get("sha256"):
        raise ValueError("Runtime library checksum mismatch")
    if release is not None and manifest.get("version") != release.removeprefix("v"):
        raise ValueError("Runtime archive version differs from pin")
    if platform is not None and manifest.get("platform") != platform:
        raise ValueError("Runtime archive platform differs from pin")
    if manifest.get("platform") not in {tag.platform for tag in sys_tags()}:
        raise ValueError("Runtime archive is not compatible with this build host")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        for name in (manifest["library"], "LICENSE", "manifest.json"):
            (Path(temporary) / name).write_bytes(contents[name])
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(temporary, destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--archive", type=Path, help="Use a locally built archive")
    source.add_argument("--source", type=Path, help="Build, test, and stage a runtime source checkout")
    parser.add_argument("--sha256", help="Required checksum for --archive")
    parser.add_argument("--platform", help="Select a platform from the release pin")
    parser.add_argument("--expect-release", help="Require the pin and compiler tag to name this release")
    args = parser.parse_args()
    if args.source:
        manifest = build_source(args.source)
    elif args.archive:
        if not args.sha256:
            parser.error("--archive requires --sha256")
        manifest = stage(args.archive.read_bytes(), args.sha256, DESTINATION, platform=args.platform)
    else:
        pin = json.loads((ROOT / "runtime-host.json").read_text())
        release = pin["release"]
        if not release or not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", release):
            parser.error("Pin a published host runtime release in runtime-host.json before building release wheels")
        if args.expect_release:
            from setuptools_scm import get_version

            if release != args.expect_release or get_version(root=ROOT) != release[1:]:
                parser.error("Compiler tag, setuptools-scm version, and runtime release pin must agree")
        platform = args.platform or next((tag.platform for tag in sys_tags() if tag.platform in pin["artifacts"]), None)
        if platform not in pin["artifacts"]:
            parser.error("No pinned runtime archive for this platform")
        artifact = pin["artifacts"][platform]
        filename = f"tigris-host-{release[1:]}-{platform}.tar.gz"
        url = f"https://github.com/raws-labs/tigris-runtime/releases/download/{release}/{filename}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        manifest = stage(data, artifact["sha256"], DESTINATION, release=release, platform=platform)
    print(f"Staged runtime {manifest['version']} for {manifest['platform']}")


if __name__ == "__main__":
    main()
