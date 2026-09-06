# ABACUS 3.11 H0Lite exporter

See [README_zh.md](README_zh.md) for the complete Chinese user and build guide.

Download `abacus-h0lite-v311_source.tar.gz` from this directory. It contains
the already-patched H0Lite sources, required ABACUS source/header dependencies,
standalone CMake build, build script, and licenses. No separate ABACUS download
or patch application is needed. The legacy
`abacus-postprocess-v353_source.tar.gz` archive is retained. The loose patch is
only for maintaining a full ABACUS developer checkout.

Build prerequisites are x86_64 Linux, GCC 9+ (10.2+ recommended), CMake 3.20+,
GNU Make, binutils, `file`, GCC static runtimes (`libstdc++`, `libgcc`,
`libgomp.a`), and oneMKL static development libraries (2023.2 recommended).
Set `MKLROOT` to that oneMKL installation. Once dependencies are installed,
the build needs no network, Python, Conda, MPI, or full ABACUS installation.
The installation must also supply `license.txt` and `third-party-programs.txt`.
The build finds them beside the actual linked MKL library, including unified
oneAPI symlinks. For a nonstandard layout, set `H0LITE_MKL_LICENSE_DIR` to the
licensing directory from that exact installation; another version's notices
are not a substitute.

This archive was built from a fresh extraction with GCC 10.5.0, CMake 3.31.7,
oneMKL 2023.2.0, and glibc 2.28. Both default H0 and `--with-vl` execution
passed; only basic glibc libraries remained dynamically linked.
The 2026-09-06 licensing revision was rebuilt from a fresh extraction with
oneMKL 2024.2 and passed both calculation modes and embedded-license checks.
License discovery was also checked for 2023.2/2024.2 and missing notices.

Extract outside the repository and build on allocated compute resources:

```bash
tar -xzf abacus-h0lite-v311_source.tar.gz
cd abacus-h0lite-v311_source
export MKLROOT=/path/to/oneapi/mkl/2023.2.0
BUILD_CPUS=8 bash ./build_h0lite_single.sh
./bin/abacus_h0 --version
```

The script uses `SLURM_CPUS_PER_TASK` when set, otherwise `BUILD_CPUS` (default
1). The build directory is `build-h0lite-single/`; the runtime file is
`bin/abacus_h0`. Binary redistribution also requires matching source and
license materials; see [NOTICE.md](NOTICE.md). The original
`SOURCE_DIR BUILD_DIR OUTPUT_FILE` arguments
remain supported. Edit the bundled sources and rerun to rebuild; register new
translation units in `cmake/Sources.cmake`. See `SOURCE_INFO.md` in the archive
for source provenance. Build on a glibc baseline no newer than the target
machine; CentOS 7 compatibility requires building on glibc 2.17.

Load a GCC module as well as oneMKL: loading oneAPI alone does not select a
newer `g++`. The script checks GCC before configuring and passes its absolute
path to CMake. On a compiler change it preserves the old CMake cache and
compiler metadata in `BUILD_DIR/compiler-cache-backup.XXXXXX/` and configures
again. Rerun the same command after loading GCC 9+; manual cache deletion is
not required. The Chinese guide includes the current mgt module commands.

`abacus_h0` is a single-file, MPI-free x86_64 Linux executable derived from
ABACUS 3.11. It reads an ordinary LCAO SCF case but evaluates only
`H0=T+Vnl` and `S0`; `--with-vl` changes H0 to `T+Vnl+Vl`. It runs no SCF
iteration and no diagonalization.

```bash
cd CASE
/path/to/abacus_h0

/path/to/abacus_h0 CASE --with-vl

/path/to/abacus_h0 --case-list cases.txt --tasks 8 --cpus-per-task 4
```

For one case, the CPU count defaults to `SLURM_CPUS_PER_TASK`, or the CPUs
available to the process outside Slurm. In a Slurm multi-task or array launch,
the case list is sharded automatically before each node runs its local dynamic
case queue.

The runtime artifact is one `abacus_h0` file; do not commit that generated ELF
back into this source directory.

## Licensing

H0Lite is modified ABACUS, not an official upstream release. Full GPLv3 and
LGPLv3 texts are provided in `COPYING` and `COPYING.LESSER`, with attribution,
modification dates and redistribution guidance in [NOTICE.md](NOTICE.md).
`--license` also embeds the actual build's oneMKL license/third-party notices,
the GCC Runtime Library Exception, and retained third-party attribution.
One-file execution does not waive corresponding-source obligations: provide
the exact modified sources and build scripts for a distributed binary, for
example alongside it at the same download location. Sources need not be
installed on the execution node.
