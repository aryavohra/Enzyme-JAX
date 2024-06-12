load("//:workspace.bzl", "JAX_COMMIT", "JAX_SHA256", "ENZYME_COMMIT", "ENZYME_SHA256", "PYRULES_COMMIT", "PYRULES_SHA256", "XLA_PATCHES")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "rules_cc",
    sha256 = "85723d827f080c5e927334f1fb18a294c0b3f94fee6d6b45945f5cdae6ea0fd4",
    strip_prefix = "rules_cc-c8c38f8c710cbbf834283e4777916b68261b359c",
    urls = [
        "https://github.com/bazelbuild/rules_cc/archive/c8c38f8c710cbbf834283e4777916b68261b359c.tar.gz",
    ],
)

load("@rules_cc//cc:repositories.bzl", "rules_cc_dependencies")

rules_cc_dependencies()

LLVM_TARGETS = ["X86", "AArch64", "AMDGPU", "NVPTX"]

http_archive(
    name = "jax",
    sha256 = JAX_SHA256,
    strip_prefix = "jax-" + JAX_COMMIT,
    urls = ["https://github.com/google/jax/archive/{commit}.tar.gz".format(commit = JAX_COMMIT)],
    patch_args = ["-p1"],
    patches = ["//:patches/jax.patch"],
)

load("@jax//third_party/xla:workspace.bzl", "XLA_COMMIT", "XLA_SHA256")

http_archive(
    name = "xla",
    sha256 = XLA_SHA256,
    strip_prefix = "xla-" + XLA_COMMIT,
    urls = ["https://github.com/wsmoses/xla/archive/{commit}.tar.gz".format(commit = XLA_COMMIT)],
    patch_cmds = XLA_PATCHES
)

http_archive(
    name = "rules_python",
    sha256 = PYRULES_SHA256,
    strip_prefix = "rules_python-" + PYRULES_COMMIT,
    urls = ["https://github.com/bazelbuild/rules_python/archive/{commit}.tar.gz".format(commit = PYRULES_COMMIT)]
)

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

load("@rules_python//python/pip_install:repositories.bzl", "pip_install_dependencies")

pip_install_dependencies()

http_archive(
    name = "enzyme",
    sha256 = ENZYME_SHA256,
    strip_prefix = "Enzyme-" + ENZYME_COMMIT + "/enzyme",
    urls = ["https://github.com/EnzymeAD/Enzyme/archive/{commit}.tar.gz".format(commit = ENZYME_COMMIT)],
)

http_archive(
    name = "upb",
    sha256 = "61d0417abd60e65ed589c9deee7c124fe76a4106831f6ad39464e1525cef1454",
    strip_prefix = "upb-9effcbcb27f0a665f9f345030188c0b291e32482",
    patch_cmds = [
        "sed -i.bak0 's/@bazel_tools\\/\\/platforms:windows/@platforms\\/\\/os:windows/g' BUILD",
        "sed -i.bak0 's/-Werror//g' BUILD"
    ],
    url = "https://github.com/protocolbuffers/upb/archive/9effcbcb27f0a665f9f345030188c0b291e32482.tar.gz"
)

http_archive(
    name = "cxx.rs",
    sha256 = "fd3eb6ea6048648c6b5cd697d9e3074d284384cd98298c4ec7ef9a06fe17c50f",
    strip_prefix = "cxx-1.0.123",
    url = "https://github.com/dtolnay/cxx/archive/refs/tags/1.0.123.tar.gz"
)

load("@cxx.rs//tools/bazel:extension.bzl", "crate_repositories")
crate_repositories()

http_archive(
    name = "rules_rust",
    integrity = "sha256-JLN47ZcAbx9wEr5Jiib4HduZATGLiDgK7oUi/fvotzU=",
    urls = ["https://github.com/bazelbuild/rules_rust/releases/download/0.42.1/rules_rust-v0.42.1.tar.gz"],
)

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")
rules_rust_dependencies()
rust_register_toolchains()

load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")
crate_universe_dependencies()

load("@rules_rust//crate_universe:defs.bzl", "crates_repository", "crate")
crates_repository(
    name = "tensat_crate_index",
    cargo_lockfile = "//src/enzyme_ad/jax:deps/tensat/Cargo.Bazel.lock",
    lockfile = "//src/enzyme_ad/jax:deps/tensat/cargo-bazel-lock.json",
    manifests = ["//src/enzyme_ad/jax:deps/tensat/Cargo.toml"],
)

load("@tensat_crate_index//:defs.bzl", "crate_repositories")
crate_repositories()

load("@xla//third_party/llvm:workspace.bzl", llvm = "repo")
llvm("llvm-raw")
load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")
llvm_configure(name = "llvm-project", targets = LLVM_TARGETS)

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

load("@jax//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()