JAX_COMMIT = "493698e6e053641aa8c51bca657cbd763a3ced19"
JAX_SHA256 = "f8bbcc40cdee9d8d83a7f6e197ce111f1c01ee00341eab83ddd9367e48519665"

ENZYME_COMMIT = "d5eac0fc9b2f0a4054f7bcd815cc5698661ae112"
ENZYME_SHA256 = "7ac6047d15358434ec77833ebfd96704f5784dea0c79aa2408d2b4bc08183777"

PYRULES_COMMIT = "fe33a4582c37499f3caeb49a07a78fc7948a8949"
PYRULES_SHA256 = "cfa6957832ae0e0c7ee2ccf455a888a291e8419ed8faf45f4420dd7414d5dd96"

XLA_PATCHES = [
    """
    sed -i.bak0 "s/\\/\\/third_party:repo.bzl/@bazel_tools\\/\\/tools\\/build_defs\\/repo:http.bzl/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_file/patch_args = [\\\"-p1\\\"],patches/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "/link_file/d" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/build_file.*/build_file_content = \\\"# empty\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/\\/\\/third_party/@xla\\/\\/third_party/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/tf_http_archive/http_archive/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/strip_prefix/patch_cmds = [\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_BACKTRACE=1\\/NO_HAVE_BACKTRACE=0\\/g' {} +\\\"], strip_prefix/g" third_party/llvm/workspace.bzl
    """,
    "find . -type f -name BUILD -exec sed -i.bak1 's/\\/\\/third_party\\/py\\/enzyme_ad\\/\\.\\.\\./public/g' {} +", 
    "find . -type f -name BUILD -exec sed -i.bak2 's/\\/\\/xla\\/mlir\\/memref:friends/\\/\\/visibility:public/g' {} +",
    "find xla/mlir -type f -name BUILD -exec sed -i.bak3 's/\\/\\/xla:internal/\\/\\/\\/\\/visibility:public/g' {} +",
    """
    sed -i.bak0 "s/@xla\\/\\/xla\\/tsl:linux_x86_64/@bazel_tools\\/\\/src\\/conditions:linux/g" xla/tsl/mkl/build_defs.bzl
    """,
    """
    sed -i.bak0 "s/@xla\\/\\/xla\\/tsl:windows/@bazel_tools\\/\\/src\\/conditions:windows/g" xla/tsl/mkl/build_defs.bzl
    """,
    """
    sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_LINK_H=1\\/HAVE_LINK_H=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/LLVM_ENABLE_THREADS=1\\/LLVM_ENABLE_THREADS=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_MALLINFO=1\\/DONT_HAVE_ANY_MALLINFO=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_PTHREAD_GETNAME_NP=1\\/HAVE_PTHREAD_GETNAME_NP=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.bzl -exec sed -i.bak0 's\\/HAVE_PTHREAD_SETNAME_NP=1\\/HAVE_PTHREAD_SETNAME_NP=0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/ENABLE_CRASH_OVERRIDES 1\\/ENABLE_CRASH_OVERRIDES 0\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/HAVE_PTHREAD_GETNAME_NP\\/FAKE_HAVE_PTHREAD_GETNAME_NP\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
    """,
    """
    sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\[\\\"find . -type f -name config.h -exec sed -i.bak0 's\\/HAVE_PTHREAD_SETNAME_NP\\/FAKE_HAVE_PTHREAD_SETNAME_NP\\/g' {} +\\\",/g" third_party/llvm/workspace.bzl
    """,
    # """
    # sed -i.bak0 "s/patch_cmds = \\[/patch_cmds = \\['find . -type f -name BUILD.bazel -exec sed -i.bak0 \\\\\\'s\\/\\\"CAPIIR\\\",\\/\\\"CAPIIR\\\",alwayslink=1,\\/g\\\\\\\\' {} +',/g" third_party/llvm/workspace.bzl
    # """,
]
