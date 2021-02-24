import asyncio
import argparse
import pathlib


def split_and_print(prefix, text):
    lines = text.decode().splitlines(keepends=True)
    prefixed = ""
    for l in lines:
        prefixed += f"{prefix} {l.strip()}"
    if l.strip():
        print(prefixed, flush=True)


async def handle_stream(stream, cb):
    while True:
        line = await stream.readline()
        if line:
            cb(line)
        else:
            break


async def run_command(name=None, cmd=None, dry=False):
    if dry:
        print(f"[dry] {cmd}")
        return 0
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.gather(
        handle_stream(process.stdout, lambda x: split_and_print(f"[{name}]", x)),
        handle_stream(process.stderr, lambda x: split_and_print(f"[{name}]", x)),
    )
    return await process.wait()


def check_result(result):
    log_txt = "some tasks failed, please examine the log"
    if isinstance(result, list):
        assert sum(result) == 0, log_txt
    else:
        assert result == 0, log_txt


async def pipe(cmds=[], dry=False):
    result = []
    for (name, cmd) in cmds:
        result.append(await run_command(name, cmd, args.dry))
    check_result(result)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", required=False, default=False)
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    cmd = "export hello=ci && echo hello-${hello} && sleep 2 && date"
    thid_dir = pathlib.Path(__file__).parent.absolute()

    cuda_build = pipe(
        [
            (
                "cuda:build",
                f"export ONEFLOW_PKG_TYPE=cuda && bash {thid_dir}/common/build.sh",
            ),
            ("cuda:upload", cmd),
        ],
        args.dry,
    )
    check_result(loop.run_until_complete(cuda_build))

    cuda_test = run_command("cuda:test", cmd, args.dry)
    xla_build = run_command("xla:build", f"export ONEFLOW_PKG_TYPE=xla && bash {thid_dir}/common/build.sh", args.dry)
    check_result(loop.run_until_complete(asyncio.gather(cuda_test, xla_build)))

    xla_test = pipe([("xla:test", cmd),])
    cpu_build = run_command("cpu:build", "ls", args.dry)

    check_result(loop.run_until_complete(asyncio.gather(xla_test, cpu_build)))

    cpu_test = pipe([("cpu:test", cmd),])
    check_result(loop.run_until_complete(cpu_test))
    loop.close()
