import asyncio
import argparse


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


async def run_command(cmd=None, name=None, dry=False):
    if dry:
        print(f"[dry] {cmd}")
        return
    process = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.gather(
        handle_stream(process.stdout, lambda x: split_and_print(f"[{name}]", x)),
        handle_stream(process.stderr, lambda x: split_and_print(f"[{name}]", x)),
    )
    return await process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry", action="store_true", required=False, default=False)
    args = parser.parse_args()
    loop = asyncio.get_event_loop()
    cmd = " ".join(["echo hello && sleep 2 && date",])
    tasks = [
        run_command(cmd, "A", args.dry),
        run_command(cmd, "B", args.dry),
        run_command(cmd, "C", args.dry),
    ]
    loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
