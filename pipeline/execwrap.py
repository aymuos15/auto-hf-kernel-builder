from __future__ import annotations

import shlex
import subprocess
import sys


def _ssh(cfg):
    return (cfg.get("exec", {}).get("ssh") or "").strip()


def _remote(cmd, cfg, surface):
    e = cfg.get("exec", {}) or {}
    if surface == "container" and (e.get("container") or "").strip():
        return f"docker run --rm -t {e.get('docker_run_extra','')} -v {e['remote_workdir']}:{e['workdir']} -w {e['workdir']} {e['container'].strip()} bash -lc {shlex.quote(cmd)}"
    if surface == "host":
        return f"bash -lc {shlex.quote(cmd)}"
    return cmd


def wrap_exec(cmd, cfg, surface="container"):
    ssh = _ssh(cfg)
    r = _remote(cmd, cfg, surface)
    return f"{ssh} {shlex.quote(r)}".strip() if ssh else r


def run(cmd, cfg, surface="container", timeout=1800, stream=False, tee=False):
    ssh = _ssh(cfg)
    remote = _remote(cmd, cfg, surface)
    if ssh:
        parts = shlex.split(ssh)
        target, shell = [parts[0]] + (["-tt"] if (stream or tee) else []) + parts[1:] + [remote], False
    else:
        target, shell = remote, True
    if tee:
        p = subprocess.Popen(target, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        buf = []
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            buf.append(line)
        p.wait(timeout=timeout)
        return p.returncode, "".join(buf), "".join(buf)
    if stream:
        p = subprocess.run(target, shell=shell, timeout=timeout)
        return p.returncode, "", ""
    p = subprocess.run(target, shell=shell, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def rsync(src, dst, excludes=(), timeout=600):
    args = ["rsync", "-az", "--delete", "--mkpath", "-e", "ssh"]
    for e in excludes:
        args += ["--exclude", e]
    args += [src, dst]
    subprocess.run(args, check=True, capture_output=True, text=True, timeout=timeout)


def remote_repo(cfg):
    return cfg.get("exec", {}).get("remote_workdir", "$HOME/agentic-kernels")


def rsync_dest(cfg):
    p = remote_repo(cfg)
    for pre in ("$HOME/", "${HOME}/", "~/"):
        if p.startswith(pre):
            return p[len(pre):]
    return p.lstrip("/")
