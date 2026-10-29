"""Export to ONNX with fixed input sizes, checked against PyTorch: the depth model (M3.5) and the
student (M4.6).

The depth graph takes the normalized image [1, 3, h, w] and returns metric depth [1, h, w] at the
model resolution. Resizing and normalization stay outside, as in DepthModel.preprocess, so
depth_node (M4) can do them on the GPU next to the camera.

The student graph (stack3) takes the input frames [1, 6, 48, 64] and the raw state [1, 10], the
state normalization is part of it, and returns the action [1, 3] before clipping.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from dronerl import contract
from dronerl.agents.student import FRAMES, Student
from dronerl.envs import maps as mapgen
from dronerl.envs.nav_gpu import CONDITIONS, NavEnv
from dronerl.perception import nyu
from dronerl.perception.depth_model import MODEL_ID, DepthModel


class Wrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, image):
        return self.net(pixel_values=image).predicted_depth


class StudentGraph(nn.Module):
    def __init__(self, student):
        super().__init__()
        self.student = student

    def forward(self, frames, state):
        return self.student(frames, self.student.norm(state))[0]


def export(net, size, path):
    net = Wrapper(net).eval()
    x = torch.zeros(1, 3, *size)
    torch.onnx.export(net, (x,), path, input_names=["image"], output_names=["depth"], dynamo=True)


def export_student(student, path):
    graph = StudentGraph(student).eval()
    c = FRAMES[student.history] * contract.STUDENT_CHANNELS
    x = (torch.zeros(1, c, contract.IMG_H, contract.IMG_W), torch.zeros(1, contract.STATE_DIM))
    # small enough for one file, depth_node and policy_node load a single path
    names = {"input_names": ["frames", "state"], "output_names": ["action"]}
    torch.onnx.export(graph, x, path, **names, dynamo=True, external_data=False)


def session(path, threads=0):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def max_rel_error(sess, net, inputs):
    """Largest relative difference of the ONNX output from PyTorch over `inputs` [N, 3, h, w]."""
    net = Wrapper(net).eval()
    worst = 0.0
    for x in inputs:
        with torch.no_grad():
            ref = net(x[None]).numpy()
        out = sess.run(None, {"image": x[None].numpy()})[0]
        worst = max(worst, float(np.max(np.abs(out - ref) / np.maximum(np.abs(ref), 1e-3))))
    return worst


def observations(student, envs=50, steps=20, device="cpu"):
    """Input frames and raw states of the student flying the medium test maps of M3 with the
    sensors of the "test" condition, envs * steps of them."""
    level = mapgen.TEST_LEVELS["medium"]
    pool = mapgen.load("test", level)
    env = NavEnv(pool, envs, device, level=level, mix=0, randomize=True, student=CONDITIONS["test"])
    student = student.to(device)
    obs, memory = env.reset(), None
    start = torch.ones(envs, dtype=torch.bool, device=device)
    frames, states = [], []
    with torch.no_grad():
        for _ in range(steps):
            action, memory, x = student.step(obs, memory, start)
            frames.append(x.cpu())
            states.append(obs["student_state"].cpu())
            obs, _, terminated, truncated, _ = env.step(action.clamp(-1, 1))
            start = terminated | truncated
    return torch.cat(frames), torch.cat(states)


def max_abs_error(sess, graph, frames, states):
    worst = 0.0
    for x, s in zip(frames, states):
        with torch.no_grad():
            ref = graph(x[None], s[None]).numpy()
        out = sess.run(None, {"frames": x[None].numpy(), "state": s[None].numpy()})[0]
        worst = max(worst, float(np.abs(out - ref).max()))
    return worst


def latency(sess, feed, repeats=10):
    sess.run(None, feed)
    t = time.perf_counter()
    for _ in range(repeats):
        sess.run(None, feed)
    return 1000 * (time.perf_counter() - t) / repeats


def latencies(path, feed, repeats):
    threads = torch.get_num_threads()
    return {
        "ort_cpu_1_thread": latency(session(path, 1), feed, repeats),
        f"ort_cpu_{threads}_threads": latency(session(path, threads), feed, repeats),
    }


def students(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for ckpt in args.student:
        student = Student.load(ckpt)
        if student.history != "stack3":
            raise SystemExit(f"{ckpt}: history {student.history}, the export takes stack3")
        name = ckpt.parents[1].name
        path = args.out / f"student_{name}.onnx"
        export_student(student, path)
        frames, states = observations(student, device=device)
        graph = StudentGraph(student.cpu()).eval()
        result = {"checkpoint": str(ckpt), "file": str(path), "observations": len(frames)}
        result["max_abs_error"] = max_abs_error(session(path), graph, frames, states)
        if not args.no_latency:
            feed = {"frames": frames[:1].numpy(), "state": states[:1].numpy()}
            result["latency_ms"] = latencies(path, feed, 1000)
        (args.out / f"student_{name}.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF id or a fine-tuned model directory")
    parser.add_argument("--name", default="zero_shot")
    parser.add_argument("--res", type=int, default=518)
    parser.add_argument("--images", type=int, default=20)
    parser.add_argument("--student", type=Path, nargs="+", help="export these students instead")
    parser.add_argument("--out", type=Path, default=Path("runs/m3/onnx"))
    parser.add_argument("--no-latency", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if args.student:
        return students(args)
    model = DepthModel(args.res, "cpu", path=args.model)
    rgb = nyu.load(nyu.splits()["test"][: args.images])["rgb"]
    inputs = model.preprocess(rgb)
    path = args.out / f"depth_{args.name}_r{args.res}.onnx"
    export(model.net, inputs.shape[-2:], path)
    result = {"model": str(args.model), "input": list(inputs.shape[1:]), "file": str(path)}
    result["max_rel_error"] = max_rel_error(session(path), model.net, inputs)
    if not args.no_latency:
        result["latency_ms"] = latencies(path, {"image": inputs[:1].numpy()}, 10)
    (args.out / f"depth_{args.name}_r{args.res}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
