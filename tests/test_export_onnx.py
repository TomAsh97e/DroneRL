import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from dronerl import contract
from dronerl.agents.student import Student

spec = importlib.util.spec_from_file_location(
    "export_onnx", Path(__file__).parents[1] / "deploy" / "export_onnx.py"
)
export_onnx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_onnx)


class TinyDepth(nn.Module):
    """Same interface as the transformers depth model."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, pixel_values):
        return SimpleNamespace(
            predicted_depth=nn.functional.softplus(self.conv(pixel_values))[:, 0]
        )


def test_export_matches_torch(tmp_path):
    torch.manual_seed(0)
    net = TinyDepth()
    path = tmp_path / "tiny.onnx"
    export_onnx.export(net, (28, 42), path)
    inputs = torch.randn(3, 3, 28, 42)
    assert export_onnx.max_rel_error(export_onnx.session(path), net, inputs) < 1e-4
    feed = {"image": inputs[:1].numpy()}
    assert export_onnx.latency(export_onnx.session(path, 1), feed, repeats=2) > 0


def test_student_export_matches_torch(tmp_path):
    torch.manual_seed(0)
    student = Student("stack3")
    student.state_rms.update(torch.randn(100, contract.STATE_DIM) * 3 + 1)
    path = tmp_path / "student.onnx"
    export_onnx.export_student(student, path)
    frames, states = export_onnx.observations(student, envs=4, steps=3)
    assert frames.shape == (12, 6, contract.IMG_H, contract.IMG_W)
    assert states.shape == (12, contract.STATE_DIM)
    graph = export_onnx.StudentGraph(student).eval()
    assert export_onnx.max_abs_error(export_onnx.session(path), graph, frames, states) < 1e-5
    with torch.no_grad():
        action = graph(frames[:2], states[:2])
    torch.testing.assert_close(action, student(frames[:2], student.norm(states[:2]))[0])
