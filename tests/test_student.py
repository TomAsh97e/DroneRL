import json

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from dronerl import contract
from dronerl.agents import ppo_gpu
from dronerl.agents.student import Policy, Student, from_uint8, to_uint8
from dronerl.envs import maps as mapgen
from dronerl.eval import nav_eval
from dronerl.train import dagger


def fake_obs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    depth = torch.rand(n, 1, contract.IMG_H, contract.IMG_W, generator=g)
    valid = (torch.rand(n, 1, contract.IMG_H, contract.IMG_W, generator=g) > 0.1).float()
    state = torch.randn(n, contract.STATE_DIM, generator=g)
    return {"student_depth": torch.cat([depth * valid, valid], 1), "student_state": state}


def test_beta_schedule():
    assert [dagger.beta_schedule(0.5, i) for i in range(4)] == [1, 0.5, 0.25, 0.125]


def test_buffer_is_fifo():
    buf = dagger.Buffer(5, {"x": ((2,), torch.float32)})
    buf.add({"x": torch.arange(6.0).view(3, 2)})
    buf.add({"x": torch.arange(6.0, 14).view(4, 2)})
    assert buf.size == 5 and buf.pos == 2
    # the two oldest items were overwritten
    assert sorted(buf.data["x"][:, 0].tolist()) == [4, 6, 8, 10, 12]
    sample = buf.sample(100, torch.Generator().manual_seed(0), "cpu")["x"]
    assert set(sample[:, 0].tolist()) == {4, 6, 8, 10, 12}


def test_uint8_keeps_mask_and_depth():
    image = fake_obs(4)["student_depth"]
    back = from_uint8(to_uint8(image))
    torch.testing.assert_close(back[:, 1], image[:, 1])
    assert (back - image).abs().max() <= 0.5 / 254 + 1e-6


@pytest.mark.parametrize("history", ["none", "stack3", "gru"])
def test_student_history(history, tmp_path):
    student = Student(history)
    obs = [fake_obs(3, seed) for seed in range(3)]
    action, memory, frames = student.step(obs[0], update=True)
    assert action.shape == (3, contract.ACT_DIM)
    assert frames.shape[1] == (6 if history == "stack3" else 2)
    action, memory, frames = student.step(obs[1], memory, torch.tensor([False, True, False]))
    if history == "stack3":
        first, second = obs[0]["student_depth"][0], obs[1]["student_depth"][0]
        torch.testing.assert_close(frames[0], torch.cat([first, first, second]))
        # a new episode starts from its own frame only
        torch.testing.assert_close(frames[1], obs[1]["student_depth"][1].repeat(3, 1, 1))
    if history == "gru":
        fresh = student.step({k: v[1:2] for k, v in obs[1].items()})[0]
        torch.testing.assert_close(action[1:2], fresh)
    student.save(tmp_path / "s.pt", 5)
    policy, loaded = Policy(student), Policy(Student.load(tmp_path / "s.pt"))
    for o in obs:
        torch.testing.assert_close(loaded(o), policy(o))


def test_short_dagger_on_cpu(tmp_path, monkeypatch):
    load = mapgen.load

    def small(split, level):
        pool = load(split, level)
        return mapgen.collate([mapgen.get_map(pool, i) for i in range(4)])

    monkeypatch.setattr(mapgen, "load", small)
    teacher = ppo_gpu.Agent()
    with torch.no_grad():
        teacher.actor.mlp[-1].weight.mul_(100)  # labels well away from zero
    teacher.save(tmp_path / "teacher.pt", 0)
    cfg = dagger.Config(
        device="cpu",
        log_dir=str(tmp_path),
        teacher=str(tmp_path / "teacher.pt"),
        total_steps=16 * 8 * 8,
        num_envs=16,
        num_steps=8,
        batch_size=64,
        updates=16,
        buffer_size=512,
        learning_rate=1e-3,
        level=0,
        mix=0.0,
        eval_every=512,
        eval_levels={"empty": 0},
    )
    run_dir = dagger.train(cfg)
    final = json.loads((run_dir / "final.json").read_text())
    assert 0 <= final["success_rate"] <= 1 and "teacher_sr_empty" in final
    assert (run_dir / "eval.csv").exists()
    tb = EventAccumulator(str(run_dir / "tb"))
    tb.Reload()
    loss = [e.value for e in tb.Scalars("train/loss")]
    assert loss[-1] < 0.5 * loss[0]
    # the checkpoint loads in nav_eval
    result = nav_eval.evaluate(str(run_dir), "val", {"empty": 0}, "cpu", condition="test")
    assert result["condition"] == "test" and result["levels"]["empty"]["off_fov"] is not None
