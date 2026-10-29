"""Safety tests of the flight stack in SITL (plan/M4.md, M4.8), inside the container with the ROS 2
workspace built (sim/safety_tests.sh). Each test flies the empty world with sim.launch.py, makes
its failure in the air and checks from the ground truth and the PX4 status how the stack and PX4
react. The flight is recorded (bag, ulog) in runs/m4/safety/<test>_<time>, the checks go to
checks.json there and to runs/m4/safety/summary.json.

    python3 sim/safety_tests.py [test ...]   all tests by default
"""

import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from dronerl_ros import common
from dronerl_ros.probe import Probe, fly
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from px4_msgs.msg import ManualControlSetpoint, OffboardControlMode, VehicleCommand, VehicleStatus
from std_msgs.msg import String

from dronerl import contract

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "m4" / "safety"
WORLD, PAIR = "empty", 0
FAIL_AFTER = 3.0  # s of the policy in the air before the failure, the drone is at speed by then
STILL = 0.1  # m/s, horizontal speed of a hover
DRIFT = 0.2  # m/s, of a hover in Position mode, the hold on the flow estimate creeps a little
OFFBOARD = VehicleStatus.NAVIGATION_STATE_OFFBOARD
POSCTL = VehicleStatus.NAVIGATION_STATE_POSCTL
LAND = VehicleStatus.NAVIGATION_STATE_AUTO_LAND
LOITER = VehicleStatus.NAVIGATION_STATE_AUTO_LOITER
HOVER_OR_LAND = (POSCTL, LOITER, LAND)
NAMES = {POSCTL: "POSCTL", LOITER: "LOITER", OFFBOARD: "OFFBOARD", LAND: "LAND"}


def check(name, value, ok=None, limit=None):
    """One line of the result; ok None is information only."""
    if isinstance(value, np.generic):
        value = value.item()
    value = round(value, 3) if isinstance(value, float) else value
    return {"name": name, "value": value, "limit": limit, "ok": None if ok is None else bool(ok)}


def since(t, t0):
    return None if t is None else t - t0


class Watch(Probe):
    """Records one flight, makes the failure of a test and judges the reaction."""

    def __init__(self):
        super().__init__(WORLD, PAIR)
        self.status, self.reasons = [], []
        self.gt, self.safe, self.cmds, self.stream = [], [], [], []
        self.result, self.yaw = None, 0.0
        self.create_subscription(String, "/dronerl/result", self.on_result, common.LATCHED)
        self.create_subscription(VehicleStatus, common.STATUS, self.on_status, common.PX4_OUT)
        self.create_subscription(Odometry, "/ground_truth/odometry", self.on_odometry, 50)
        self.create_subscription(TwistStamped, "/dronerl/cmd", self.on_cmd, 10)
        self.create_subscription(TwistStamped, "/dronerl/safe_cmd", self.on_safe, 10)
        self.create_subscription(String, "/dronerl/safety", self.on_safety, 10)
        self.create_subscription(
            OffboardControlMode, "/fmu/in/offboard_control_mode", self.on_stream, common.PX4_OUT
        )

    def on_result(self, msg):
        self.result = json.loads(msg.data)

    def on_status(self, msg):
        s = (msg.nav_state, msg.arming_state, msg.failsafe)
        if not self.status or self.status[-1][1:] != s:
            self.status.append((self.now(), *s))

    def on_odometry(self, msg):
        t = common.seconds(msg.header.stamp)
        if self.gt and t - self.gt[-1][0] < 0.02:
            return
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
        self.gt.append((t, p.x, p.y, p.z))

    def on_cmd(self, msg):
        self.cmds.append(self.now())

    def on_safe(self, msg):
        self.safe.append((self.now(), *common.command(msg)))

    def on_safety(self, msg):
        self.reasons.append((self.now(), msg.data))

    def on_stream(self, msg):
        self.stream.append(self.now())

    def kill(self, *nodes):
        for name in nodes:
            subprocess.run(["pkill", "-9", "-f", f"lib/dronerl_ros/{name}"], check=False)
        self.events["kill"] = self.now()
        self.get_logger().info(f"killed {', '.join(nodes)}")

    def due(self):
        """True once the policy has flown FAIL_AFTER seconds."""
        return self.phase == "fly" and self.now() - self.events["fly"] >= FAIL_AFTER

    # the record as arrays for the checks

    def track(self):
        """Ground truth (t, x, y, z) and the horizontal velocity over 0.1 s."""
        gt = np.array(self.gt)
        return gt[5:], (gt[5:, 1:3] - gt[:-5, 1:3]) / (gt[5:, :1] - gt[:-5, :1])

    def still_at(self, t0):
        """First time after t0 the drone hovers, and the way it made from t0 till then."""
        gt, vel = self.track()
        after = np.flatnonzero((gt[:, 0] >= t0) & (np.linalg.norm(vel, axis=1) < STILL))
        if not len(after):
            return None, None
        t1 = gt[after[0], 0]
        return t1, self.way(t0, t1)

    def speed_at(self, t):
        gt, vel = self.track()
        return float(np.linalg.norm(vel[min(np.searchsorted(gt[:, 0], t), len(vel) - 1)]))

    def way(self, t0, t1):
        gt = np.array(self.gt)
        i0, i1 = np.searchsorted(gt[:, 0], [t0, t1])
        return float(np.hypot(*(gt[min(i1, len(gt) - 1), 1:3] - gt[i0, 1:3])))

    def top_speed(self, t0, t1, along=None):
        """Highest horizontal speed between t0 and t1, or along the direction along."""
        gt, vel = self.track()
        speed = np.linalg.norm(vel, axis=1) if along is None else vel @ along
        window = (gt[:, 0] > t0) & (gt[:, 0] < t1)
        return float(speed[window].max()) if window.any() else None

    def modes(self, t0):
        """Navigation states after t0, with their start relative to t0, as a check line."""
        seq = [f"{NAMES.get(s[1], s[1])} {s[0] - t0:+.1f}" for s in self.status if s[0] >= t0]
        return check("navigation states [s]", ", ".join(dict.fromkeys(seq)))

    def first_status(self, after, test):
        return next((s for s in self.status if s[0] >= after and test(s)), None)

    def landed(self):
        states = [s[2] for s in self.status]
        ok = VehicleStatus.ARMING_STATE_ARMED in states
        ok = ok and states[-1] == VehicleStatus.ARMING_STATE_DISARMED
        return check("landed and disarmed", ok, ok)

    def clearance(self):
        clear = self.result["min_clearance"] if self.result else None
        ok = clear is not None and clear >= 0
        return check("min clearance beyond COLLISION_DIST [m]", clear, ok, ">= 0")

    def checks(self):
        return []


class PolicyKill(Watch):
    """1. policy_node dies: the watchdog of safety_node gives a zero command within 0.5 s, then
    mission_node lands."""

    nodes = ("policy_node",)

    def tick(self):
        if "kill" not in self.events and self.due():
            self.kill(*self.nodes)

    def checks(self):
        kill = self.events.get("kill")
        if kill is None:
            return [check("failure in the air", False, False)]
        last = max(t for t in self.cmds if t <= kill)
        out = [check("speed at the failure [m/s]", self.speed_at(kill)), *self.reaction(last)]
        still, way = self.still_at(last)
        out.append(check("last command -> hover [s]", since(still, last)))
        out.append(check("way after the last command [m]", way))
        return out + [self.landed(), self.clearance()]

    def reaction(self, last):
        zero = next((s[0] for s in self.safe if s[0] > last and not any(s[1:])), None)
        delay = since(zero, last)
        ok = delay is not None and delay <= 0.5
        land = next((t for t, p in self.phases if t > last and p == "land"), None)
        return [
            check("last command -> zero command [s]", delay, ok, "<= 0.5"),
            check("last command -> landing [s]", since(land, last), land is not None),
        ]


class OffboardLoss(PolicyKill):
    """2. policy_node and safety_node die: px4_bridge stops the stream and PX4 goes to its offboard
    loss failsafe, a hover or a landing. mission_node waits longer than PX4 here, else its own
    landing would come first."""

    nodes = ("policy_node", "safety_node")
    launch = {"params": "mission_slow.yaml"}

    def reaction(self, last):
        end = max(self.stream)
        fs = self.first_status(last, lambda s: s[1] != OFFBOARD)
        ok = fs is not None and fs[1] in HOVER_OR_LAND
        out = [
            check("last command -> end of the offboard stream [s]", end - last),
            check("navigation state after the loss", fs and fs[1], ok, "POSCTL, LOITER or LAND"),
        ]
        if fs:
            delay = fs[0] - end
            out.append(check("end of the stream -> failsafe [s]", delay, limit="COM_OF_LOSS_T"))
            out.append(check("way until the failsafe [m]", self.way(last, fs[0])))
        return out + [self.modes(last)]


class Fence(Watch):
    """3a. A command out of the arena: the soft geofence of safety_node cuts it, the drone stops
    inside. Without the policy, the probe yaws towards the farthest edge and flies at V_FWD_MAX."""

    launch = {"policy": "false"}
    PUSH = 4.0  # s of the outward command once the fence acts

    def __init__(self):
        super().__init__()
        start, size = self.map["start"], self.map["size"]
        # the axis with the most room, a straight run to the edge
        room = [size[0] - start[0], size[1] - start[1], start[0], start[1]]
        k = int(np.argmax(room))
        self.heading, self.room = k * math.pi / 2, room[k]
        self.dir = np.round([math.cos(self.heading), math.sin(self.heading)])

    def pushing(self):
        if self.phase != "fly":
            return False
        fence = next((t for t, r in self.reasons if r == "fence"), None)
        return fence is None or self.now() - fence < self.PUSH

    def tick(self):
        if not self.pushing():
            return
        err = math.remainder(self.heading - self.yaw, math.tau)
        yaw_rate = float(np.clip(2 * err, -contract.YAW_RATE_MAX, contract.YAW_RATE_MAX))
        fwd = contract.V_FWD_MAX if abs(err) < 0.1 else 0.0
        self.send((fwd, 0.0, yaw_rate))

    def way_out(self):
        """Largest run from the start towards the edge [m] and the room left to the edge."""
        gt = np.array(self.gt)
        run = float(((gt[:, 1:3] - self.map["start"]) @ self.dir).max())
        return run, self.room - run

    def run_at(self, t):
        gt = np.array(self.gt)
        i = min(np.searchsorted(gt[:, 0], t), len(gt) - 1)
        return float((gt[i, 1:3] - self.map["start"]) @ self.dir)

    def checks(self):
        fence = next((t for t, r in self.reasons if r == "fence"), None)
        out = [check("fence acted", fence is not None, fence is not None)]
        run, left = self.way_out()
        out.append(check("room to the edge at the start [m]", self.room))
        out.append(self.edge(left))
        if fence is not None:
            out.append(check("start of the fence -> edge [m]", self.room - self.run_at(fence)))
            # the command along the fence stays, an error of the estimated heading gives some
            top = self.top_speed(fence + 2.0, fence + self.PUSH, self.dir)
            ok = top is not None and top < STILL
            out.append(check("outward speed while pushed, from 2 s [m/s]", top, ok, f"< {STILL}"))
        return out + [self.landed(), self.clearance()]

    def edge(self, left):
        limit = contract.COLLISION_DIST
        return check("closest to the edge [m]", left, left >= limit, f">= {limit}")


class HardGeofence(Fence):
    """3b. The same command without the soft geofence: the cylinder of PX4 (GF_MAX_HOR_DIST) around
    home ends offboard with a landing (GF_ACTION)."""

    RADIUS = 3.0
    launch = {"policy": "false", "fence": "false", "geofence": str(RADIUS)}

    def pushing(self):
        return self.phase == "fly"

    def checks(self):
        fly = self.events.get("fly", math.inf)
        fs = self.first_status(fly, lambda s: s[1] != OFFBOARD)
        ok = fs is not None and fs[1] in (LOITER, LAND)
        land = fs and self.first_status(fs[0], lambda s: s[1] == LAND)
        after = land and land[0] - fs[0]
        out = [
            check("navigation state at the breach", fs and fs[1], ok, "LOITER or LAND"),
            check("landing after the breach [s]", after, bool(land), "GF_ACTION"),
            self.modes(fly),
        ]
        if fs:
            gt = np.array(self.gt)
            i = min(np.searchsorted(gt[:, 0], fs[0]), len(gt) - 1)
            dist = float(np.hypot(*(gt[i, 1:3] - self.map["start"])))
            ok, limit = dist < self.RADIUS + 0.5, f"radius {self.RADIUS}"
            out.append(check("distance from home at the breach [m]", dist, ok, limit))
        run, left = self.way_out()
        out.append(check("largest distance from home [m]", run))
        return out + [self.edge(left), self.landed(), self.clearance()]


class Pilot(Watch):
    """4. The pilot takes over: a switch to Position mode stops the policy and mission_node does
    not go back to offboard. The probe is the RC of the pilot: centred sticks all the time, the
    mode switch as a VehicleCommand, after HOLD seconds the landing."""

    HOLD = 5.0

    def __init__(self):
        super().__init__()
        self.rc_pub = self.create_publisher(
            ManualControlSetpoint, "/fmu/in/manual_control_input", common.PX4_IN
        )

    def tick(self):
        t = common.px4_time(self)
        rc = ManualControlSetpoint(timestamp=t, timestamp_sample=t, valid=True)
        rc.data_source = ManualControlSetpoint.SOURCE_MAVLINK_0  # COM_RC_IN_MODE 1 in SITL
        self.rc_pub.publish(rc)
        if "switch" not in self.events and self.due():
            self.vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1, 3)  # Position
            self.events["switch"] = self.now()
        elif "switch" in self.events and "land" not in self.events:
            if self.now() - self.events["switch"] >= self.HOLD:
                self.vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
                self.events["land"] = self.now()

    def checks(self):
        switch = self.events.get("switch")
        if switch is None:
            return [check("mode switch in the air", False, False)]
        out = [check("speed at the switch [m/s]", self.speed_at(switch))]
        pos = self.first_status(switch, lambda s: s[1] == POSCTL)
        delay = None if pos is None else pos[0] - switch
        ok = delay is not None and delay <= 0.5
        out.append(check("switch -> Position [s]", delay, ok, "<= 0.5"))
        land = self.events.get("land", math.inf)
        back = [s for s in self.status if switch < s[0] < land and s[1] == OFFBOARD]
        out.append(check("back to offboard before the landing", bool(back), not back))
        pilot = any(t > switch and p == "pilot" for t, p in self.phases)
        out.append(check("mission_node in the phase pilot", pilot, pilot))
        still, way = self.still_at(switch)
        out.append(check("switch -> hover [s]", since(still, switch)))
        out.append(check("way after the switch [m]", way))
        top = self.top_speed(switch + 2.0, land)
        ok = top is not None and top < DRIFT
        out.append(check("speed in Position, from 2 s [m/s]", top, ok, f"< {DRIFT}"))
        out.append(check("way in Position, from 2 s [m]", self.way(switch + 2.0, land)))
        return out + [self.landed(), self.clearance()]


TESTS = {
    "policy_kill": PolicyKill,
    "offboard_loss": OffboardLoss,
    "fence": Fence,
    "geofence": HardGeofence,
    "pilot": Pilot,
}


def run(name):
    out = OUT / f"{name}_{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True)
    test = TESTS[name]
    launch = dict(test.launch)
    if "params" in launch:
        # mission_node lands only well after the offboard loss failsafe of PX4
        params = {"mission_node": {"ros__parameters": {"policy_timeout": 10.0}}}
        (out / launch["params"]).write_text(yaml.safe_dump(params))
        launch["params"] = str(out / launch["params"])
    probe = test()
    fly(probe, out, {"world": WORLD, "pair": PAIR, "gui": "false", **launch})
    checks = probe.checks()
    passed = all(c["ok"] is not False for c in checks)
    report = {"test": name, "out": str(out.relative_to(ROOT)), "passed": passed, "checks": checks}
    report["events"] = probe.events
    report["status"] = probe.status
    report["safety"] = probe.reasons
    (out / "checks.json").write_text(json.dumps(report, indent=2))
    probe.destroy_node()
    print(f"\n{name}: {'PASS' if passed else 'FAIL'} ({report['out']})")
    for c in checks:
        mark = {True: "ok  ", False: "FAIL", None: "    "}[c["ok"]]
        limit = f" ({c['limit']})" if c["limit"] else ""
        print(f"  {mark} {c['name']}: {c['value']}{limit}")
    return report


def main():
    names = sys.argv[1:] or list(TESTS)
    rclpy.init()
    reports = [run(name) for name in names]
    rclpy.shutdown()
    path = OUT / "summary.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    summary.update({r["test"]: {"passed": r["passed"], "out": r["out"]} for r in reports})
    path.write_text(json.dumps(summary, indent=2))
    sys.exit(0 if all(r["passed"] for r in reports) else 1)


if __name__ == "__main__":
    main()
