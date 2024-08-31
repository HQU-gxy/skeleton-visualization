import numpy as np
import re
from transforms3d.euler import euler2mat, mat2euler
from typing import Optional
from os import PathLike


class BvhChannel:
    _index: int = 0
    _name: str = ""
    _joint: "BvhJoint"

    def __init__(self, index: int, name: str, joint: "BvhJoint"):
        self._index = index
        self._name = name
        self._joint = joint

    @property
    def index(self):
        return self._index

    @property
    def name(self):
        return self._name

    @property
    def joint(self):
        return self._joint

    def __repr__(self) -> str:
        return f"{self._name}/{self.index}@{self.joint.name}"

    @property
    def is_position(self):
        return self._name.endswith("position")

    @property
    def is_rotation(self):
        return self._name.endswith("rotation")


class BvhJoint:
    name: str = ""
    parent: Optional["BvhJoint"] = None
    offset: np.ndarray = np.zeros(3)
    channels: list[BvhChannel] = []
    children: list["BvhJoint"] = []

    def __init__(self, name: str, parent: Optional["BvhJoint"]):
        self.name = name
        self.parent = parent
        self.offset = np.zeros(3)
        self.channels = []
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def __repr__(self):
        return self.name

    def position_animated(self):
        return any([x.is_position for x in self.channels])

    def rotation_animated(self):
        return any([x.is_rotation for x in self.channels])


class Bvh:
    joints: dict[str, BvhJoint] = {}
    root: Optional[BvhJoint] = None
    _keyframes: Optional[np.ndarray] = None
    frames: int = 0
    fps: int = 0

    def __init__(self):
        self.joints = {}
        self.root = None
        self._keyframes = None
        self.frames = 0
        self.fps = 0

    @property
    def keyframes(self):
        if self._keyframes is None:
            raise ValueError("No keyframes loaded")
        return self._keyframes

    def _parse_hierarchy(self, text):
        lines = re.split("\\s*\\n+\\s*", text)

        joint_stack: list[BvhJoint] = []

        channel_count = 0
        for line in lines:
            words = re.split("\\s+", line)
            instruction = words[0]

            if instruction == "JOINT" or instruction == "ROOT":
                parent = joint_stack[-1] if instruction == "JOINT" else None
                joint = BvhJoint(words[1], parent)
                self.joints[joint.name] = joint
                if parent:
                    parent.add_child(joint)
                joint_stack.append(joint)
                if instruction == "ROOT":
                    self.root = joint
            elif instruction == "CHANNELS":
                for i in range(2, len(words)):
                    name = words[i]
                    joint_stack[-1].channels.append(
                        BvhChannel(channel_count, name, joint_stack[-1])
                    )
                    channel_count += 1
            elif instruction == "OFFSET":
                for i in range(1, len(words)):
                    joint_stack[-1].offset[i - 1] = float(words[i])
            elif instruction == "End":
                joint = BvhJoint(joint_stack[-1].name + "_end", joint_stack[-1])
                joint_stack[-1].add_child(joint)
                joint_stack.append(joint)
                self.joints[joint.name] = joint
            elif instruction == "}":
                joint_stack.pop()

    def _add_pose_recursive(self, joint: BvhJoint, offset, poses):
        pose = joint.offset + offset
        poses.append(pose)

        for c in joint.children:
            self._add_pose_recursive(c, pose, poses)

    def plot_hierarchy(self, fig=None):
        """
        draw the skeleton in T-pose
        """
        import matplotlib.pyplot as plt

        poses = []
        assert self.root is not None
        self._add_pose_recursive(self.root, np.zeros(3), poses)
        pos = np.array(poses)

        if fig is None:
            fig = plt.figure()
        assert type(fig) == type(
            plt.figure()
        ), f"Expected matplotlib figure, got {type(fig)}"
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(pos[:, 0], pos[:, 2], pos[:, 1])
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-30, 30)  # type: ignore
        return fig

    def parse_motion(self, text: str):
        lines = re.split("\\s*\\n+\\s*", text)

        frame = 0
        for line in lines:
            if line == "":
                continue
            words = re.split("\\s+", line)

            if line.startswith("Frame Time:"):
                self.fps = round(1 / float(words[2]))
                continue
            if line.startswith("Frames:"):
                self.frames = int(words[1])
                continue

            if self._keyframes is None:
                self._keyframes = np.empty((self.frames, len(words)), dtype=np.float32)

            for angle_index in range(len(words)):
                if words[angle_index] == "":
                    continue
                self.keyframes[frame, angle_index] = float(words[angle_index])

            frame += 1

    def parse_string(self, text: str):
        hierarchy, motion = text.split("MOTION")
        self._parse_hierarchy(hierarchy)
        self.parse_motion(motion)

    def _extract_rotation(self, frame_pose, index_offset: int, joint: BvhJoint):
        local_rotation = np.zeros(3)
        for channel in joint.channels:
            if channel.is_position:
                continue
            if channel.name == "Xrotation":
                local_rotation[0] = frame_pose[index_offset]
            elif channel.name == "Yrotation":
                local_rotation[1] = frame_pose[index_offset]
            elif channel.name == "Zrotation":
                local_rotation[2] = frame_pose[index_offset]
            else:
                raise ValueError(f"Unknown channel {channel}")
            index_offset += 1

        local_rotation = np.deg2rad(local_rotation)
        M_rotation = np.eye(3)
        for channel in joint.channels:
            if channel.is_position:
                continue

            if channel.name == "Xrotation":
                euler_rot = np.array([local_rotation[0], 0.0, 0.0])
            elif channel.name == "Yrotation":
                euler_rot = np.array([0.0, local_rotation[1], 0.0])
            elif channel.name == "Zrotation":
                euler_rot = np.array([0.0, 0.0, local_rotation[2]])
            else:
                raise ValueError(f"Unknown channel {channel}")

            M_channel = euler2mat(*euler_rot)
            M_rotation = M_rotation.dot(M_channel)

        return M_rotation, index_offset

    def _extract_position(self, joint: BvhJoint, frame_pose, index_offset):
        offset_position = np.zeros(3)
        for channel in joint.channels:
            if channel.is_rotation:
                continue
            if channel.name == "Xposition":
                offset_position[0] = frame_pose[index_offset]
            elif channel.name == "Yposition":
                offset_position[1] = frame_pose[index_offset]
            elif channel.name == "Zposition":
                offset_position[2] = frame_pose[index_offset]
            else:
                raise ValueError(f"Unknown channel {channel}")
            index_offset += 1

        return offset_position, index_offset

    def _recursive_apply_frame(
        self, joint: BvhJoint, frame_pose, index_offset, p, r, M_parent, p_parent
    ):
        if joint.position_animated():
            offset_position, index_offset = self._extract_position(
                joint, frame_pose, index_offset
            )
        else:
            offset_position = np.zeros(3)

        if len(joint.channels) == 0:
            joint_index = list(self.joints.values()).index(joint)
            p[joint_index] = p_parent + M_parent.dot(joint.offset)
            r[joint_index] = mat2euler(M_parent)
            return index_offset

        if joint.rotation_animated():
            M_rotation, index_offset = self._extract_rotation(
                frame_pose, index_offset, joint
            )
        else:
            M_rotation = np.eye(3)

        M = M_parent.dot(M_rotation)
        position = p_parent + M_parent.dot(joint.offset) + offset_position

        rotation = np.rad2deg(mat2euler(M))
        joint_index = list(self.joints.values()).index(joint)
        p[joint_index] = position
        r[joint_index] = rotation

        for c in joint.children:
            index_offset = self._recursive_apply_frame(
                c, frame_pose, index_offset, p, r, M, position
            )

        return index_offset

    def frame_pose(self, frame: int):
        p = np.empty((len(self.joints), 3))
        r = np.empty((len(self.joints), 3))
        frame_pose = self.keyframes[frame]
        M_parent = np.zeros((3, 3))
        M_parent[0, 0] = 1
        M_parent[1, 1] = 1
        M_parent[2, 2] = 1
        assert self.root is not None
        self._recursive_apply_frame(
            self.root, frame_pose, 0, p, r, M_parent, np.zeros(3)
        )

        return p, r

    @property
    def all_frame_poses(self):
        p = np.empty((self.frames, len(self.joints), 3))
        r = np.empty((self.frames, len(self.joints), 3))

        for frame in range(len(self.keyframes)):
            p[frame], r[frame] = self.frame_pose(frame)

        return p, r

    def _plot_pose(self, p, r, fig=None):
        import matplotlib.pyplot as plt

        if fig is None:
            fig = plt.figure()
        assert type(fig) == type(
            plt.figure()
        ), f"Expected matplotlib figure, got {type(fig)}"
        ax = fig.add_subplot(111, projection="3d")

        ax.cla()
        ax.scatter(p[:, 0], p[:, 2], p[:, 1])
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(-1, 120)  # type: ignore
        return fig

    def plot_frame(self, frame: int, fig=None):
        p, r = self.frame_pose(frame)
        return self._plot_pose(p, r, fig)

    @property
    def joint_names(self):
        return self.joints.keys()

    @property
    def channels(self):
        return [c for j in self.joints.values() for c in j.channels]

    def parse_file(self, path: str | PathLike, encoding="utf-8"):
        with open(path, "r", encoding=encoding) as f:
            self.parse_string(f.read())

    def __repr__(self):
        return f"BVH {len(self.joints.keys())} joints, {self.frames} frames"


def main():
    # create Bvh parser
    anim = Bvh()
    # parser file
    anim.parse_file("example.bvh")

    # draw the skeleton in T-pose
    anim.plot_hierarchy()

    # extract single frame pose: axis0=joint, axis1=positionXYZ/rotationXYZ
    p, r = anim.frame_pose(0)

    # extract all poses: axis0=frame, axis1=joint, axis2=positionXYZ/rotationXYZ
    all_p, all_r = anim.all_frame_poses

    # print all joints, their positions and orientations
    for _p, _r, _j in zip(p, r, anim.joint_names):
        print(f"{_j}: p={_p}, r={_r}")

    # draw the skeleton for the given frame
    anim.plot_frame(22)


if __name__ == "__main__":
    main()
