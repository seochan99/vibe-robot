from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def box(ax, x, y, w, h, text, fc="#f2f4f8", ec="#5a6473", fontsize=9, lw=1.2):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)


def arrow(ax, x1, y1, x2, y2, color="#2d3748"):
    ar = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="->", mutation_scale=12, linewidth=1.3, color=color)
    ax.add_patch(ar)


def make_system_overview() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.96, "ArmCraft Pipeline (Implemented + Proposed)", fontsize=13, fontweight="bold", va="top")

    box(ax, 0.03, 0.60, 0.18, 0.23, "Natural Language\nTask Request", fc="#e8f0ff")
    box(ax, 0.25, 0.60, 0.18, 0.23, "Task Spec Builder\n(goal, constraints,\nsuccess criteria)", fc="#e8f0ff")
    box(ax, 0.47, 0.60, 0.18, 0.23, "Task-to-Code Agent\n(intent + affordance +\nplan + safety)", fc="#e8f0ff")
    box(ax, 0.69, 0.60, 0.18, 0.23, "Sim-First Runner\n(preview + randomization\nchecks)", fc="#e8f0ff")

    arrow(ax, 0.21, 0.71, 0.25, 0.71)
    arrow(ax, 0.43, 0.71, 0.47, 0.71)
    arrow(ax, 0.65, 0.71, 0.69, 0.71)

    box(ax, 0.15, 0.20, 0.24, 0.24, "Failure Inspector\n(collision/grasp/IK/\nworkspace diagnosis)", fc="#fff3dd")
    box(ax, 0.45, 0.20, 0.24, 0.24, "Patch Suggestions\n(angle/speed/path/\nforce/target edits)", fc="#fff3dd")
    box(ax, 0.75, 0.20, 0.20, 0.24, "Approve & Execute\n(in live sim / robot)", fc="#ddf7e7")

    arrow(ax, 0.78, 0.60, 0.80, 0.44)
    arrow(ax, 0.57, 0.60, 0.56, 0.44)
    arrow(ax, 0.34, 0.44, 0.45, 0.32)
    arrow(ax, 0.69, 0.32, 0.75, 0.32)
    arrow(ax, 0.24, 0.20, 0.24, 0.57)

    ax.text(0.03, 0.06, "Key mechanism: execution requires simulation preview and explicit user approval.", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "system_overview.png", bbox_inches="tight")
    plt.close(fig)


def make_interaction_loop() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.2), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.95, "Vibe-to-Verify Interaction Loop", fontsize=13, fontweight="bold", va="top")

    box(ax, 0.03, 0.46, 0.17, 0.27, "User command\n(plain language)", fc="#ecebff")
    box(ax, 0.24, 0.46, 0.18, 0.27, "AI plan +\nSafety contract", fc="#ecebff")
    box(ax, 0.46, 0.46, 0.16, 0.27, "Chat preview\n(animation)", fc="#ecebff")
    box(ax, 0.66, 0.46, 0.13, 0.27, "Approve?", fc="#ecebff")
    box(ax, 0.83, 0.46, 0.14, 0.27, "Execute\nin simulator", fc="#ddf7e7")

    arrow(ax, 0.20, 0.60, 0.24, 0.60)
    arrow(ax, 0.42, 0.60, 0.46, 0.60)
    arrow(ax, 0.62, 0.60, 0.66, 0.60)
    arrow(ax, 0.79, 0.60, 0.83, 0.60)

    box(ax, 0.24, 0.10, 0.34, 0.23, "Failure Inspector\n(what failed, where, why)", fc="#ffe6e6")
    box(ax, 0.62, 0.10, 0.35, 0.23, "Patch Suggestions\n(apply suggestion -> re-simulate -> compare)", fc="#fff3dd")

    arrow(ax, 0.88, 0.46, 0.79, 0.22)
    arrow(ax, 0.58, 0.22, 0.62, 0.22)
    arrow(ax, 0.24, 0.22, 0.20, 0.46)

    ax.text(0.03, 0.02, "Loop objective: keep novice users in-control while reducing unsafe or non-explainable execution.", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "interaction_loop.png", bbox_inches="tight")
    plt.close(fig)


def make_study_design() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.5), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.95, "Planned Evaluation Design (CHI 2027)", fontsize=13, fontweight="bold", va="top")

    box(ax, 0.03, 0.56, 0.28, 0.28, "Study 1 (N=24-36)\nControlled experiment\nTasks: pick-place, sorting,\nreach-avoid", fc="#e8f0ff")
    box(ax, 0.35, 0.56, 0.20, 0.28, "Condition A\nVibe-only", fc="#f8f9fb")
    box(ax, 0.58, 0.56, 0.20, 0.28, "Condition B\nSim-first only", fc="#f8f9fb")
    box(ax, 0.81, 0.56, 0.16, 0.28, "Condition C\nArmCraft", fc="#ddf7e7")

    box(ax, 0.03, 0.15, 0.30, 0.28, "Metrics\n- time-to-first-success\n- failures/collisions\n- transfer success\n- NASA-TLX", fc="#fff3dd")
    box(ax, 0.36, 0.15, 0.30, 0.28, "RQ2/RQ3\n- explanation usefulness\n- attribution target\n(self/robot/agent)\n- trust & control", fc="#fff3dd")
    box(ax, 0.69, 0.15, 0.28, 0.28, "Study 2\nThink-aloud + interview\nFailure episodes coding\nfor repair behavior", fc="#ffe6e6")

    arrow(ax, 0.31, 0.70, 0.35, 0.70)
    arrow(ax, 0.55, 0.70, 0.58, 0.70)
    arrow(ax, 0.78, 0.70, 0.81, 0.70)

    fig.tight_layout()
    fig.savefig(OUT / "study_design.png", bbox_inches="tight")
    plt.close(fig)


def make_failure_inspector() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=220)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.95, "Failure Inspector Card Anatomy", fontsize=13, fontweight="bold", va="top")

    box(ax, 0.03, 0.64, 0.27, 0.23, "Failure Type\ncollision / grasp fail /\nunreachable / IK fail", fc="#ffe6e6")
    box(ax, 0.34, 0.64, 0.27, 0.23, "When It Failed\nstep index + timeline\nclip in chat preview", fc="#fff3dd")
    box(ax, 0.65, 0.64, 0.32, 0.23, "Why It Failed\ncontact overlap / pose mismatch /\nworkspace constraint violation", fc="#fff3dd")

    box(ax, 0.05, 0.26, 0.28, 0.24, "Patch 1\nTop-down regrasp\n+ slower approach", fc="#e8f0ff")
    box(ax, 0.36, 0.26, 0.28, 0.24, "Patch 2\nRetarget + path replan\nwith object lock", fc="#e8f0ff")
    box(ax, 0.67, 0.26, 0.28, 0.24, "Patch 3\nLower speed/force limit\n+ settle wait", fc="#e8f0ff")

    arrow(ax, 0.16, 0.64, 0.19, 0.50)
    arrow(ax, 0.47, 0.64, 0.50, 0.50)
    arrow(ax, 0.81, 0.64, 0.80, 0.50)

    ax.text(
        0.03,
        0.07,
        "Design choice: show at most 2-3 high-utility patches to reduce novice decision fatigue.",
        fontsize=9,
    )

    fig.tight_layout()
    fig.savefig(OUT / "failure_inspector.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    make_system_overview()
    make_interaction_loop()
    make_study_design()
    make_failure_inspector()


if __name__ == "__main__":
    main()
