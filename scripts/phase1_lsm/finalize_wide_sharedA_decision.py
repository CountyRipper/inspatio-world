#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


DECISIONS = (
    "ENTER_PHASE2",
    "REMAIN_PHASE1",
    "STOP_AND_RETHINK",
    "REMAIN_PHASE1 — PENDING_USER_VISUAL_REVIEW",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--decision", required=True, choices=DECISIONS)
    parser.add_argument("--visual-summary")
    parser.add_argument("--wide-recovery-count", type=int, default=0)
    parser.add_argument("--failure-reason")
    parser.add_argument("--wrong-content-destruction", action="store_true")
    parser.add_argument("--progressive-degradation", action="store_true")
    parser.add_argument("--no-dominant-artifacts", action="store_true")
    parser.add_argument("--exploratory", action="store_true")
    parser.add_argument("--result-path")
    args = parser.parse_args()

    root = Path(args.root)
    aggregate_path = root / "aggregate_metrics.json"
    if (root / "PHASE_DECISION.md").exists():
        raise FileExistsError("refusing to overwrite PHASE_DECISION.md")
    if args.failure_reason:
        if args.decision != "REMAIN_PHASE1":
            raise AssertionError("an incomplete non-OOM attempt can only remain in Phase 1")
        failure = {
            "status": "INCOMPLETE_NON_OOM_FAILURE",
            "reason": args.failure_reason,
            "scientific_results_available": False,
            "phase_decision": args.decision,
        }
        aggregate_path.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        for name, reached in (("shared_A_audit.json", False), ("projection_audit.json", False), ("hard_gate_audit.json", False)):
            path = root / name
            if not path.exists():
                path.write_text(json.dumps({"passed": False, "complete": reached, "reason": args.failure_reason}, indent=2) + "\n", encoding="utf-8")
        (root / "metrics_per_query.csv").write_text(
            "scene,query,sample_id,condition,actual_yaw_degrees,overlap_coverage,latent_displacement_mean_pixels,eligible,overlap_masked_latent_raw_l1,overlap_decoded_pixel_l1,invalid_spill_l1,full_composite_raw_l1,runtime_ms\n",
            encoding="utf-8",
        )
        (root / "PHASE1_WIDE_REPORT_ZH.md").write_text(
            "# Phase 1 宽视角 shared-memory 验证\n\n"
            "本 attempt 因非 OOM 工程断言错误在 S0 exact projection 审计处停止；按预注册规则未迁移 GPU、未重跑。\n\n"
            f"失败原因：{args.failure_reason}\n\n"
            "没有产生训练、四组评测、aggregate、完整视频或 montage，因此没有证据进入 Phase 2。失败现场和 GPU 快照均已保留。\n\n"
            "修正后仍需新的明确授权才能重新执行；本轮没有启动 Phase 2，也没有解冻 backbone。\n\n"
            f"## Phase 决策\n\n{args.decision}\n",
            encoding="utf-8",
        )
        (root / "PHASE_DECISION.md").write_text(args.decision + "\n", encoding="utf-8")
        with (root / "COMMAND_LOG.md").open("a", encoding="utf-8") as handle:
            handle.write(
                "\n## Incomplete non-OOM finalization\n\n"
                f"- Time: {datetime.now().astimezone().isoformat()}\n"
                f"- Reason: {args.failure_reason}\n"
                f"- Decision: {args.decision}\n"
            )
        return
    if args.visual_summary is None:
        raise ValueError("--visual-summary is required for a completed run")
    aggregate = json.loads(aggregate_path.read_text())
    quantitative = aggregate["quantitative_gates"]
    visual = {
        "summary": args.visual_summary,
        "recognizable_wide_recovery_count": args.wide_recovery_count,
        "wrong_memory_content_related_destruction": args.wrong_content_destruction,
        "reasonable_progressive_degradation": args.progressive_degradation,
        "no_dominant_new_ghosting_structure_drift_or_flicker": args.no_dominant_artifacts,
    }
    if args.decision == "ENTER_PHASE2":
        if args.exploratory:
            rates = aggregate["win_rates"]
            required = (
                all(aggregate["audits"].values())
                and rates["correct_beats_both_mask_only_and_wrong_same_mask"] >= 0.8
                and rates["correct_beats_no_memory"] >= 0.7
                and quantitative["both_scenes_exact_medium_direction_consistent"]
                and args.wide_recovery_count >= 1
                and args.wrong_content_destruction
                and args.progressive_degradation
                and args.no_dominant_artifacts
            )
        else:
            required = (
                all(quantitative.values())
                and args.wide_recovery_count >= 2
                and args.wrong_content_destruction
                and args.progressive_degradation
                and args.no_dominant_artifacts
            )
        if not required:
            raise AssertionError("ENTER_PHASE2 requested without all quantitative and visual gates")
    aggregate["visual_review"] = visual
    aggregate["phase_decision"] = args.decision
    aggregate_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    (root / "PHASE_DECISION.md").write_text(args.decision + "\n", encoding="utf-8")

    report_path = root / "PHASE1_WIDE_REPORT_ZH.md"
    with report_path.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## 最终定性观察\n\n"
            f"{args.visual_summary}\n\n"
            f"- 可识别 wide-view 历史恢复 query 数：{args.wide_recovery_count}\n"
            f"- wrong-memory 内容相关破坏：{args.wrong_content_destruction}\n"
            f"- 随角度/coverage 合理渐进退化：{args.progressive_degradation}\n"
            f"- 无主导性新增鬼影、结构漂移或闪烁：{args.no_dominant_artifacts}\n"
            "\n## Phase 决策\n\n"
            f"{args.decision}\n"
        )
    with (root / "COMMAND_LOG.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## Final visual decision\n\n"
            f"- Time: {datetime.now().astimezone().isoformat()}\n"
            f"- Decision: {args.decision}\n"
            f"- Recognizable wide recovery count: {args.wide_recovery_count}\n"
        )
    if args.result_path:
        result_path = Path(args.result_path)
        if result_path.exists():
            raise FileExistsError(f"refusing to overwrite {result_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    main()
