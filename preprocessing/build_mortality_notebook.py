from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any

from jupyter_client import KernelManager


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = Path(__file__).resolve().parent / "mortality_data_preprocessing.ipynb"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def markdown_cell(source: str, index: int) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "id": f"cell-{index:02d}",
        "metadata": {},
        "source": source.strip() + "\n",
    }


def code_cell(source: str, index: int) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": f"cell-{index:02d}",
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def build_notebook() -> dict[str, Any]:
    summary = json.loads((OUTPUT_DIR / "preprocessing_summary.json").read_text(encoding="utf-8"))
    verification = json.loads(
        (OUTPUT_DIR / "independent_verification.json").read_text(encoding="utf-8")
    )

    rows = int(summary["output"]["feature_rows"])
    deaths = int(summary["source"]["death_count"])
    prevalence = float(summary["source"]["death_prevalence"])
    features = int(summary["output"]["model_feature_columns"])
    boundary_rows = int(summary["output"]["boundary_sensitivity_rows"])
    conflicts = int(summary["audits"]["conflicting_time_parameter_keys"])
    no_dynamic = int(verification["stays_without_valid_dynamic_observations"])

    cells: list[dict[str, Any]] = [
        markdown_cell(
            f"""
# 院内死亡预测数据预处理：可复现 QA

## tl;dr

**结论：PASS。** 预处理产物包含 {rows:,} 个 ICU stay、{features:,} 个模型特征；标签中死亡 {deaths:,} 例（{prevalence:.3%}）。固定分层拆分见 split_summary.csv；训练集参数驱动的填补后，三个 X 文件均无缺失且全为有限数。

需要保留的风险提示：{boundary_rows:,} 个 stay 在恰好 48:00 有记录，已单独导出边界敏感性行；原始记录存在 {conflicts:,} 个“同一时间、同一参数、不同值”键；{no_dynamic:,} 个 stay 在主窗口内没有有效动态观察。该笔记本只验收数据处理，**没有训练或比较任何模型**。
""",
            1,
        ),
        markdown_cell(
            """
## Context & Methods

本笔记本是 `preprocessing/outputs` 的轻量、可重跑验收件。它检查产物的一致性、拆分、标签分布、缺失、潜在标签泄漏、48:00 边界敏感性、同时间冲突，以及填补参数是否只由训练集计算。它不重新扫描原始病人文件，也不改变任何产物。

### Key Assumptions

- 任务定义：在 48 小时 landmark，利用前 48 小时数据预测最终 `In-hospital_death`。
- 主时间窗：`0 <= elapsed_minutes < 2880`；恰好 48:00 的记录只进入敏感性产物。
- `-1` 按缺失处理；连续变量同一时间冲突取中位数，`MechVent` 取最大值，`Urine` 求和，然后生成纵向摘要。
- 不做临床阈值裁剪，不做模型特定标准化；后续若需要，应只在训练折内拟合。
- `RecordID` 仅是连接键，不能作为模型特征。
""",
            2,
        ),
        code_cell(
            r'''
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd


def find_project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "preprocessing" / "outputs" / "preprocessing_summary.json").exists():
            return candidate
    raise FileNotFoundError("找不到 preprocessing/outputs；请从项目目录或其子目录运行。")


PROJECT_ROOT = find_project_root()
OUTPUT_DIR = PROJECT_ROOT / "preprocessing" / "outputs"
print(f"项目根目录: {PROJECT_ROOT}")
print(f"Python {platform.python_version()} | pandas {pd.__version__} | numpy {np.__version__}")
print(f"输入目录: {OUTPUT_DIR}")
''',
            3,
        ),
        markdown_cell(
            """
## Data

所有检查均基于预处理脚本已经生成的固定产物。标签与拆分在一个小文件中；未填补特征用于审计原始缺失模式；三个已填补 X 文件分别对应 train、validation、test；边界文件仅包含恰好 48:00 有记录的 stay。
""",
            4,
        ),
        code_cell(
            r'''
required_files = [
    "preprocessing_summary.json",
    "preprocessing_state.json",
    "labels_and_splits.csv",
    "patient_features_unimputed.csv.gz",
    "patient_features_48h_inclusive_affected_rows.csv.gz",
    "X_train_imputed.csv.gz",
    "X_validation_imputed.csv.gz",
    "X_test_imputed.csv.gz",
    "feature_dictionary.csv",
    "feature_missingness.csv",
    "quality_flags_by_stay.csv",
]
missing_files = [name for name in required_files if not (OUTPUT_DIR / name).exists()]
assert not missing_files, f"缺少产物: {missing_files}"

inventory = pd.DataFrame(
    {
        "file": required_files,
        "size_mib": [round((OUTPUT_DIR / name).stat().st_size / 1024**2, 3) for name in required_files],
    }
)
print(inventory.to_string(index=False))
''',
            5,
        ),
        code_cell(
            r'''
summary = json.loads((OUTPUT_DIR / "preprocessing_summary.json").read_text(encoding="utf-8"))
state = json.loads((OUTPUT_DIR / "preprocessing_state.json").read_text(encoding="utf-8"))
labels = pd.read_csv(OUTPUT_DIR / "labels_and_splits.csv")
features_unimputed = pd.read_csv(OUTPUT_DIR / "patient_features_unimputed.csv.gz")
feature_columns = state["feature_columns"]

split_table = (
    labels.groupby("split", sort=False)["In-hospital_death"]
    .agg(rows="size", deaths="sum", death_prevalence="mean")
    .reset_index()
)
split_table["survivors"] = split_table["rows"] - split_table["deaths"]
split_table = split_table[["split", "rows", "deaths", "survivors", "death_prevalence"]]

id_sets = {name: set(group["RecordID"]) for name, group in labels.groupby("split")}
split_names = sorted(id_sets)
splits_disjoint = all(
    id_sets[split_names[i]].isdisjoint(id_sets[split_names[j]])
    for i in range(len(split_names))
    for j in range(i + 1, len(split_names))
)

cohort_checks = {
    "feature_rows_equal_label_rows": len(features_unimputed) == len(labels),
    "feature_RecordID_unique": features_unimputed["RecordID"].is_unique,
    "label_RecordID_unique": labels["RecordID"].is_unique,
    "RecordID_sets_equal": set(features_unimputed["RecordID"]) == set(labels["RecordID"]),
    "labels_binary_and_complete": labels["In-hospital_death"].notna().all()
    and set(labels["In-hospital_death"]) == {0, 1},
    "split_sets_disjoint": splits_disjoint,
    "split_union_complete": set().union(*id_sets.values()) == set(labels["RecordID"]),
    "feature_schema_matches_state": features_unimputed.columns.tolist()
    == ["RecordID", *feature_columns],
}
assert all(cohort_checks.values()), cohort_checks

print(split_table.to_string(index=False, formatters={"death_prevalence": "{:.3%}".format}))
print(f"\n总死亡: {int(labels['In-hospital_death'].sum()):,} / {len(labels):,} "
      f"({labels['In-hospital_death'].mean():.3%})")
''',
            6,
        ),
        markdown_cell(
            """
## Results

下面的单元逐项验证特征模式、原始缺失、训练集填补、拆分文件及两个需要显式交接的质量风险。任一核心断言失败，笔记本都会停止执行。
""",
            7,
        ),
        code_cell(
            r'''
forbidden_terms = ("in_hospital_death", "survival", "length_of_stay", "sofa", "saps_i")
leakage_hits = [
    column
    for column in feature_columns
    if any(term in column.lower().replace("-", "_") for term in forbidden_terms)
]
missing_cells = int(features_unimputed[feature_columns].isna().sum().sum())
top_missing = (
    features_unimputed[feature_columns]
    .isna()
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .rename("missing_fraction")
    .reset_index()
    .rename(columns={"index": "feature"})
)

assert not leakage_hits, leakage_hits
assert missing_cells == int(summary["output"]["unimputed_missing_cells"])
assert len(feature_columns) == len(set(feature_columns)) == int(summary["output"]["model_feature_columns"])

print(f"模型特征数: {len(feature_columns):,}")
print(f"未填补缺失单元格: {missing_cells:,}")
print(f"泄漏字段命中: {leakage_hits}")
print("\n缺失率最高的 10 个特征：")
print(top_missing.to_string(index=False, formatters={"missing_fraction": "{:.2%}".format}))
''',
            8,
        ),
        code_cell(
            r'''
train_ids = set(labels.loc[labels["split"].eq("train"), "RecordID"])
train_unimputed = features_unimputed.loc[
    features_unimputed["RecordID"].isin(train_ids), feature_columns
]
computed_train_fill = train_unimputed.median(axis=0, skipna=True).fillna(0.0)
saved_train_fill = pd.Series(state["fill_values"], dtype="float64").reindex(feature_columns)
fill_values_match = bool(
    np.allclose(
        computed_train_fill.to_numpy(dtype="float64"),
        saved_train_fill.to_numpy(dtype="float64"),
        rtol=2e-6,
        atol=2e-6,
    )
)

imputed_rows = []
all_imputed_ids: set[int] = set()
all_finite = True
for split in ["train", "validation", "test"]:
    frame = pd.read_csv(OUTPUT_DIR / f"X_{split}_imputed.csv.gz")
    expected_ids = set(labels.loc[labels["split"].eq(split), "RecordID"])
    values = frame[feature_columns].to_numpy(dtype="float64", copy=False)
    finite = bool(np.isfinite(values).all())
    schema_ok = frame.columns.tolist() == ["RecordID", *feature_columns]
    ids_ok = set(frame["RecordID"]) == expected_ids
    all_finite = all_finite and finite
    all_imputed_ids |= set(frame["RecordID"])
    imputed_rows.append(
        {
            "split": split,
            "rows": len(frame),
            "schema_ok": schema_ok,
            "ids_ok": ids_ok,
            "missing_cells": int(frame[feature_columns].isna().sum().sum()),
            "all_finite": finite,
        }
    )
    del frame, values

imputed_check_table = pd.DataFrame(imputed_rows)
assert fill_values_match
assert imputed_check_table[["schema_ok", "ids_ok", "all_finite"]].all().all()
assert int(imputed_check_table["missing_cells"].sum()) == 0
assert all_imputed_ids == set(labels["RecordID"])

print(f"训练集填补参数逐列复算一致: {fill_values_match} ({len(feature_columns):,} / {len(feature_columns):,})")
print(imputed_check_table.to_string(index=False))
''',
            9,
        ),
        code_cell(
            r'''
quality = pd.read_csv(OUTPUT_DIR / "quality_flags_by_stay.csv")
boundary = pd.read_csv(OUTPUT_DIR / "patient_features_48h_inclusive_affected_rows.csv.gz")
affected_ids = set(quality.loc[quality["exact_48h_rows"].gt(0), "RecordID"])

primary_affected = features_unimputed.set_index("RecordID").loc[boundary["RecordID"]]
boundary_indexed = boundary.set_index("RecordID")
same_values = np.isclose(
    primary_affected[feature_columns].to_numpy(dtype="float64"),
    boundary_indexed[feature_columns].to_numpy(dtype="float64"),
    equal_nan=True,
    rtol=1e-7,
    atol=1e-7,
)
changed_feature_cells = int((~same_values).sum())
changed_stays = int((~same_values).any(axis=1).sum())

conflict_keys = int(quality["conflicting_time_parameter_keys"].sum())
conflict_stays = int(quality["conflicting_time_parameter_keys"].gt(0).sum())
exact_duplicates = int(quality["exact_duplicate_rows"].sum()) if "exact_duplicate_rows" in quality else 0
no_dynamic = int(features_unimputed["record__0_48h__valid_observation_count"].eq(0).sum())

risk_table = pd.DataFrame(
    [
        ["恰好 48:00 的原始行", int(quality["exact_48h_rows"].sum()), "主特征排除；敏感性文件纳入"],
        ["48:00 受影响 stay", len(affected_ids), f"{changed_stays} 个 stay 的特征发生变化"],
        ["边界敏感性变化单元格", changed_feature_cells, f"仅限 {len(boundary)} 行的替代特征"],
        ["同时间-同参数冲突键", conflict_keys, f"涉及 {conflict_stays} 个 stay；按预设规则聚合"],
        ["完全相同的重复行", exact_duplicates, "未发现"],
        ["无有效动态观察 stay", no_dynamic, "保留；依靠静态、缺失指示和填补值"],
    ],
    columns=["检查项", "数量", "解释"],
)

assert len(boundary) == len(affected_ids) == int(summary["output"]["boundary_sensitivity_rows"])
assert set(boundary["RecordID"]) == affected_ids
assert boundary.columns.tolist() == features_unimputed.columns.tolist()
assert conflict_keys == int(summary["audits"]["conflicting_time_parameter_keys"])

print(risk_table.to_string(index=False))
''',
            10,
        ),
        code_cell(
            r'''
qa_checks = {
    **cohort_checks,
    "death_count_matches_summary": int(labels["In-hospital_death"].sum())
    == int(summary["source"]["death_count"]),
    "split_sizes": labels["split"].value_counts().to_dict()
    == {row["split"]: int(row["rows"]) for row in summary["split_summary"]},
    "no_forbidden_outcome_or_score_features": not leakage_hits,
    "unimputed_missing_count_reconciles": missing_cells
    == int(summary["output"]["unimputed_missing_cells"]),
    "train_only_fill_values_recomputed": fill_values_match,
    "all_imputed_splits_complete_and_finite": all_finite
    and int(imputed_check_table["missing_cells"].sum()) == 0,
    "boundary_file_reconciles": set(boundary["RecordID"]) == affected_ids
    and len(boundary) == int(summary["output"]["boundary_sensitivity_rows"]),
    "primary_window_excludes_exact_48h": summary["qa"]["primary_excludes_48h"] is True,
}
qa_table = pd.DataFrame(
    {"check": list(qa_checks), "status": ["PASS" if value else "FAIL" for value in qa_checks.values()]}
)
assert all(qa_checks.values()), qa_table.loc[qa_table["status"].eq("FAIL")].to_dict("records")
print(qa_table.to_string(index=False))
print(f"\n最终状态: PASS ({len(qa_checks)} / {len(qa_checks)} 项)")
''',
            11,
        ),
        markdown_cell(
            f"""
## Takeaways

- 数据处理产物通过本笔记本的端到端一致性检查：{rows:,} 行、{features:,} 个特征、死亡率 {prevalence:.3%}，固定拆分互斥且覆盖完整。
- 三个已填补 X 文件可交给建模同学；标签需从 `labels_and_splits.csv` 按 `RecordID` 连接，随后必须丢弃 `RecordID`，不能把它作为特征。
- {boundary_rows:,} 个 stay 应做一次“是否纳入恰好 48:00”的敏感性比较；这不是主分析的替代，而是验证时间边界是否影响结论。
- 同时间冲突已经按显式规则聚合并保留审计计数；若团队想换规则，需要重跑预处理并记录为不同版本。
- 仍需团队确认 48 小时 landmark 的任务定义，并由医学同学审核未来任何生理范围规则。当前版本没有裁剪异常值，也没有训练模型。
""",
            12,
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": platform.python_version(),
                "mimetype": "text/x-python",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
                "nbconvert_exporter": "python",
                "file_extension": ".py",
            },
            "qa_scope": "mortality preprocessing outputs only; no model training",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def normalize_mime_bundle(data: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for mime, value in data.items():
        if isinstance(value, bytes):
            normalized[mime] = value.decode("utf-8", errors="replace")
        else:
            normalized[mime] = value
    return normalized


def execute_notebook(notebook: dict[str, Any]) -> None:
    kernel_manager = KernelManager(kernel_name="python3")
    kernel_manager.start_kernel(cwd=str(PROJECT_ROOT))
    client = kernel_manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        execution_count = 0
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            execution_count += 1
            message_id = client.execute(cell["source"], store_history=True)
            outputs: list[dict[str, Any]] = []
            deadline = time.monotonic() + 300
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Cell {cell['id']} exceeded 300 seconds")
                message = client.get_iopub_msg(timeout=min(remaining, 30))
                if message.get("parent_header", {}).get("msg_id") != message_id:
                    continue
                message_type = message["header"]["msg_type"]
                content = message["content"]
                if message_type == "stream":
                    text = content.get("text", "")
                    if outputs and outputs[-1].get("output_type") == "stream" and outputs[-1].get("name") == content.get("name"):
                        outputs[-1]["text"] += text
                    else:
                        outputs.append(
                            {
                                "name": content.get("name", "stdout"),
                                "output_type": "stream",
                                "text": text,
                            }
                        )
                elif message_type in {"execute_result", "display_data"}:
                    output: dict[str, Any] = {
                        "data": normalize_mime_bundle(content.get("data", {})),
                        "metadata": content.get("metadata", {}),
                        "output_type": message_type,
                    }
                    if message_type == "execute_result":
                        output["execution_count"] = content.get("execution_count")
                    outputs.append(output)
                elif message_type == "error":
                    error_output = {
                        "ename": content.get("ename", "Error"),
                        "evalue": content.get("evalue", ""),
                        "output_type": "error",
                        "traceback": content.get("traceback", []),
                    }
                    outputs.append(error_output)
                    cell["outputs"] = outputs
                    cell["execution_count"] = execution_count
                    raise RuntimeError(
                        f"Cell {cell['id']} failed: {error_output['ename']}: {error_output['evalue']}"
                    )
                elif message_type == "status" and content.get("execution_state") == "idle":
                    break
            reply = client.get_shell_msg(timeout=30)
            if reply.get("parent_header", {}).get("msg_id") == message_id:
                status = reply.get("content", {}).get("status")
                if status != "ok":
                    raise RuntimeError(f"Cell {cell['id']} shell status: {status}")
            cell["outputs"] = outputs
            cell["execution_count"] = execution_count
    finally:
        client.stop_channels()
        kernel_manager.shutdown_kernel(now=True)


def write_notebook(notebook: dict[str, Any]) -> None:
    NOTEBOOK_PATH.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    notebook = build_notebook()
    write_notebook(notebook)
    try:
        execute_notebook(notebook)
    except Exception:
        write_notebook(notebook)
        raise
    write_notebook(notebook)
    print(f"Executed notebook written to {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
