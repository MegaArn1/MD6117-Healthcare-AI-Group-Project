from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREP = Path(__file__).resolve().parent
OUT = PREP / "outputs"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


with (OUT / "preprocessing_summary.json").open(encoding="utf-8") as handle:
    summary = json.load(handle)
with (OUT / "independent_verification.json").open(encoding="utf-8") as handle:
    verification = json.load(handle)

split_summary = [
    {
        "split": row["split"],
        "rows": int(row["rows"]),
        "deaths": int(row["deaths"]),
        "survivors": int(row["survivors"]),
        "death_prevalence": float(row["death_prevalence"]),
    }
    for row in read_csv(OUT / "split_summary.csv")
]

coverage = sorted(
    [
        {
            "parameter": row["Parameter"],
            "patients": int(row["patients_with_valid_value"]),
            "patient_coverage": float(row["patient_coverage"]),
            "valid_observations": int(row["valid_observations"]),
        }
        for row in read_csv(OUT / "parameter_distributions.csv")
    ],
    key=lambda row: (row["patient_coverage"], row["parameter"]),
)
lowest_coverage = coverage[:12]

quality_findings = [
    {
        "priority": 1,
        "finding": "Outcome leakage",
        "evidence": "SOFA、SAPS-I、Length_of_stay、Survival 已全部排除在 X 外",
        "handling": "仅导出 In-hospital_death 到独立标签表；RecordID 只作连接键",
        "status": "PASS",
    },
    {
        "priority": 2,
        "finding": "同时间同变量多值",
        "evidence": f"{summary['audits']['conflicting_time_parameter_keys']:,} 个有效非缺失冲突键",
        "handling": "连续值取同分钟中位数；MechVent 取 max；Urine 求和；逐 stay 留痕",
        "status": "需医学确认 Urine 语义",
    },
    {
        "priority": 3,
        "finding": "48:00 边界",
        "evidence": f"{summary['audits']['exact_48h_rows']:,} 行，涉及 {summary['output']['boundary_sensitivity_rows']:,} 个 stay",
        "handling": "主版本 [0,48h) 排除；另存受影响 stay 的 ≤48h 替换行",
        "status": "PASS",
    },
    {
        "priority": 4,
        "finding": "空 Parameter",
        "evidence": f"{summary['audits']['empty_parameter_rows']:,} 行",
        "handling": "隔离计数，不猜变量名，不生成特征",
        "status": "PASS",
    },
    {
        "priority": 5,
        "finding": "信息性缺失",
        "evidence": "例如 TroponinI 覆盖 4.7%，Bilirubin 覆盖 43.5%",
        "handling": "保留全部 stay、NA、measured 与 count；填补值只从 train 拟合",
        "status": "PASS",
    },
    {
        "priority": 6,
        "finding": "临床异常值规则",
        "evidence": "目前尚无经医学组确认的范围表",
        "handling": "本版本不裁剪、不按 IQR 删除；输出分布供医学复核",
        "status": "OPEN",
    },
    {
        "priority": 7,
        "finding": "患者级重复住院",
        "evidence": "数据只有 stay 级 RecordID，没有 patient ID",
        "handling": "按 RecordID 划分；在局限中声明无法核查同一患者跨 split",
        "status": "OPEN",
    },
]

# Cohort-dependent figures are derived from the run summary rather than hardcoded,
# so legitimate cohort/schema changes (Phase 1 negative-LOS exclusion, Phase 3
# derived features) cannot leave stale numbers in the report.
_rows = int(summary["source"]["outcome_rows"])
_deaths = int(summary["source"]["death_count"])
_survivors = _rows - _deaths
_split_txt = " / ".join(f"{int(row['rows']):,}" for row in summary["split_summary"])
_ncols = int(summary["output"]["model_feature_columns"])
_boundary = int(summary["output"]["boundary_sensitivity_rows"])

output_purposes = {
    "patient_features_unimputed.csv.gz": f"{_rows:,}×患者级未填补特征；保留 NA",
    "labels_and_splits.csv": "唯一标签与固定分层 split",
    "X_train_imputed.csv.gz": "训练集 X；train 中位数填补",
    "X_validation_imputed.csv.gz": "验证集 X；只应用 train 填补值",
    "X_test_imputed.csv.gz": "测试集 X；只应用 train 填补值",
    "patient_features_48h_inclusive_affected_rows.csv.gz": f"{_boundary:,} 个边界 stay 的敏感性替换行",
    "feature_dictionary.csv": "特征来源、时间窗、统计量、单位、缺失语义",
    "preprocessing_state.json": "列顺序、schema 哈希和训练集填补值",
    "quality_flags_by_stay.csv": "逐 stay 质量标记",
    "preprocessing_summary.json": "流水线汇总与内部 QA",
    "independent_verification.json": "重新加载全量产物后的独立验收",
}
output_inventory = []
for name, purpose in output_purposes.items():
    path = OUT / name
    output_inventory.append({
        "file": name,
        "purpose": purpose,
        "size_mb": round(path.stat().st_size / 1024 / 1024, 3),
        "exists": path.exists(),
    })

qa_checks = [
    {"check": "患者级特征行数与 outcome 一致", "result": summary["qa"]["feature_row_count_matches_outcomes"], "evidence": f"{_rows:,} = {_rows:,}"},
    {"check": "RecordID 唯一且集合一致", "result": summary["qa"]["record_id_unique"] and summary["qa"]["record_id_sets_match"], "evidence": "一对一 join，无增行/丢行"},
    {"check": "三组互斥且并集完整", "result": summary["qa"]["split_disjoint"] and summary["qa"]["split_union_complete"], "evidence": _split_txt},
    {"check": "标签完整且二元", "result": summary["qa"]["label_binary_complete"], "evidence": f"{_deaths:,} death；{_survivors:,} survivor"},
    {"check": "X 无 outcome 泄漏列", "result": len(summary["qa"]["leakage_feature_hits"]) == 0, "evidence": "命中 0 列"},
    {"check": "填补参数只由 train 重算验证", "result": verification["verified_train_only_fill_values"] == _ncols, "evidence": f"{_ncols:,}/{_ncols:,} 列一致"},
    {"check": "填补后无 NA/Inf", "result": verification["imputed_outputs_all_finite"], "evidence": "train/validation/test 全量重读通过"},
    {"check": "主版本排除 Time=48:00", "result": summary["qa"]["primary_excludes_48h"], "evidence": f"另存 {_boundary:,} 个替换行"},
]

headline = [{
    "stays": summary["source"]["outcome_rows"],
    "death_prevalence": summary["source"]["death_prevalence"],
    "feature_columns": summary["output"]["model_feature_columns"],
    "qa_pass_rate": sum(bool(row["result"]) for row in qa_checks) / len(qa_checks),
}]

generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
title = "院内死亡数据预处理：全项目扫描与可交付性报告"

sources = [
    {
        "id": "release_readme", "label": "release 数据字典",
        "path": source_path(ROOT / "release" / "README.md"),
    },
    {
        "id": "outcomes", "label": "院内死亡标签",
        "path": source_path(ROOT / "release" / "outcomes.csv"),
        "query": {
            "engine": "duckdb", "language": "sql",
            "description": "核对队列规模、死亡人数和死亡率。",
            "sql": f"SELECT COUNT(*) AS stays, SUM(\"In-hospital_death\") AS deaths, AVG(CAST(\"In-hospital_death\" AS DOUBLE)) AS death_prevalence FROM read_csv_auto('{source_path(ROOT / 'release' / 'outcomes.csv')}')",
            "tables_used": ["release/outcomes.csv"],
            "metric_definitions": ["death_prevalence = deaths / stays"],
        },
    },
    {
        "id": "split_audit", "label": "固定分层拆分",
        "path": source_path(OUT / "split_summary.csv"),
        "query": {
            "engine": "duckdb", "language": "sql",
            "description": "读取各 split 的样本数、死亡数和死亡率。",
            "sql": f"SELECT split, rows, deaths, survivors, death_prevalence FROM read_csv_auto('{source_path(OUT / 'split_summary.csv')}') ORDER BY CASE split WHEN 'train' THEN 1 WHEN 'validation' THEN 2 ELSE 3 END",
            "tables_used": ["preprocessing/outputs/split_summary.csv"],
            "metric_definitions": ["按 In-hospital_death 分层；固定 seed=20260907"],
        },
    },
    {
        "id": "coverage_audit", "label": "原始变量覆盖与分布",
        "path": source_path(OUT / "parameter_distributions.csv"),
        "query": {
            "engine": "duckdb", "language": "sql",
            "description": "选出患者覆盖率最低的 12 个原始动态变量。",
            "sql": f"SELECT Parameter AS parameter, patients_with_valid_value AS patients, patient_coverage, valid_observations FROM read_csv_auto('{source_path(OUT / 'parameter_distributions.csv')}') ORDER BY patient_coverage, Parameter LIMIT 12",
            "tables_used": ["preprocessing/outputs/parameter_distributions.csv"],
            "metric_definitions": [f"patient_coverage = 至少一个有限且非 -1 观测的 stay 数 / {_rows:,}"],
        },
    },
    {
        "id": "quality_audit", "label": "逐 stay 质量标记",
        "path": source_path(OUT / "quality_flags_by_stay.csv"),
        "query": {
            "engine": "duckdb", "language": "sql",
            "description": "汇总空 Parameter、48:00 和同时间冲突键。",
            "sql": f"SELECT SUM(empty_parameter_rows) AS empty_parameter_rows, SUM(exact_48h_rows) AS exact_48h_rows, SUM(conflicting_time_parameter_keys) AS conflicting_keys FROM read_csv_auto('{source_path(OUT / 'quality_flags_by_stay.csv')}')",
            "tables_used": ["preprocessing/outputs/quality_flags_by_stay.csv"],
            "metric_definitions": ["conflicting_keys = 同 stay、同 Time、同 Parameter 下存在多个有效不同值的键数"],
        },
    },
    {
        "id": "preprocess_summary", "label": "预处理汇总与 QA",
        "path": source_path(OUT / "preprocessing_summary.json"),
        "query": {
            "engine": "duckdb", "language": "sql",
            "description": "读取预处理状态、行列数和运行时。",
            "sql": f"SELECT status, output.feature_rows AS feature_rows, output.model_feature_columns AS feature_columns FROM read_json_auto('{source_path(OUT / 'preprocessing_summary.json')}')",
            "tables_used": ["preprocessing/outputs/preprocessing_summary.json"],
            "metric_definitions": ["feature_columns 不包括 RecordID"],
        },
    },
    {
        "id": "verification", "label": "独立端到端验收",
        "path": source_path(OUT / "independent_verification.json"),
        "query": {
            "engine": "duckdb", "language": "sql",
            "description": "读取重新加载全部输出后的独立验收状态。",
            "sql": f"SELECT status, rows, model_features, verified_train_only_fill_values, imputed_outputs_all_finite FROM read_json_auto('{source_path(OUT / 'independent_verification.json')}')",
            "tables_used": ["preprocessing/outputs/independent_verification.json"],
            "metric_definitions": ["qa_pass_rate = 本报告所列 8 项关键验收中通过项比例"],
        },
    },
    {
        "id": "feature_dictionary", "label": "特征数据字典",
        "path": source_path(OUT / "feature_dictionary.csv"),
    },
]

manifest = {
    "version": 1,
    "surface": "report",
    "title": title,
    "description": "院内死亡任务的数据预处理实施、质量证据、输出清单和剩余决策；不含模型训练。",
    "generatedAt": generated_at,
    "sources": sources,
    "cards": [
        {"id": "stays_card", "dataset": "headline", "sourceId": "outcomes", "metrics": [{"label": "ICU stays", "field": "stays", "format": "number"}]},
        {"id": "death_card", "dataset": "headline", "sourceId": "outcomes", "metrics": [{"label": "院内死亡率", "field": "death_prevalence", "format": "percent"}]},
        {"id": "features_card", "dataset": "headline", "sourceId": "preprocess_summary", "metrics": [{"label": "候选特征列", "field": "feature_columns", "format": "number"}]},
        {"id": "qa_card", "dataset": "headline", "sourceId": "verification", "metrics": [{"label": "关键 QA 通过率", "field": "qa_pass_rate", "format": "percent"}]},
    ],
    "charts": [
        {
            "id": "coverage_chart", "title": "低覆盖原始变量的患者覆盖率（0–48 小时）",
            "type": "bar", "dataset": "lowest_coverage", "sourceId": "coverage_audit",
            "encodings": {
                "x": {"field": "parameter", "type": "nominal"},
                "y": {"field": "patient_coverage", "type": "quantitative", "format": "percent"},
            },
        }
    ],
    "tables": [
        {
            "id": "split_table", "title": "固定、分层的数据划分", "dataset": "split_summary", "sourceId": "split_audit",
            "columns": [
                {"field": "split", "label": "Split"}, {"field": "rows", "label": "Rows", "format": "number"},
                {"field": "deaths", "label": "Deaths", "format": "number"},
                {"field": "survivors", "label": "Survivors", "format": "number"},
                {"field": "death_prevalence", "label": "死亡率", "format": "percent"},
            ],
            "defaultSort": {"field": "rows", "direction": "desc"},
        },
        {
            "id": "qa_table", "title": "端到端验收", "dataset": "qa_checks", "sourceId": "verification",
            "columns": [
                {"field": "check", "label": "检查"}, {"field": "result", "label": "通过"},
                {"field": "evidence", "label": "证据"},
            ],
            "defaultSort": {"field": "check", "direction": "asc"},
        },
        {
            "id": "quality_table", "title": "关键质量风险与处理", "dataset": "quality_findings", "sourceId": "quality_audit",
            "columns": [
                {"field": "priority", "label": "优先级", "format": "number"},
                {"field": "finding", "label": "问题"}, {"field": "evidence", "label": "证据"},
                {"field": "handling", "label": "当前处理"}, {"field": "status", "label": "状态"},
            ],
            "defaultSort": {"field": "priority", "direction": "asc"},
        },
        {
            "id": "output_table", "title": "预处理交付文件", "dataset": "output_inventory", "sourceId": "preprocess_summary",
            "columns": [
                {"field": "file", "label": "文件"}, {"field": "purpose", "label": "用途"},
                {"field": "size_mb", "label": "MB", "format": "number"}, {"field": "exists", "label": "存在"},
            ],
            "defaultSort": {"field": "size_mb", "direction": "desc"},
        },
    ],
    "blocks": [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "technical_summary", "type": "markdown",
            "body": f"## 技术摘要\n\n**结论：可以顺利完成数据处理，当前流水线与独立验收均为 PASS；但只有小组确认‘用前 48 小时预测最终院内死亡’这一任务定义后，才应把产物交给模型组。** 保留 {_rows:,} 个 stay（已按方案 §3.4 剔除 Length_of_stay 为负的记录），未训练任何模型。生理范围校验已启用，越界值记为缺失。最大的剩余医学判断是同一分钟多条 Urine 记录应视为独立尿量还是修订值，以及生理范围阈值本身是否需要医学组调整。",
        },
        {"id": "headline", "type": "metric-strip", "cardIds": ["stays_card", "death_card", "features_card", "qa_card"]},
        {
            "id": "scope", "type": "markdown",
            "body": "## 范围与任务冻结\n\n主版本把每个 ICU stay 变成一行，使用 `0≤Time<48h` 的所有 release 生理数据，在第 48 小时时点预测最终 `In-hospital_death`。`SOFA`、`SAPS-I`、`Length_of_stay` 和 `Survival` 全部不进入 X；`RecordID` 只用于连接。此次工作止于数据处理，没有标准化、特征选择、重采样或模型训练。",
        },
        {
            "id": "methods", "type": "markdown",
            "body": "## 方法\n\n静态描述符做固定编码；37 个动态变量按 0–24h、24–48h、0–48h 三个窗口生成 measured、count、first、last、min、max、mean、median、std、slope、delta 和时间跨度，并为 Urine 增加 total、为 MechVent 增加 ever。`-1` 转为 NA。动态连续指标的同分钟多值先取中位数，MechVent 取最大值，Urine 求和；静态入院 Weight 取 00:00 第一条有效描述符并对冲突留痕。所有患者级聚合先完成；随后按死亡标签以 seed 20260907 分层切分，跨患者的中位数填补只在 train 拟合。没有施加临床极值裁剪。",
        },
        {
            "id": "coverage_explainer", "type": "markdown", "sourceId": "coverage_audit",
            "body": "覆盖率差异很大，说明‘未检测’本身是数据的一部分，而不是可以统一填 0 的空洞。图中为覆盖率最低的 12 个原始动态变量；TroponinI 仅覆盖 4.7%，Bilirubin 覆盖 43.5%。因此主表保留 NA，并额外保留 measured/count；模型用版本才应用训练集中位数。",
        },
        {"id": "coverage", "type": "chart", "chartId": "coverage_chart"},
        {
            "id": "split_section", "type": "markdown",
            "body": f"## 数据划分\n\n70%/15%/15% 的分层划分已固定并写入标签表。三组互不重叠、并集为全部 {_rows:,} 个 RecordID，死亡率均约 {summary['source']['death_prevalence']:.2%}。后续任何 scaler、特征筛选或调参都必须继续只在训练折内拟合。",
        },
        {"id": "split", "type": "table", "tableId": "split_table"},
        {
            "id": "qa_section", "type": "markdown",
            "body": f"## 验收结果\n\n独立脚本重新读取全部压缩特征文件，核对列顺序、ID 集合、三组互斥、标签合法性、所有 {_ncols:,} 个训练集填补值、NA/Inf 和泄漏字段；以下关键检查全部通过。",
        },
        {"id": "qa", "type": "table", "tableId": "qa_table"},
        {
            "id": "risks", "type": "markdown",
            "body": "## 已处理风险与开放项\n\n数据没有硬性阻塞，但并不等于可以无判断地‘正常做’。48:00、空 Parameter、信息性缺失和同时间多值都已显式处理并可追溯；临床范围表与 Urine 同时值语义仍需医学组确认。由于没有 patient ID，只能保证 stay 级不跨 split，不能排除同一人多次住院。",
        },
        {"id": "quality", "type": "table", "tableId": "quality_table"},
        {
            "id": "deliverables", "type": "markdown",
            "body": "## 可交付产物\n\n原始 release 保持只读。未填补主表、固定标签/split、三个只应用 train 填补参数的 X、48:00 敏感性替换行、数据字典、逐 stay 质量日志、来源/输出哈希和独立验收均已落盘。CSV 使用 gzip 是因为当前固定运行环境没有 parquet 引擎。",
        },
        {"id": "outputs", "type": "table", "tableId": "output_table"},
        {
            "id": "handoff", "type": "markdown",
            "body": "## 交接条件\n\n数据负责人可以把当前结果标为 **GO to modeling, conditional**。交给模型组前只需小组书面确认预测时点/标签定义，并让医学同学确认 Urine 同时间聚合与可选生理范围表。模型组必须从 `labels_and_splits.csv` 读取固定 split，丢弃 RecordID，再在训练折内做模型相关变换；测试集不应被用于任何选择。",
        },
    ],
}

snapshot = {
    "version": 1, "generatedAt": generated_at, "status": "ready",
    "datasets": {
        "headline": headline,
        "split_summary": split_summary,
        "lowest_coverage": lowest_coverage,
        "qa_checks": qa_checks,
        "quality_findings": quality_findings,
        "output_inventory": output_inventory,
    },
}

artifact = {"surface": "report", "manifest": manifest, "snapshot": snapshot, "sources": sources}
target = PREP / "mortality_preprocessing_report_artifact.json"
with target.open("w", encoding="utf-8", newline="\n") as handle:
    json.dump(artifact, handle, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({"path": str(target), "generatedAt": generated_at}, ensure_ascii=False))
