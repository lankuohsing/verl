#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
对比两个 parquet 数据集的差异（重点排查会影响 verl / HF datasets 加载的点）：
- Arrow schema（列名、类型、是否有 list/struct 等嵌套类型）
- Parquet 文件级 metadata（key_value_metadata，尤其是 huggingface / datasets 写入的 features 信息）
- 尝试用 datasets.load_dataset("parquet") 加载并打印 features；若失败则打印异常堆栈要点

用法：
  python examples/data_preprocess/compare_parquet_datasets.py \
    --a /path/to/gsm8k_for_ppo/train.parquet \
    --b /path/to/gsm8k_good/train.parquet
"""

from __future__ import annotations

import argparse
import json
import textwrap
from typing import Any, Dict, List, Optional, Tuple


def _safe_decode(b: Any) -> str:
    if b is None:
        return ""
    if isinstance(b, bytes):
        return b.decode("utf-8", errors="replace")
    return str(b)


def _try_import_pyarrow() -> bool:
    try:
        import pyarrow as pa  # noqa: F401
        import pyarrow.parquet as pq  # noqa: F401
        return True
    except Exception:
        return False


def _try_import_datasets() -> bool:
    try:
        import datasets  # noqa: F401
        return True
    except Exception:
        return False


def read_parquet_summary(path: str) -> Dict[str, Any]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    meta = pf.metadata
    schema = pf.schema_arrow

    kv: Dict[str, str] = {}
    if meta is not None and meta.metadata is not None:
        # meta.metadata: Dict[bytes, bytes]
        for k, v in meta.metadata.items():
            kv[_safe_decode(k)] = _safe_decode(v)

    cols: List[Dict[str, str]] = []
    for field in schema:
        cols.append({"name": field.name, "type": str(field.type), "nullable": str(field.nullable)})

    return {
        "path": path,
        "num_row_groups": int(meta.num_row_groups) if meta is not None else None,
        "num_rows": int(meta.num_rows) if meta is not None else None,
        "created_by": _safe_decode(meta.created_by) if meta is not None else None,
        "schema_str": str(schema),
        "columns": cols,
        "file_kv_metadata": kv,
    }


def try_hf_datasets_load(path: str) -> Dict[str, Any]:
    """
    尝试复现 verl 里的加载方式：
      datasets.load_dataset("parquet", data_files=parquet_file)["train"]
    """
    out: Dict[str, Any] = {"ok": False, "error": None, "features": None, "num_rows": None, "column_names": None}
    try:
        import datasets

        ds = datasets.load_dataset("parquet", data_files=path)["train"]
        out["ok"] = True
        out["num_rows"] = int(ds.num_rows)
        out["column_names"] = list(ds.column_names)
        # features 里经常能看到 List / Sequence / LargeList 等
        out["features"] = ds.features.to_dict() if ds.features is not None else None
    except Exception as e:
        out["error"] = repr(e)
    return out


def _extract_hf_features_from_kv(kv: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """
    有些 parquet 是用 HF datasets 写出的，会在 metadata 里写入 JSON：
    - "huggingface" / "hf" / "datasets" 相关 key
    - 或者包含 {"info": {"features": ...}} 结构

    这里做宽松解析：扫描所有 metadata value，尝试 json.loads 并找到疑似 features 的结构。
    """
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for k, v in kv.items():
        vv = v.strip()
        if not vv or len(vv) < 2:
            continue
        if vv[0] not in "{[":
            continue
        try:
            obj = json.loads(vv)
        except Exception:
            continue
        if isinstance(obj, dict):
            candidates.append((k, obj))

    # 优先找 datasets 的典型结构
    for k, obj in candidates:
        info = obj.get("info") if isinstance(obj, dict) else None
        if isinstance(info, dict) and "features" in info:
            return {"metadata_key": k, "info_features": info.get("features")}

    # 其次：直接包含 features
    for k, obj in candidates:
        if "features" in obj:
            return {"metadata_key": k, "features": obj.get("features")}

    return None


def _diff_lists(a: List[str], b: List[str]) -> Dict[str, List[str]]:
    sa, sb = set(a), set(b)
    return {
        "only_in_a": sorted(sa - sb),
        "only_in_b": sorted(sb - sa),
        "in_both": sorted(sa & sb),
    }


def compare(a_path: str, b_path: str) -> None:
    a = read_parquet_summary(a_path)
    b = read_parquet_summary(b_path)

    print("\n" + "=" * 100)
    print("### 1) Arrow schema (pyarrow) 概览")
    print("=" * 100)
    for tag, s in [("A", a), ("B", b)]:
        print(f"\n[{tag}] {s['path']}")
        print(f"  - rows: {s['num_rows']}, row_groups: {s['num_row_groups']}")
        print(f"  - created_by: {s['created_by']}")
        print("  - columns:")
        for c in s["columns"]:
            print(f"    - {c['name']}: {c['type']} (nullable={c['nullable']})")

    a_cols = [c["name"] for c in a["columns"]]
    b_cols = [c["name"] for c in b["columns"]]
    print("\n" + "-" * 100)
    print("列名差异（只看 name）")
    print(json.dumps(_diff_lists(a_cols, b_cols), ensure_ascii=False, indent=2))

    # 进一步对齐同名列的类型差异
    a_types = {c["name"]: c["type"] for c in a["columns"]}
    b_types = {c["name"]: c["type"] for c in b["columns"]}
    type_diff = []
    for name in sorted(set(a_types) & set(b_types)):
        if a_types[name] != b_types[name]:
            type_diff.append({"column": name, "a_type": a_types[name], "b_type": b_types[name]})
    print("\n同名列的类型差异：")
    print(json.dumps(type_diff, ensure_ascii=False, indent=2))

    print("\n" + "=" * 100)
    print("### 2) Parquet 文件级 metadata (key_value_metadata)")
    print("=" * 100)
    a_kv = a["file_kv_metadata"]
    b_kv = b["file_kv_metadata"]

    a_keys = sorted(a_kv.keys())
    b_keys = sorted(b_kv.keys())
    print("\nmetadata key 差异：")
    print(json.dumps(_diff_lists(a_keys, b_keys), ensure_ascii=False, indent=2))

    # 打印所有 key（value 可能很长，做截断）
    def _print_kv(tag: str, kv: Dict[str, str]) -> None:
        print(f"\n[{tag}] metadata keys ({len(kv)}):")
        for k in sorted(kv.keys()):
            v = kv[k]
            v1 = v if len(v) <= 240 else v[:240] + " ... (truncated)"
            print(f"  - {k} = {v1}")

    _print_kv("A", a_kv)
    _print_kv("B", b_kv)

    # 尝试从 metadata 解析 HF features
    a_hf = _extract_hf_features_from_kv(a_kv)
    b_hf = _extract_hf_features_from_kv(b_kv)
    print("\n" + "-" * 100)
    print("metadata 中疑似 HF features 结构（若存在）")
    print("[A]")
    print(json.dumps(a_hf, ensure_ascii=False, indent=2))
    print("[B]")
    print(json.dumps(b_hf, ensure_ascii=False, indent=2))

    print("\n" + "=" * 100)
    print("### 3) datasets.load_dataset(\"parquet\") 尝试加载（对齐 verl 行为）")
    print("=" * 100)
    a_hf_load = try_hf_datasets_load(a_path)
    b_hf_load = try_hf_datasets_load(b_path)
    print("\n[A] load_dataset 结果：")
    print(json.dumps(a_hf_load, ensure_ascii=False, indent=2))
    print("\n[B] load_dataset 结果：")
    print(json.dumps(b_hf_load, ensure_ascii=False, indent=2))

    print("\n" + "=" * 100)
    print("### 4) 结论提示（针对你这个报错）")
    print("=" * 100)
    msg = """
你遇到的错误：
  ValueError: Feature type 'List' not found. Available feature types: ... LargeList, Sequence ...

这通常意味着：parquet 的 metadata 里记录了一份 HF datasets 的 features 描述，其中使用了老式/不兼容的
类型名 "List"（而你当前环境的 datasets 版本只认识 LargeList/Sequence 等）。

如果 A（会报错的文件）在 metadata 中能解析到 features 且包含 "_type": "List"，
而 B（正常的文件）没有，或者用的是 "LargeList"/"Sequence"，那基本就能定位根因。

修复思路一般有两类：
  - 数据侧：重写 parquet，去掉/修正 metadata 的 features（用 pyarrow 读表再写回；或用 datasets 重新保存）
  - 环境侧：切换到能识别 "List" 的 datasets 版本（不推荐在训练环境里随意降级，除非你确定兼容性）
"""
    print(textwrap.dedent(msg).strip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="parquet 路径 A（例如 gsm8k_for_ppo/train.parquet）")
    parser.add_argument("--b", required=True, help="parquet 路径 B（例如 gsm8k_good/train.parquet）")
    parser.add_argument(
        "--skip_hf",
        action="store_true",
        help="跳过 datasets.load_dataset 尝试（仅用 pyarrow 对比 parquet 元信息）",
    )
    args = parser.parse_args()

    has_pyarrow = _try_import_pyarrow()
    has_datasets = _try_import_datasets()

    if not has_pyarrow:
        raise SystemExit(
            "缺少依赖 pyarrow。请在你的训练环境/容器里运行（通常已装），或先安装：pip install pyarrow"
        )

    if args.skip_hf:
        # 仅 parquet 对比
        compare(args.a, args.b)
        return

    if not has_datasets:
        print("提示：缺少依赖 datasets，将跳过 load_dataset 尝试（不影响 parquet 元信息对比）。")

        # 复用 compare，但让 datasets 部分不崩溃
        global try_hf_datasets_load  # noqa: PLW0603

        def try_hf_datasets_load(_path: str) -> Dict[str, Any]:  # type: ignore[no-redef]
            return {"ok": False, "error": "datasets not installed", "features": None, "num_rows": None, "column_names": None}

    compare(args.a, args.b)


if __name__ == "__main__":
    main()

