"""PNG 文件结构审计 —— 字节级解析签名与 chunk 序列，发现预览看不到的问题。

按 PNG 规范依次解析：

  8 字节签名 → 若干 chunk（4 字节长度 | 4 字节类型 | 数据 | 4 字节 CRC32）
  → IEND 结束块 → （规范之外）IEND 之后的附加字节。

输出（纯内存、只读，只用标准库，不依赖 Pillow 解码）：

  - 逐块列出：类型、字节范围（半开区间）、数据长度、CRC 校验是否通过；
  - IEND 之后附加字节的长度与 SHA-256（普通图片预览看不到的尾随内容，
    没有尾随内容时明确为 0 字节）；
  - 截断 / 越界长度 / 缺少或重复 IEND / CRC 错误等问题的**偏移**与结论。

设计原则：
  - 结构结论只依据字节级解析——「能被解码器打开」不等于「结构完整」；
  - 与水印任务、攻击记录、报告归档完全隔离：不写盘、不建任务。
"""

from __future__ import annotations

import hashlib
import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# PNG 规范：chunk 长度字段是 31 位无符号整数（最高位必须为 0）
MAX_CHUNK_LENGTH = 0x7FFFFFFF

# 结构性错误（数据流本身不完整，无法解析到 IEND）
_STRUCTURAL_KINDS = {
    "bad_signature",
    "truncated_chunk_header",
    "length_out_of_bounds",
    "missing_iend",
}


def _issue(kind: str, offset: int, message: str) -> dict:
    return {"kind": kind, "offset": int(offset), "message": message}


def _type_text(raw: bytes) -> tuple[str, bool]:
    """chunk 类型 4 字节 -> (可展示文本, 是否为规范的 ASCII 字母类型)。

    PNG 规范要求类型为 4 个 ASCII 字母（大小写位承载语义标志）；
    非字母字节不阻断解析，但记为结构问题。
    """
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        text = "".join(chr(b) if 32 <= b < 127 else f"\\x{b:02x}" for b in raw)
        return text, False
    valid = all(("A" <= c <= "Z") or ("a" <= c <= "z") for c in text)
    return text, valid


def audit_png(data: bytes) -> dict:
    """对 PNG 字节做结构审计，返回可直接序列化为 JSON 的结果字典。

    任何输入（包括非 PNG 字节）都返回结构化结论，不抛异常；
    调用方负责空文件 / 大小限制等输入级校验。
    """
    total = len(data)
    issues: list[dict] = []
    chunks: list[dict] = []

    # ---------------------------------------------------------- 1) 签名
    if total < len(PNG_SIGNATURE):
        issues.append(_issue(
            "bad_signature", 0,
            f"文件仅 {total} 字节，不足 PNG 签名所需的 8 字节。"))
    elif data[:len(PNG_SIGNATURE)] != PNG_SIGNATURE:
        found = " ".join(f"{b:02X}" for b in data[:8])
        issues.append(_issue(
            "bad_signature", 0,
            "文件头 8 字节与 PNG 签名不匹配"
            f"（期望 89 50 4E 47 0D 0A 1A 0A，实际 {found}）。"))

    iend_offset: int | None = None
    iend_count = 0
    first_iend_end: int | None = None

    if not issues:
        # -------------------------------------------------- 2) 逐块解析
        pos = len(PNG_SIGNATURE)

        while pos < total:
            after_iend = first_iend_end is not None
            remaining = total - pos
            if remaining < 8:
                if not after_iend:
                    issues.append(_issue(
                        "truncated_chunk_header", pos,
                        f"偏移 {pos} 处仅剩 {remaining} 字节，不足一个 chunk 头"
                        "（4 字节长度 + 4 字节类型），数据被截断。"))
                # IEND 之后的任意字节统一按原始尾随内容报告，停止尽力解析
                break

            (length,) = struct.unpack(">I", data[pos:pos + 4])
            type_raw = data[pos + 4:pos + 8]
            type_text, type_valid = _type_text(type_raw)
            label = type_text if type_valid else f"{type_text}（非规范类型）"

            if length > MAX_CHUNK_LENGTH:
                if after_iend:
                    break  # 尾随区域不是合法 chunk 序列：按原始尾随字节报告
                # 长度字段越界：后续边界不可信，停止解析
                issues.append(_issue(
                    "length_out_of_bounds", pos,
                    f"偏移 {pos} 处的 chunk {label} 声明长度 {length}，"
                    f"超过 PNG 规范上限 {MAX_CHUNK_LENGTH}（长度字段最高位必须为 0）。"))
                break
            end = pos + 12 + length
            if end > total:
                if after_iend:
                    break  # 尾随区域不是合法 chunk 序列：按原始尾随字节报告
                issues.append(_issue(
                    "length_out_of_bounds", pos,
                    f"偏移 {pos} 处的 chunk {label} 声明数据长度 {length} 字节，"
                    f"越过文件末尾（块结束需偏移 {end}，文件共 {total} 字节），"
                    "数据被截断或长度字段被篡改。"))
                break

            data_offset = pos + 8
            crc_offset = data_offset + length
            (crc_stored,) = struct.unpack(">I", data[crc_offset:crc_offset + 4])
            crc_computed = zlib.crc32(type_raw) & 0xFFFFFFFF
            crc_computed = zlib.crc32(data[data_offset:crc_offset],
                                      crc_computed) & 0xFFFFFFFF
            crc_ok = crc_stored == crc_computed

            if not type_valid:
                issues.append(_issue(
                    "invalid_chunk_type", pos + 4,
                    f"偏移 {pos + 4} 处的 chunk 类型 {type_text!r} 含非 ASCII 字母字节，"
                    "不符合 PNG 规范（类型必须为 4 个字母）。"))
            if not crc_ok:
                issues.append(_issue(
                    "crc_mismatch", crc_offset,
                    f"chunk {label}（数据偏移 {data_offset}，长度 {length}）CRC 校验失败："
                    f"文件记录 0x{crc_stored:08X}，按类型+数据实算 0x{crc_computed:08X}。"))

            chunks.append({
                "index": len(chunks),
                "type": type_text,
                "type_valid": type_valid,
                "offset": pos,
                "end_offset": end,
                "byte_range": [pos, end],
                "data_offset": data_offset,
                "data_length": length,
                "crc_offset": crc_offset,
                "crc_stored": f"0x{crc_stored:08X}",
                "crc_computed": f"0x{crc_computed:08X}",
                "crc_ok": crc_ok,
                "after_iend": after_iend,
            })

            if type_raw == b"IEND":
                iend_count += 1
                if first_iend_end is None:
                    iend_offset = pos
                    first_iend_end = end
                else:
                    issues.append(_issue(
                        "duplicate_iend", pos,
                        f"偏移 {pos} 处出现重复的 IEND 结束块"
                        f"（首个 IEND 在偏移 {iend_offset}），规范只允许一个。"))
            pos = end

    # ---------------------------------------------------------- 3) 整体结构检查
    if first_iend_end is None:
        issues.append(_issue(
            "missing_iend", total,
            f"解析到文件末尾（偏移 {total}）仍未遇到 IEND 结束块，"
            "PNG 数据流不完整。"))

    pre_iend = [c for c in chunks if not c["after_iend"]]
    if pre_iend and pre_iend[0]["type"] != "IHDR":
        issues.append(_issue(
            "ihdr_not_first", pre_iend[0]["offset"],
            f"首个 chunk 是 {pre_iend[0]['type']}（偏移 {pre_iend[0]['offset']}），"
            "PNG 规范要求 IHDR 必须是第一个 chunk。"))
    if chunks and not any(c["type"] == "IHDR" for c in pre_iend):
        issues.append(_issue(
            "missing_ihdr", len(PNG_SIGNATURE),
            "未找到 IHDR 头块，PNG 数据流缺少必需的图像头。"))
    if first_iend_end is not None and not any(
            c["type"] == "IDAT" for c in pre_iend):
        issues.append(_issue(
            "missing_idat", len(PNG_SIGNATURE),
            "未找到 IDAT 数据块，PNG 数据流不含任何图像数据。"))

    # IEND 之后的全部字节（无论是否能按 chunk 解析）都计入尾随内容
    if first_iend_end is not None and total > first_iend_end:
        issues.append(_issue(
            "trailing_bytes", first_iend_end,
            f"IEND（偏移 {iend_offset}）之后存在 {total - first_iend_end} 字节"
            "附加数据，普通图片预览看不到这部分内容。"))

    # ---------------------------------------------------------- 4) 尾随字节
    if first_iend_end is not None:
        tail = data[first_iend_end:]
        trailing = {
            "offset": first_iend_end,
            "bytes": len(tail),
            # 没有尾随内容时明确为 0 字节、不给出无意义的空串摘要
            "sha256": hashlib.sha256(tail).hexdigest() if tail else None,
        }
    else:
        trailing = {"offset": None, "bytes": 0, "sha256": None}

    signature_ok = not any(i["kind"] == "bad_signature" for i in issues)
    conclusion = _conclusion(issues, chunks, iend_offset, trailing)
    return {
        "signature_ok": signature_ok,
        "bytes": total,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "iend_found": first_iend_end is not None,
        "iend_offset": iend_offset,
        "iend_count": iend_count,
        "trailing": trailing,
        "issues": issues,
        "issue_count": len(issues),
        "ok": signature_ok and not issues,
        "conclusion": conclusion,
    }


def _conclusion(issues: list[dict], chunks: list[dict],
                iend_offset: int | None, trailing: dict) -> str:
    """由问题清单生成一句明确的总体结论。"""
    structural = [i for i in issues if i["kind"] in _STRUCTURAL_KINDS]
    bad_sig = [i for i in issues if i["kind"] == "bad_signature"]
    if bad_sig:
        return "不是 PNG 文件：" + bad_sig[0]["message"]
    if structural:
        # 结构性错误：数据流不完整，不能把能解码当成结构完整
        return "PNG 结构不完整：" + structural[0]["message"]

    # 至此数据流可完整解析至 IEND；汇总 CRC / IEND / 类型 / 尾随等问题
    parts: list[str] = []
    crc_n = sum(1 for i in issues if i["kind"] == "crc_mismatch")
    if crc_n:
        first_crc = next(i for i in issues if i["kind"] == "crc_mismatch")
        parts.append(f"{crc_n} 个 chunk CRC 校验失败（首个在偏移 {first_crc['offset']}）")
    if any(i["kind"] == "duplicate_iend" for i in issues):
        parts.append("存在重复 IEND 结束块")
    if any(i["kind"] in ("invalid_chunk_type", "ihdr_not_first",
                         "missing_ihdr", "missing_idat") for i in issues):
        parts.append("存在不符合 PNG 规范的 chunk 排布（详见下方逐条问题）")
    if trailing["bytes"] > 0:
        parts.append(f"IEND 之后存在 {trailing['bytes']} 字节附加数据"
                     f"（SHA-256 {trailing['sha256'][:16]}…）")

    if not parts:
        return (f"PNG 结构完整：{len(chunks)} 个 chunk 全部通过 CRC 校验，"
                f"IEND（偏移 {iend_offset}）之后无附加字节（0 字节）。")
    return ("PNG 数据流可解析至 IEND，但" + "；".join(parts) +
            "。能被图片预览器正常显示不等于结构完整，请逐条核对下方问题。")


__all__ = [
    "PNG_SIGNATURE",
    "MAX_CHUNK_LENGTH",
    "audit_png",
]
