"""PNG 文件结构审计 —— 字节级逐块解析（只读、全内存）。

上传 PNG 后严格按 ``签名(8B) → 长度(4B) → 类型(4B) → 数据(长度) →
CRC(4B)`` 的顺序解析，直到第一个 IEND，逐块列出：

  - 块类型（4 字节 ASCII，非法类型给出原始十六进制）
  - 字节范围（长度字段起点 ~ CRC 之后，半开区间）
  - 数据长度
  - CRC 是否正确（对「类型 + 数据」重算 zlib CRC32 比对）

并专门统计 **IEND 之后的附加字节**：长度、起始偏移、SHA-256 与
十六进制/ASCII 预览——常规图片预览器与多数解码器会静默忽略这部分内容，
它正是图种 / 隐藏载荷常见的藏匿位置；没有尾随内容时明确报告 0 字节。

对以下结构问题逐一给出错误码、所在偏移与明确中文结论，**绝不因为
Pillow 能解码预览就判定结构完整**：

  - 截断（块头不足 8 字节 / 数据或 CRC 越过文件结尾）
  - 越界长度（声明长度把数据或 CRC 指到文件之外）
  - 缺少 IEND（块流正常耗尽却没有 IEND）
  - 重复 IEND（IEND 之后的尾随内容中又出现成帧的 IEND，例如追加第二张 PNG）
  - CRC 错误
  - 首个块不是 IHDR / IHDR 重复 / IEND 数据长度非 0 / 块类型非法

本模块不依赖 Pillow 做结构判定：Pillow 仅用于回答「该文件**能不能**被
常规解码器打开」，并在结论中明确区分「能解码」与「结构完整」。

与水印任务、攻击记录、报告归档、隐私导出完全隔离；上传字节不落盘。
"""

from __future__ import annotations

import hashlib
import io
import struct
import zlib

from PIL import Image

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

# 块头固定为 长度(4) + 类型(4)；块尾固定为 CRC(4)
HEADER_LEN = 8
CRC_LEN = 4

# 尾随内容预览长度（十六进制 + ASCII 各取前 32 字节）
PREVIEW_BYTES = 32

IHDR = b"IHDR"
IEND = b"IEND"


class PngAuditError(ValueError):
    """上传内容不是 PNG（签名不匹配）或为空（调用方返回 400）。"""


# ---------------------------------------------------------------- 工具

def _type_valid(ctype: bytes) -> bool:
    """PNG 块类型必须是 4 个 ASCII 字母（大小写承载 ancillary 等位语义）。"""
    return len(ctype) == 4 and all(65 <= b <= 90 or 97 <= b <= 122 for b in ctype)


def _err(code: str, message: str, offset: int | None = None,
         chunk_index: int | None = None,
         chunk_type: str | None = None, **extra) -> dict:
    e = {
        "code": code,
        "message": message,
        "offset": offset,
        "chunk_index": chunk_index,
        "chunk_type": chunk_type,
    }
    e.update(extra)
    return e


def _hex8(value: int | None) -> str | None:
    return f"{value:08x}" if value is not None else None


def _ascii_preview(blob: bytes) -> str:
    return "".join(chr(b) if 32 <= b <= 126 else "·" for b in blob)


# ---------------------------------------------------------------- 尾随内容扫描

def _scan_trailing(trailing: bytes, base: int,
                   errors: list[dict], warnings: list[dict]) -> None:
    """在 IEND 之后的尾随字节中按块成帧方式寻找重复 IEND / 嵌入 PNG。

    尾随内容不是 PNG 主结构的一部分（可能是任意二进制），因此这里只做
    「能对齐就继续、对不齐就停止」的尽力扫描，发现问题才记录，不因尾随
    内容不成块而报错。
    """
    pos = 0
    n = len(trailing)
    while pos + HEADER_LEN <= n:
        # 允许尾随内容里嵌一个完整的第二张 PNG（图种常见形态）
        if trailing[pos:pos + 8] == PNG_SIGNATURE:
            warnings.append(_err(
                "embedded_png_signature",
                f"IEND 之后偏移 {base + pos} 处发现另一个完整 PNG 签名，"
                "文件可能由多张 PNG 拼接而成（常规预览只显示第一张）。",
                offset=base + pos))
            pos += 8
            continue

        (length,) = struct.unpack(">I", trailing[pos:pos + 4])
        ctype = trailing[pos + 4:pos + 8]
        end = pos + HEADER_LEN + length + CRC_LEN
        if end > n or not _type_valid(ctype):
            # 剩余内容不再符合 PNG 块成帧：到此停止扫描
            break

        if ctype == IEND:
            msg = (f"IEND 之后偏移 {base + pos} 处又出现一个 IEND 块"
                   "（重复 IEND），解码器只会读取到第一个 IEND，"
                   "其后内容在常规图片预览中完全不可见。")
            declared = struct.unpack(">I", trailing[pos + 8 + length:end])[0]
            computed = zlib.crc32(
                ctype + trailing[pos + 8:pos + 8 + length]) & 0xffffffff
            errors.append(_err(
                "duplicate_iend", msg, offset=base + pos,
                crc_valid=(declared == computed)))
        pos = end


# ---------------------------------------------------------------- 解码可行性（仅作旁证）

def _try_decode(data: bytes) -> tuple[bool, str | None, str | None]:
    """用 Pillow 尝试解码，回答「常规预览器能不能打开」，不参与结构判定。"""
    try:
        im = Image.open(io.BytesIO(data))
        im.load()
        fmt = im.format
        im.close()
        return True, fmt, None
    except Exception as e:  # 任何解码失败都只作事实记录
        return False, None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- 主入口

def audit_png(data: bytes) -> dict:
    """对 PNG 字节做结构审计，返回可直接 JSON 序列化的结果字典。

    签名不匹配 / 空文件抛 :class:`PngAuditError`；其余任何结构问题都
    正常返回 200 + 审计报告（``structurally_valid=False``），因为「这份
    文件到底坏在哪」本身就是审计结果。
    """
    size = len(data)
    if size == 0:
        raise PngAuditError("上传内容为空。")
    if size < len(PNG_SIGNATURE) or data[:8] != PNG_SIGNATURE:
        raise PngAuditError(
            "不是 PNG 文件：缺少或不匹配 PNG 签名（89 50 4E 47 0D 0A 1A 0A），"
            "PNG 结构审计仅支持 PNG。")

    errors: list[dict] = []
    warnings: list[dict] = []
    chunks: list[dict] = []

    pos = len(PNG_SIGNATURE)
    iend: dict | None = None
    ihdr_count = 0

    # ------------------------------------------------ 逐块主解析（至首个 IEND）
    while pos < size:
        chunk_index = len(chunks)
        chunk_start = pos

        # 1) 块头：长度 + 类型 至少需要 8 字节
        if pos + HEADER_LEN > size:
            errors.append(_err(
                "truncated_header",
                f"偏移 {pos} 处块头被截断：仅剩 {size - pos} 字节，"
                "不足块头所需的 8 字节（长度 4 + 类型 4），文件不完整。",
                offset=pos, chunk_index=chunk_index))
            break

        (length,) = struct.unpack(">I", data[pos:pos + 4])
        ctype = data[pos + 4:pos + 8]
        type_ok = _type_valid(ctype)
        try:
            type_str = ctype.decode("ascii")
        except UnicodeDecodeError:
            type_str = ctype.decode("latin-1")

        data_start = pos + HEADER_LEN
        data_end = data_start + length
        crc_end = data_end + CRC_LEN

        entry: dict = {
            "index": chunk_index,
            "type": type_str,
            "type_hex": ctype.hex(),
            "type_valid": type_ok,
            "offset": chunk_start,
            "end": crc_end,
            "length_field_offset": chunk_start,
            "type_offset": chunk_start + 4,
            "data_offset": data_start,
            "data_range": [data_start, data_end],
            "data_length": length,
            "crc_offset": data_end,
            "crc_range": [data_end, crc_end],
            "crc_valid": None,
            "crc_declared": None,
            "crc_computed": None,
            "complete": False,
        }

        # 2) 越界长度 / 数据截断：声明范围越过文件结尾
        if data_end > size:
            entry["end"] = size
            chunks.append(entry)
            errors.append(_err(
                "length_out_of_bounds",
                f"偏移 {chunk_start} 处 {type_str if type_ok else '类型非法的'}块"
                f"声明数据长度 {length} 字节，数据需到偏移 {data_end}，"
                f"但文件总长仅 {size} 字节（越界 {data_end - size} 字节），"
                "块数据被截断，后续块无法定位。",
                offset=chunk_start, chunk_index=chunk_index,
                chunk_type=type_str if type_ok else None,
                declared_length=length, available=size - data_start))
            break

        # 3) CRC 截断：数据完整但末尾 4 字节 CRC 不完整
        if crc_end > size:
            computed = zlib.crc32(ctype + data[data_start:data_end]) & 0xffffffff
            entry["crc_computed"] = _hex8(computed)
            entry["end"] = size
            chunks.append(entry)
            errors.append(_err(
                "crc_truncated",
                f"偏移 {chunk_start} 处 {type_str} 块数据完整（{length} 字节），"
                f"但末尾 CRC 被截断（需要偏移 {data_end}~{crc_end}，"
                f"文件仅到 {size}），无法校验完整性。",
                offset=data_end, chunk_index=chunk_index,
                chunk_type=type_str))
            break

        # 4) CRC 校验：覆盖「类型 + 数据」
        declared_crc = struct.unpack(">I", data[data_end:crc_end])[0]
        computed_crc = zlib.crc32(ctype + data[data_start:data_end]) & 0xffffffff
        crc_ok = declared_crc == computed_crc
        entry.update({
            "complete": True,
            "end": crc_end,
            "crc_valid": crc_ok,
            "crc_declared": _hex8(declared_crc),
            "crc_computed": _hex8(computed_crc),
        })
        chunks.append(entry)

        # 5) 结构规则
        if chunk_index == 0 and ctype != IHDR:
            errors.append(_err(
                "missing_ihdr",
                f"首个块（偏移 {chunk_start}）类型为 {type_str}，"
                "PNG 规范要求第一个块必须是 IHDR 图像头。",
                offset=chunk_start, chunk_index=0, chunk_type=type_str))
        if not type_ok:
            errors.append(_err(
                "invalid_type",
                f"偏移 {chunk_start} 处块类型 {ctype.hex()} 不是合法的 4 字节"
                " ASCII 字母组合，不符合 PNG 块类型命名规范。",
                offset=chunk_start + 4, chunk_index=chunk_index,
                chunk_type=None))
        if ctype == IHDR:
            ihdr_count += 1
            if ihdr_count > 1:
                errors.append(_err(
                    "duplicate_ihdr",
                    f"偏移 {chunk_start} 处出现第二个 IHDR 块，"
                    "PNG 规范要求 IHDR 有且仅有一个。",
                    offset=chunk_start, chunk_index=chunk_index,
                    chunk_type="IHDR"))
        if not crc_ok:
            errors.append(_err(
                "crc_error",
                f"偏移 {chunk_start} 处 {type_str} 块 CRC 错误："
                f"文件中为 {declared_crc:08x}，按类型+数据重算应为 "
                f"{computed_crc:08x}，该块数据可能被修改或损坏。",
                offset=data_end, chunk_index=chunk_index,
                chunk_type=type_str,
                crc_declared=_hex8(declared_crc),
                crc_computed=_hex8(computed_crc)))
        if ctype == IEND:
            if length != 0:
                errors.append(_err(
                    "iend_bad_length",
                    f"IEND 块（偏移 {chunk_start}）数据长度应为 0，"
                    f"实际声明为 {length} 字节。",
                    offset=chunk_start, chunk_index=chunk_index,
                    chunk_type="IEND", declared_length=length))
            iend = {
                "present": True,
                "index": chunk_index,
                "offset": chunk_start,
                "end": crc_end,
                "data_length": length,
                "crc_valid": crc_ok,
            }
            break

        pos = crc_end

    # 块流正常耗尽（最后一个块完整结束、文件也恰好结束）却没有 IEND
    fatal_codes = {"truncated_header", "length_out_of_bounds", "crc_truncated"}
    if iend is None and not any(e["code"] in fatal_codes for e in errors):
        errors.append(_err(
            "missing_iend",
            f"块流在偏移 {pos} 正常结束，但未遇到 IEND 结束块，"
            "文件缺少正常的 PNG 结束标记。",
            offset=pos))

    # ------------------------------------------------ IEND 之后的附加字节
    trailing_out: dict | None = None
    if iend is not None:
        trail = bytes(data[iend["end"]:])
        tlen = len(trail)
        trailing_out = {
            "present": tlen > 0,
            "length": tlen,
            "offset": iend["end"],
            "sha256": hashlib.sha256(trail).hexdigest() if tlen > 0 else None,
            "preview_hex": trail[:PREVIEW_BYTES].hex(" ") if tlen > 0 else None,
            "preview_ascii": _ascii_preview(trail[:PREVIEW_BYTES]) if tlen > 0 else None,
            "preview_truncated": tlen > PREVIEW_BYTES,
        }
        if tlen > 0:
            _scan_trailing(trail, iend["end"], errors, warnings)

    # ------------------------------------------------ 可解码性（旁证，不影响判定）
    decodable, decode_format, decode_error = _try_decode(data)

    structurally_valid = not errors
    conclusion = _build_conclusion(
        structurally_valid, decodable, len(chunks), iend, trailing_out, errors)

    return {
        "format": "PNG",
        "bytes": size,
        "signature_ok": True,
        "structurally_valid": structurally_valid,
        "decodable": decodable,
        "decode_format": decode_format,
        "decode_error": decode_error,
        "conclusion": conclusion,
        "errors": errors,
        "warnings": warnings,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "crc_error_count": sum(
            1 for c in chunks if c["crc_valid"] is False),
        "iend": iend or {"present": False, "index": None, "offset": None,
                         "end": None, "data_length": None, "crc_valid": None},
        "trailing": trailing_out,
    }


def _build_conclusion(valid: bool, decodable: bool, chunk_count: int,
                      iend: dict | None, trailing: dict | None,
                      errors: list[dict]) -> str:
    """生成一句话明确结论；核心原则：能解码 ≠ 结构完整。"""
    if valid:
        head = (f"PNG 结构完整：签名正确，共 {chunk_count} 个块，"
                "全部块长度合法、CRC 校验通过，IEND 正常结束")
        if trailing and trailing["length"] > 0:
            return (
                head + "；但 IEND 之后存在 "
                f"{trailing['length']} 字节附加内容（偏移 {trailing['offset']}，"
                "常规图片预览看不到、解码器通常忽略），其 SHA-256 为 "
                f"{trailing['sha256']}，请确认该尾随内容来源是否可信。")
        return head + "，IEND 之后无附加字节（0 字节）。"

    first = errors[0]["message"] if errors else "存在结构问题"
    codes = "、".join(dict.fromkeys(e["code"] for e in errors))
    if decodable:
        return (
            "该文件可以被常规图片解码器打开预览，但 PNG 结构不完整 / 不合法"
            f"（问题类型：{codes}）：{first} "
            "注意：能被解码显示不代表文件结构完整，预览器会跳过或容忍上述问题。")
    return (
        f"PNG 结构损坏且无法正常解码（问题类型：{codes}）：{first}")


__all__ = ["PNG_SIGNATURE", "PngAuditError", "audit_png"]
