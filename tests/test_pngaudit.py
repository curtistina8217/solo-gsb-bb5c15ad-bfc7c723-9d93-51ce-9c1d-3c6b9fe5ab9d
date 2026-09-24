"""PNG 文件结构审计端到端测试。

覆盖：
  - 正常 PNG：逐块类型 / 字节范围（半开区间）/ 数据长度 / CRC 全对，
    IEND 之后明确 0 字节、无 SHA-256
  - IEND 后附加字节：长度与 SHA-256 与实际尾随数据一致
  - 截断（块头不足 / 块数据越界）、长度字段越界（>2^31 或越过 EOF）
  - 缺少 IEND / 重复 IEND（重复块标 after_iend 且计入尾随字节）
  - CRC 篡改：块 crc_ok=false、给出 CRC 偏移与记录/实算值，结论不采信解码
  - 非 PNG 签名（JPEG / 不足 8 字节）返回 200 + 明确审计结论而非崩溃
  - 空文件 400、超 16MB 413
  - 非规范 chunk 类型、IHDR 非首块、缺少 IDAT
  - 全程内存：不建任务目录、不写归档；现有隐私检查 / 导出接口行为不变

运行：
    python -m pytest tests/test_pngaudit.py -q
或：
    python tests/test_pngaudit.py
"""
import hashlib
import io
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-png-audit-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    sys.modules.pop("app.main", None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR
    return TestClient(app), Path(DATA_DIR)


# ---------------------------------------------------------------- 构造工具

PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _chunk(ctype: bytes, payload: bytes = b"") -> bytes:
    crc = zlib.crc32(ctype + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + ctype + payload + struct.pack(">I", crc)


def _ihdr(w: int = 8, h: int = 8) -> bytes:
    payload = struct.pack(">II", w, h) + b"\x08\x02\x00\x00\x00"
    return _chunk(b"IHDR", payload)


def _plain_png(color=(10, 20, 30), size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _plain_jpeg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (5, 6, 7)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _find_chunk(data: bytes, ctype: bytes):
    """返回 (块起始偏移, 长度字段值)；从签名后按 chunk 边界扫描。"""
    pos = 8
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        if data[pos + 4:pos + 8] == ctype:
            return pos, length
        pos += 12 + length
    raise AssertionError(f"chunk {ctype!r} not found")


def _audit(client, data, name="p.png", ctype="image/png"):
    r = client.post("/api/png-audit",
                    files={"image": (name, data, ctype)})
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- 正常文件

def test_valid_png_all_chunks_crc_ok_zero_trailing():
    client, _ = _fresh_client()
    png = _plain_png()
    d = _audit(client, png)
    assert d["signature_ok"] is True
    assert d["ok"] is True
    assert d["bytes"] == len(png)
    types = [c["type"] for c in d["chunks"]]
    assert types == ["IHDR", "IDAT", "IEND"]
    assert all(c["crc_ok"] for c in d["chunks"])
    assert all(not c["after_iend"] for c in d["chunks"])
    assert d["iend_found"] is True and d["iend_count"] == 1
    assert d["iend_offset"] == len(png) - 12
    # 无尾随内容时必须明确为 0 字节，且不给出无意义的空串摘要
    assert d["trailing"] == {
        "offset": len(png), "bytes": 0, "sha256": None}
    assert d["issues"] == [] and d["issue_count"] == 0
    assert "结构完整" in d["conclusion"] and "0 字节" in d["conclusion"]
    assert d["stored"] is False and "job_id" not in d


def test_chunk_ranges_contiguous_and_crc_hex_matches():
    client, _ = _fresh_client()
    png = _plain_png()
    d = _audit(client, png)
    chunks = d["chunks"]
    assert chunks[0]["offset"] == 8  # 首块紧跟 8 字节签名
    for i, c in enumerate(chunks):
        # 半开区间 [offset, end_offset) = 长度4 + 类型4 + 数据 + CRC4
        assert c["byte_range"] == [c["offset"], c["end_offset"]]
        assert c["end_offset"] - c["offset"] == 12 + c["data_length"]
        assert c["data_offset"] == c["offset"] + 8
        assert c["crc_offset"] == c["data_offset"] + c["data_length"]
        assert int(c["crc_stored"], 16) == int(c["crc_computed"], 16)
        if i + 1 < len(chunks):
            assert c["end_offset"] == chunks[i + 1]["offset"]
    # IHDR 数据长度必须为 13
    assert chunks[0]["data_length"] == 13
    # IEND 永远是 0 数据长度，CRC 为规范固定值 AE426082
    iend = chunks[-1]
    assert iend["type"] == "IEND" and iend["data_length"] == 0
    assert iend["crc_stored"] == "0xAE426082"


# ---------------------------------------------------------------- IEND 后尾随

def test_trailing_bytes_length_and_sha256():
    client, _ = _fresh_client()
    png = _plain_png()
    extra = b"SECRET-APPENDED-PAYLOAD-\x00\xff"
    d = _audit(client, png + extra)
    assert d["ok"] is False
    t = d["trailing"]
    assert t["offset"] == len(png)
    assert t["bytes"] == len(extra)
    assert t["sha256"] == hashlib.sha256(extra).hexdigest()
    kinds = [i["kind"] for i in d["issues"]]
    assert kinds == ["trailing_bytes"]
    issue = d["issues"][0]
    assert issue["offset"] == len(png)
    assert str(len(extra)) in issue["message"]
    assert str(len(extra)) in d["conclusion"]
    assert d["iend_found"] is True


def test_no_trailing_means_explicit_zero_not_omitted():
    client, _ = _fresh_client()
    d = _audit(client, _plain_png())
    assert "trailing" in d and d["trailing"]["bytes"] == 0
    assert d["trailing"]["sha256"] is None


def test_duplicate_iend_reported_as_chunk_and_trailing():
    client, _ = _fresh_client()
    png = _plain_png()
    second_iend = _chunk(b"IEND", b"")
    d = _audit(client, png + second_iend)
    assert d["ok"] is False
    assert d["iend_count"] == 2
    dup = next(i for i in d["issues"] if i["kind"] == "duplicate_iend")
    assert dup["offset"] == len(png)  # 重复 IEND 的偏移
    last = d["chunks"][-1]
    assert last["type"] == "IEND" and last["after_iend"] is True
    assert last["offset"] == len(png) and last["crc_ok"] is True
    # 重复 IEND 本身也计入 IEND 之后的附加字节
    t = d["trailing"]
    assert t["bytes"] == 12
    assert t["sha256"] == hashlib.sha256(second_iend).hexdigest()
    assert "重复 IEND" in d["conclusion"]


def test_trailing_chunk_chain_listed_with_after_iend_flag():
    """IEND 后拼接合法 chunk：逐块列出并标 after_iend，整体仍计入尾随。"""
    client, _ = _fresh_client()
    png = _plain_png()
    extra = _chunk(b"tEXt", b"note\x00hidden") + _chunk(b"IEND", b"")
    d = _audit(client, png + extra)
    after = [c for c in d["chunks"] if c["after_iend"]]
    assert [c["type"] for c in after] == ["tEXt", "IEND"]
    assert all(c["crc_ok"] for c in after)
    assert d["trailing"]["bytes"] == len(extra)
    assert d["trailing"]["sha256"] == hashlib.sha256(extra).hexdigest()
    assert any(i["kind"] == "duplicate_iend" for i in d["issues"])


# ---------------------------------------------------------------- 截断 / 越界

def test_truncated_mid_chunk_reports_offset_and_incomplete_conclusion():
    client, _ = _fresh_client()
    png = _plain_png()
    idat_off, idat_len = _find_chunk(png, b"IDAT")
    cut = idat_off + 8 + 5  # 只保留 IDAT 前 5 字节数据，CRC 与其余数据丢失
    d = _audit(client, png[:cut])
    assert d["ok"] is False
    first = d["issues"][0]
    assert first["kind"] == "length_out_of_bounds"
    assert first["offset"] == idat_off
    assert str(idat_len) in first["message"]
    assert any(i["kind"] == "missing_iend" for i in d["issues"])
    assert d["iend_found"] is False
    assert "结构不完整" in d["conclusion"]


def test_truncated_chunk_header_under_eight_bytes():
    client, _ = _fresh_client()
    png = _plain_png()
    stray_off = len(png) - 12  # 去掉 IEND
    d = _audit(client, png[:-12] + b"\x00\x01\x02")
    kinds = [i["kind"] for i in d["issues"]]
    assert "truncated_chunk_header" in kinds
    hdr = next(i for i in d["issues"]
               if i["kind"] == "truncated_chunk_header")
    assert hdr["offset"] == stray_off
    assert "missing_iend" in kinds


def test_missing_iend_alone():
    client, _ = _fresh_client()
    png = _plain_png()
    d = _audit(client, png[:-12])  # 仅删除末尾 IEND
    assert d["ok"] is False
    issue = next(i for i in d["issues"] if i["kind"] == "missing_iend")
    assert issue["offset"] == len(png) - 12
    assert d["iend_found"] is False
    assert "IEND" in d["conclusion"]


def test_length_field_over_spec_limit_and_over_eof():
    client, _ = _fresh_client()
    png = _plain_png()

    # 1) 长度字段最高位置 1（>2^31-1）：违反 PNG 规范
    bad1 = bytearray(png)
    bad1[8:12] = struct.pack(">I", 0x80000000)
    d1 = _audit(client, bytes(bad1))
    i1 = next(i for i in d1["issues"] if i["kind"] == "length_out_of_bounds")
    assert i1["offset"] == 8 and "2147483647" in i1["message"]
    assert d1["ok"] is False

    # 2) 声明一个越过文件末尾的长度
    bad2 = bytearray(png)
    bad2[8:12] = struct.pack(">I", 0x00100000)
    d2 = _audit(client, bytes(bad2))
    i2 = next(i for i in d2["issues"] if i["kind"] == "length_out_of_bounds")
    assert i2["offset"] == 8 and "越过文件末尾" in i2["message"]


# ---------------------------------------------------------------- CRC

def test_crc_mismatch_after_data_tamper():
    client, _ = _fresh_client()
    png = _plain_png()
    idat_off, _ = _find_chunk(png, b"IDAT")
    data_off = idat_off + 8

    tampered = bytearray(png)
    tampered[data_off] ^= 0xFF  # 翻转 IDAT 数据一个字节
    d = _audit(client, bytes(tampered))
    assert d["signature_ok"] is True and d["iend_found"] is True
    assert d["ok"] is False
    issue = next(i for i in d["issues"] if i["kind"] == "crc_mismatch")
    assert issue["offset"] == data_off + _find_chunk(png, b"IDAT")[1]
    assert "IDAT" in issue["message"]
    idat = next(c for c in d["chunks"] if c["type"] == "IDAT")
    assert idat["crc_ok"] is False
    assert idat["crc_stored"] != idat["crc_computed"]
    assert int(idat["crc_computed"], 16) == zlib.crc32(
        b"IDAT" + bytes(tampered[data_off:data_off + idat["data_length"]]
                        )) & 0xFFFFFFFF
    assert "CRC" in d["conclusion"]
    # 尾随仍为 0：问题仅在 CRC，不能误报尾随
    assert d["trailing"]["bytes"] == 0


def test_decodable_but_structurally_bad_is_not_called_ok():
    """能被 Pillow 正常解码不等于结构完整：带尾随内容的 PNG 仍判为有问题。"""
    client, _ = _fresh_client()
    png = _plain_png()
    # 追加内容后 Pillow 仍可打开（解码器忽略 IEND 之后字节）
    Image.open(io.BytesIO(png + b"IGNORED-BY-PREVIEW")).verify()
    d = _audit(client, png + b"IGNORED-BY-PREVIEW")
    assert d["ok"] is False
    assert d["trailing"]["bytes"] == len(b"IGNORED-BY-PREVIEW")


# ---------------------------------------------------------------- 签名 / 规范

def test_non_png_and_short_file_return_audit_conclusion():
    client, _ = _fresh_client()
    d = _audit(client, _plain_jpeg(), name="p.jpg", ctype="image/jpeg")
    assert d["signature_ok"] is False and d["ok"] is False
    assert d["chunks"] == [] and d["iend_found"] is False
    assert "不是 PNG" in d["conclusion"]
    assert d["issues"][0]["kind"] == "bad_signature"

    d2 = _audit(client, b"abc", name="short.bin", ctype="application/octet-stream")
    assert d2["signature_ok"] is False
    assert "8 字节" in d2["conclusion"]


def test_invalid_chunk_type_still_parsed_and_flagged():
    client, _ = _fresh_client()
    data = (PNG_SIG + _ihdr() + _chunk(b"A\x01BC", b"")
            + _chunk(b"IDAT", b"x") + _chunk(b"IEND", b""))
    d = _audit(client, data)
    bad = next(c for c in d["chunks"] if not c["type_valid"])
    assert bad["type"].startswith("A") and bad["crc_ok"] is True
    issue = next(i for i in d["issues"] if i["kind"] == "invalid_chunk_type")
    assert issue["offset"] == bad["offset"] + 4
    assert d["ok"] is False


def test_ihdr_not_first_and_missing_idat():
    client, _ = _fresh_client()
    # tEXt 出现在 IHDR 之前
    d1 = _audit(client, PNG_SIG + _chunk(b"tEXt", b"k\x00v") + _ihdr()
                + _chunk(b"IDAT", b"x") + _chunk(b"IEND", b""))
    issue = next(i for i in d1["issues"] if i["kind"] == "ihdr_not_first")
    assert issue["offset"] == 8

    # 有 IHDR / IEND 但没有任何 IDAT
    d2 = _audit(client, PNG_SIG + _ihdr() + _chunk(b"IEND", b""))
    assert any(i["kind"] == "missing_idat" for i in d2["issues"])
    assert d2["trailing"]["bytes"] == 0


# ---------------------------------------------------------------- 接口边界 / 隔离

def test_empty_file_400():
    client, _ = _fresh_client()
    r = client.post("/api/png-audit",
                    files={"image": ("empty.png", b"", "image/png")})
    assert r.status_code == 400


def test_upload_over_16mb_returns_413():
    client, _ = _fresh_client()
    over = PNG_SIG + b"\x00" * (16 * 1024 * 1024 + 1 - len(PNG_SIG))
    assert len(over) == 16 * 1024 * 1024 + 1
    r = client.post("/api/png-audit",
                    files={"image": ("big.png", over, "image/png")})
    assert r.status_code == 413


def test_no_persistence_jobs_or_archive():
    client, data_dir = _fresh_client()
    for payload in (_plain_png(), _plain_png() + b"TRAIL",
                    _plain_png()[:-12], _plain_jpeg()):
        r = client.post("/api/png-audit",
                        files={"image": ("p.png", payload, "image/png")})
        assert r.status_code == 200
    assert not list(data_dir.glob("*/meta.json"))
    assert not (data_dir / "archive.log.jsonl").exists()
    assert [p.name for p in data_dir.iterdir()] == []


def test_existing_privacy_endpoints_unchanged():
    client, _ = _fresh_client()
    # 只读隐私检查与结构审计互不影响
    r = client.post("/api/privacy-check",
                    files={"image": ("p.jpg", _plain_jpeg(), "image/jpeg")})
    assert r.status_code == 200 and r.json()["format"] == "JPEG"
    r2 = client.post("/api/privacy-export",
                     files={"image": ("p.png", _plain_png(), "image/png")})
    assert r2.status_code == 200 and r2.json()["recheck"]["passed"] is True
    r3 = client.post("/api/png-audit",
                     files={"image": ("p.png", _plain_png(), "image/png")})
    assert r3.status_code == 200 and r3.json()["ok"] is True


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个 PNG 结构审计测试全部通过 ✔")
