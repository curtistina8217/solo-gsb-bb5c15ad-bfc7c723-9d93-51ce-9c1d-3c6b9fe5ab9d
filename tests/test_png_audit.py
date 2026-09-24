"""PNG 文件结构审计端到端测试。

覆盖：
  - 合法 PNG：签名 / 逐块字段（类型、字节范围、数据长度、CRC 全部正确）
  - IEND 之后 0 字节时明确报告零字节
  - 尾随内容：长度 / 偏移 / SHA-256 / 预览正确
  - 追加第二张 PNG / 裸 IEND：重复 IEND（能解码但结构不完整）
  - CRC 错误：定位到块偏移与重算值
  - 截断（块头）/ 越界长度 / CRC 截断 / 缺少 IEND
  - 首块非 IHDR / IHDR 重复 / IEND 长度非 0 / 块类型非法
  - 非 PNG / 空文件 400，超过 16MB 413
  - 只读：不落盘、不建任务、不影响归档与隐私导出

运行：
    python -m pytest tests/test_png_audit.py -q
或：
    python tests/test_png_audit.py
"""
import hashlib
import io
import os
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------- 构造工具

def _plain_png(size=96) -> bytes:
    arr = np.random.RandomState(3).randint(0, 255, (size, size, 3), np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _chunk(ctype: bytes, payload: bytes) -> bytes:
    return (struct.pack(">I", len(payload)) + ctype + payload
            + struct.pack(">I", zlib.crc32(ctype + payload) & 0xffffffff))


def _split_chunks(png: bytes):
    """按审计器相同的布局切出 (start, end, ctype, length)。"""
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    out, pos = [], 8
    while pos + 12 <= len(png):
        (length,) = struct.unpack(">I", png[pos:pos + 4])
        ctype = png[pos + 4:pos + 8]
        end = pos + 12 + length
        out.append((pos, end, ctype, length))
        pos = end
        if ctype == b"IEND":
            break
    return out


def _fresh_client():
    tmp = tempfile.mkdtemp(prefix="wm-pngaudit-")
    os.environ["WM_DATA_DIR"] = tmp
    os.environ["WM_MAX_JOBS"] = "200"
    sys.modules.pop("app.main", None)
    from fastapi.testclient import TestClient
    from app.main import app, DATA_DIR

    return TestClient(app), Path(DATA_DIR)


def _audit(client, png: bytes, name="p.png"):
    return client.post(
        "/api/png-audit",
        files={"image": (name, png, "image/png")})


# ---------------------------------------------------------------- 正常用例

def test_valid_png_chunks_and_zero_trailing():
    client, _ = _fresh_client()
    png = _plain_png()
    r = _audit(client, png)
    assert r.status_code == 200, r.text
    d = r.json()

    assert d["signature_ok"] is True
    assert d["structurally_valid"] is True
    assert d["decodable"] is True
    assert d["decode_format"] == "PNG"
    assert d["errors"] == []
    assert d["bytes"] == len(png)
    assert d["stored"] is False

    # 逐块字段
    types = [c["type"] for c in d["chunks"]]
    assert types[0] == "IHDR"
    assert types[-1] == "IEND"
    assert "IDAT" in types

    expected = _split_chunks(png)
    assert len(d["chunks"]) == len(expected)
    for c, (start, end, ctype, length) in zip(d["chunks"], expected):
        assert c["offset"] == start
        assert c["end"] == end
        assert c["type"] == ctype.decode()
        assert c["data_length"] == length
        assert c["data_range"] == [start + 8, start + 8 + length]
        assert c["crc_range"] == [start + 8 + length, end]
        assert c["crc_valid"] is True
        assert c["complete"] is True
        assert c["type_valid"] is True

    # IHDR 数据长度固定 13；IEND 数据长度固定 0
    assert d["chunks"][0]["data_length"] == 13
    assert d["chunks"][-1]["data_length"] == 0

    # IEND 信息
    ie = d["iend"]
    assert ie["present"] is True
    assert ie["index"] == len(d["chunks"]) - 1
    assert ie["crc_valid"] is True

    # 没有尾随内容时必须明确显示 0 字节
    t = d["trailing"]
    assert t["present"] is False
    assert t["length"] == 0
    assert t["sha256"] is None
    assert t["offset"] == ie["end"]
    assert "0 字节" in d["conclusion"]


def test_trailing_bytes_length_sha256_and_preview():
    client, _ = _fresh_client()
    png = _plain_png()
    payload = b"SECRET-TRAILING-PAYLOAD-\x00\x01\x02"
    r = _audit(client, png + payload)
    assert r.status_code == 200, r.text
    d = r.json()

    # PNG 主结构本身合法；尾随内容单独高亮
    assert d["structurally_valid"] is True
    assert d["decodable"] is True
    t = d["trailing"]
    assert t["present"] is True
    assert t["length"] == len(payload)
    assert t["sha256"] == hashlib.sha256(payload).hexdigest()
    assert t["preview_hex"].startswith("53 45 43 52")
    assert "SECRET" in t["preview_ascii"]
    assert t["preview_truncated"] is False
    assert "附加内容" in d["conclusion"]
    assert t["sha256"] in d["conclusion"]
    # IEND 结束 + 尾随长度 = 文件总长
    assert t["offset"] + t["length"] == d["bytes"]


def test_trailing_preview_truncated_for_long_payload():
    client, _ = _fresh_client()
    payload = bytes((i * 7) % 256 for i in range(320))
    d = _audit(client, _plain_png() + payload).json()
    assert d["trailing"]["preview_truncated"] is True
    # 预览只取前 32 字节（hex 中 31 个空格）
    assert d["trailing"]["preview_hex"].count(" ") == 31


def test_duplicate_iend_from_second_png_decodable_but_invalid():
    client, _ = _fresh_client()
    # 图种形态：两张 PNG 直接拼接
    d = _audit(client, _plain_png(64) + _plain_png(48)).json()
    assert d["structurally_valid"] is False
    codes = [e["code"] for e in d["errors"]]
    assert "duplicate_iend" in codes
    dup = next(e for e in d["errors"] if e["code"] == "duplicate_iend")
    assert dup["offset"] > 0
    # 常规解码器只看第一张：能解码，但这不代表结构完整
    assert d["decodable"] is True
    assert "能被解码" in d["conclusion"] or "能解码" in d["conclusion"]
    # 嵌入的第二张 PNG 签名有警告
    assert any(w["code"] == "embedded_png_signature" for w in d["warnings"])
    assert d["trailing"]["length"] > 0


def test_bare_iend_appended_is_duplicate():
    client, _ = _fresh_client()
    png = _plain_png()
    bare_iend = _chunk(b"IEND", b"")
    d = _audit(client, png + bare_iend).json()
    assert d["structurally_valid"] is False
    dups = [e for e in d["errors"] if e["code"] == "duplicate_iend"]
    assert len(dups) == 1
    assert dups[0]["offset"] == d["iend"]["end"]
    assert dups[0]["crc_valid"] is True


# ---------------------------------------------------------------- CRC / 截断

def test_crc_error_localized():
    client, _ = _fresh_client()
    png = bytearray(_plain_png())
    chunks = _split_chunks(bytes(png))
    # 找第一个 IDAT，翻转其数据首字节（CRC 必然不匹配）
    start, end, ctype, length = next(c for c in chunks if c[2] == b"IDAT")
    png[start + 8] ^= 0xFF
    d = _audit(client, bytes(png)).json()

    bad = [c for c in d["chunks"] if c["crc_valid"] is False]
    assert len(bad) == 1 and bad[0]["type"] == "IDAT"
    errs = [e for e in d["errors"] if e["code"] == "crc_error"]
    assert len(errs) == 1
    assert errs[0]["chunk_type"] == "IDAT"
    assert errs[0]["offset"] == start + 8 + length  # 指向 CRC 字段
    assert errs[0]["crc_declared"] == bad[0]["crc_declared"]
    assert errs[0]["crc_computed"] == bad[0]["crc_computed"]
    assert errs[0]["crc_declared"] != errs[0]["crc_computed"]
    assert d["crc_error_count"] == 1
    assert d["structurally_valid"] is False


def test_truncated_chunk_header():
    client, _ = _fresh_client()
    png = _plain_png()
    chunks = _split_chunks(png)
    idat = next(c for c in chunks if c[2] == b"IDAT")
    cut = idat[0] + 3  # 块头 8 字节只剩 3 字节
    d = _audit(client, png[:cut]).json()
    assert d["structurally_valid"] is False
    errs = [e for e in d["errors"] if e["code"] == "truncated_header"]
    assert len(errs) == 1
    assert errs[0]["offset"] == idat[0]
    assert "8 字节" in errs[0]["message"]
    assert d["iend"]["present"] is False
    assert d["trailing"] is None


def test_out_of_bounds_data_length():
    client, _ = _fresh_client()
    png = _plain_png()
    chunks = _split_chunks(png)
    idat = next(c for c in chunks if c[2] == b"IDAT")
    cut = idat[0] + 8 + 5  # 数据只留 5 字节
    d = _audit(client, png[:cut]).json()
    errs = [e for e in d["errors"] if e["code"] == "length_out_of_bounds"]
    assert len(errs) == 1
    assert errs[0]["offset"] == idat[0]
    assert d["structurally_valid"] is False


def test_huge_declared_length_reported_as_oob():
    client, _ = _fresh_client()
    # 签名后立刻声明一个 0xFFFFFFFF 长度的块
    crafted = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 0xFFFFFFFF) + b"abCD"
    r = _audit(client, crafted)
    assert r.status_code == 200, r.text
    d = r.json()
    errs = [e for e in d["errors"] if e["code"] == "length_out_of_bounds"]
    assert len(errs) == 1
    assert errs[0]["offset"] == 8
    assert errs[0]["declared_length"] == 0xFFFFFFFF
    # 不能真的尝试分配那么大的内存（能跑到这里即说明按声明值比较、未切片分配）
    assert d["bytes"] == 16


def test_truncated_crc_after_complete_data():
    client, _ = _fresh_client()
    png = _plain_png()
    chunks = _split_chunks(png)
    start, end, ctype, length = next(c for c in chunks if c[2] == b"IDAT")
    data_end = start + 8 + length
    cut = data_end + 2  # 数据完整，CRC 4 字节只剩 2
    d = _audit(client, png[:cut]).json()
    errs = [e for e in d["errors"] if e["code"] == "crc_truncated"]
    assert len(errs) == 1
    assert errs[0]["offset"] == data_end
    assert d["structurally_valid"] is False


def test_missing_iend_when_stream_runs_out():
    client, _ = _fresh_client()
    png = _plain_png()
    chunks = _split_chunks(png)
    iend = next(c for c in chunks if c[2] == b"IEND")
    d = _audit(client, png[:iend[0]]).json()  # 整个 IEND 块被切掉
    errs = [e for e in d["errors"] if e["code"] == "missing_iend"]
    assert len(errs) == 1
    assert d["iend"]["present"] is False
    # Pillow 往往容忍缺失 IEND 仍能解码像素——这正是「能解码 ≠ 结构完整」
    assert d["structurally_valid"] is False
    assert "结构" in d["conclusion"]


# ---------------------------------------------------------------- 规范违规

def test_first_chunk_not_ihdr():
    client, _ = _fresh_client()
    text = _chunk(b"tEXt", b"Comment\x00hi")
    png = _plain_png()
    crafted = png[:8] + text + png[8:]
    d = _audit(client, crafted).json()
    errs = [e for e in d["errors"] if e["code"] == "missing_ihdr"]
    assert len(errs) == 1
    assert errs[0]["offset"] == 8
    assert d["structurally_valid"] is False


def test_duplicate_ihdr():
    client, _ = _fresh_client()
    png = _plain_png()
    chunks = _split_chunks(png)
    ihdr = chunks[0]
    ihdr_bytes = png[ihdr[0]:ihdr[1]]
    crafted = png[:ihdr[1]] + ihdr_bytes + png[ihdr[1]:]
    d = _audit(client, crafted).json()
    errs = [e for e in d["errors"] if e["code"] == "duplicate_ihdr"]
    assert len(errs) == 1
    assert errs[0]["offset"] == ihdr[1]
    assert d["structurally_valid"] is False


def test_iend_with_nonzero_data_length():
    client, _ = _fresh_client()
    png = bytearray(_plain_png())
    chunks = _split_chunks(bytes(png))
    iend = next(c for c in chunks if c[2] == b"IEND")
    # 把 IEND 长度从 0 改为 4，并在 CRC 前插入 4 字节、重算 CRC
    payload = b"\x00" * 4
    new_iend = _chunk(b"IEND", payload)
    crafted = bytes(png[:iend[0]]) + new_iend
    d = _audit(client, crafted).json()
    errs = [e for e in d["errors"] if e["code"] == "iend_bad_length"]
    assert len(errs) == 1
    assert errs[0]["declared_length"] == 4
    assert d["iend"]["data_length"] == 4
    assert d["structurally_valid"] is False


def test_invalid_chunk_type():
    client, _ = _fresh_client()
    bad = _chunk(b"xy!1", b"hello")  # '!' '1' 不是字母
    png = _plain_png()
    crafted = png[:8] + bad + png[8:]
    d = _audit(client, crafted).json()
    errs = [e for e in d["errors"] if e["code"] == "invalid_type"]
    assert len(errs) == 1
    assert errs[0]["offset"] == 12  # 8 签名 + 4 长度
    flagged = next(c for c in d["chunks"] if c["type_valid"] is False)
    assert flagged["type_hex"] == b"xy!1".hex()
    assert d["structurally_valid"] is False
    # CRC 仍按原始字节正常校验（这里构造时 CRC 是对的）
    assert flagged["crc_valid"] is True


# ---------------------------------------------------------------- 输入限制 / 只读

def test_non_png_rejected_400():
    client, _ = _fresh_client()
    buf = io.BytesIO()
    Image.new("RGB", (32, 32)).save(buf, format="JPEG")
    r = client.post("/api/png-audit",
                    files={"image": ("a.jpg", buf.getvalue(), "image/jpeg")})
    assert r.status_code == 400
    assert "PNG 签名" in r.json()["detail"]

    r2 = _audit(client, b"not a png at all" * 4)
    assert r2.status_code == 400


def test_empty_file_400():
    client, _ = _fresh_client()
    r = _audit(client, b"")
    assert r.status_code == 400


def test_over_large_upload_413():
    client, _ = _fresh_client()
    # 16MB + 1（签名合法，在审计前拦截，不解析）
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (16 * 1024 * 1024 - 7)
    r = _audit(client, oversized)
    assert r.status_code == 413


def test_no_persistence():
    client, data_dir = _fresh_client()
    n_before = len(list(data_dir.glob("*"))) if data_dir.exists() else 0
    for payload in (b"", b"hidden"):
        r = _audit(client, _plain_png() + payload)
        assert r.status_code == 200
    n_after = len(list(data_dir.glob("*"))) if data_dir.exists() else 0
    assert n_after == n_before
    assert not list(data_dir.glob("*/meta.json"))
    assert not (data_dir / "archive.log.jsonl").exists()


if __name__ == "__main__":
    import inspect
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and inspect.isfunction(v)]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\n{len(fns)} 个 PNG 结构审计测试全部通过 ✔")
