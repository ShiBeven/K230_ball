"""
JSON 配置加载 —— MicroPython (ujson) / CPython (json) 双兼容。
"""

try:
    import ujson as json
except ImportError:
    import json


def load_config(path="config.json"):
    """
    从 JSON 文件加载配置。
    """
    try:
        with open(path, "r") as f:
            return json.load(f)
    except OSError as e:
        print("[CONFIG] Failed to load config file: {}".format(path))
        print("[CONFIG] Error: {}".format(e))
        raise SystemExit(1)
    except ValueError as e:
        print("[CONFIG] JSON parse error in: {}".format(path))
        print("[CONFIG] Error: {}".format(e))
        raise SystemExit(1)


def get_with_default(config, key_path, default):
    """
    从嵌套 dict 中按 "." 分隔的路径取值，不存在时返回默认值。

    Args:
        config: 配置 dict
        key_path: "a.b.c" 形式的键路径
        default: 默认值

    Example:
        get_with_default(config, "kpu.threshold", 0.7)
    """
    keys = key_path.split(".")
    node = config
    for key in keys:
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return default
    return node
