#!/usr/bin/env python3
"""OpenAI-chat fake with a tool mode for the canary's cross-protocol TOOL
round-trip. In tool mode it inspects the (aivo-transformed) tools array, emits
a tool_call that creates canary_done.txt=CANARY_OK, then on the follow-up turn
(tool result present) returns a final text. Proves the full
tool_call->tool_result cycle survives aivo's protocol converters.

FAKE_MODE=text|tool|task (default text). `task` first answers with a Task
tool_call so the CLI spawns a subagent, then behaves like `tool`.
Logs received paths + per-request models + a debug dump."""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

REPLY = "CANARY_OK"
MODE = os.environ.get("FAKE_MODE", "text")
SEEN = os.environ.get("FAKE_SEEN_LOG", "/tmp/fake_seen.log")
DUMP = os.environ.get("FAKE_DUMP_LOG", "/tmp/fake_dump.log")


def logp(s):
    with open(SEEN, "a") as f:
        f.write(s + "\n")


def dump(s):
    with open(DUMP, "a") as f:
        f.write(s + "\n")


def pick_tool_and_args(tools):
    """From OpenAI-format function tools, choose a write-file or shell tool and
    synthesize arguments that create canary_done.txt=CANARY_OK."""
    write_t = shell_t = js_t = None
    for t in tools or []:
        fn = t.get("function", t)
        name = (fn.get("name") or "").lower()
        desc = (fn.get("description") or "").lower()
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        prop_keys = {p.lower() for p in props}
        # codex >=0.143 sol code-mode `exec` runs JavaScript — raw shell in
        # `input` would be a JS syntax error, so prefer JS over shell_t.
        if js_t is None and name == "exec" and "javascript" in desc:
            js_t = (fn.get("name"), props)
        # Shell/exec tool — preferred: schema-stable ({command}/{cmd}) and every
        # agent has one (claude Bash, gemini run_shell_command, codex
        # exec_command), so `printf ... > file` works universally. Match by NAME
        # only — description matching mis-hit gemini's grep_search.
        if shell_t is None and any(k in name for k in ["bash", "shell", "exec", "run_command", "terminal"]):
            shell_t = (fn.get("name"), props)
        # File-CREATE tool — fallback. A `content`/`text` param means create;
        # exclude edit/replace tools (old_string/new_string) which need an
        # existing file.
        is_edit = bool(prop_keys & {"new_string", "old_string", "new_str"})
        if (write_t is None and not is_edit
                and any(k in name for k in ["write", "create"])
                and (not prop_keys or prop_keys & {"content", "contents", "text", "file_text"})):
            write_t = (fn.get("name"), props)

    if js_t:
        nm, props = js_t
        param = next(iter(props), "input")
        return nm, {param: 'await tools.exec_command({cmd: "printf %s CANARY_OK > canary_done.txt"})'}
    if shell_t:
        nm, props = shell_t
        cmd_param = None
        for p in props:
            if any(k in p.lower() for k in ["command", "cmd", "script", "input"]):
                cmd_param = p
                break
        if not cmd_param and props:
            cmd_param = list(props.keys())[0]
        return nm, {cmd_param or "command": "printf %s CANARY_OK > canary_done.txt"}
    if write_t:
        nm, props = write_t
        args = {}
        for p in props:
            pl = p.lower()
            if pl in ("content", "contents", "text", "file_text"):
                args[p] = REPLY
            elif any(k in pl for k in ["path", "file", "filename"]):
                args[p] = "canary_done.txt"
        return nm, args
    return None, None


TASK_EMITTED = False


def pick_task_tool(tools):
    """Synthesize a dispatch whose subagent creates the sentinel file."""
    for t in tools or []:
        fn = t.get("function", t)
        name = (fn.get("name") or "").lower()
        if name not in ("task", "dispatch_agent"):
            continue
        props = ((fn.get("parameters") or {}).get("properties")) or {}
        args = {}
        for p in props:
            pl = p.lower()
            if "prompt" in pl:
                args[p] = "Create a file named canary_done.txt with content CANARY_OK"
            elif "description" in pl:
                args[p] = "canary subagent"
            elif "subagent" in pl or "agent" in pl:
                args[p] = "general-purpose"
        return fn.get("name"), args
    return None, None


def has_tool_result(messages):
    """True once the conversation carries a tool result (the follow-up turn)."""
    for m in messages or []:
        if m.get("role") == "tool":
            return True
        c = m.get("content")
        if isinstance(c, list) and any(isinstance(p, dict) and p.get("type") in ("tool_result", "tool_call_output") for p in c):
            return True
    return False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        logp("GET " + self.path)
        if "models" in self.path:
            self._json(200, {"object": "list", "data": [{"id": "canary-model", "object": "model", "owned_by": "fake"}]})
        else:
            self._json(404, {"error": {"message": "not found"}})

    def _tool_call(self, model, nm, args, stream):
        tc = {"index": 0, "id": "call_canary", "type": "function",
              "function": {"name": nm, "arguments": json.dumps(args)}}
        if stream:
            self._stream(model, [{"role": "assistant", "tool_calls": [tc]}], "tool_calls")
        else:
            self._json(200, {"id": "chatcmpl-canary", "object": "chat.completion", "created": 1,
                             "model": model, "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": None, "tool_calls": [tc]}}],
                             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

    def _stream(self, model, delta_seq, finish):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for delta in delta_seq:
            chunk = {"id": "chatcmpl-canary", "object": "chat.completion.chunk", "created": 1,
                     "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        fin = {"id": "chatcmpl-canary", "object": "chat.completion.chunk", "created": 1,
               "model": model, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
        self.wfile.write(f"data: {json.dumps(fin)}\n\n".encode())
        usage = {"id": "chatcmpl-canary", "object": "chat.completion.chunk", "created": 1, "model": model,
                 "choices": [], "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        self.wfile.write(f"data: {json.dumps(usage)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def do_POST(self):
        logp("POST " + self.path)
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b""
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        if "chat/completions" not in self.path:
            self._json(404, {"error": {"message": "unsupported endpoint"}})
            return
        model = req.get("model", "canary-model")
        messages = req.get("messages", [])
        tools = req.get("tools", [])
        stream = bool(req.get("stream"))
        logp("MODEL " + model)

        if tools and not has_tool_result(messages):
            dump("FULLTOOLS: " + json.dumps([
                {(t.get("function") or t).get("name"):
                 list(((t.get("function") or t).get("parameters") or {}).get("properties", {}).keys())}
                for t in tools]))

        global TASK_EMITTED
        if MODE == "task" and tools and not TASK_EMITTED and not has_tool_result(messages):
            nm, args = pick_task_tool(tools)
            dump(f"task picked={nm} args={args}")
            if nm:
                TASK_EMITTED = True
                self._tool_call(model, nm, args, stream)
                return

        emit_tool = MODE in ("tool", "task") and tools and not has_tool_result(messages)
        if emit_tool:
            nm, args = pick_tool_and_args(tools)
            dump(f"picked={nm} args={args}")
            if nm:
                self._tool_call(model, nm, args, stream)
                return
            dump("tool mode but no write/shell tool found; falling back to text")

        # text reply (v3a, or v3b follow-up after the tool result)
        if stream:
            self._stream(model, [{"role": "assistant"}, {"content": REPLY + " done"}], "stop")
        else:
            self._json(200, {"id": "chatcmpl-canary", "object": "chat.completion", "created": 1, "model": model,
                             "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": REPLY + " done"}}],
                             "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    open(SEEN, "w").close()
    open(DUMP, "w").close()
    srv = HTTPServer(("127.0.0.1", port), Handler)
    print(srv.server_address[1], flush=True)
    srv.serve_forever()
