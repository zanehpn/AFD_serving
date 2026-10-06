"""Separate AFD attention sends from receives to break DBO stream cycles."""
from pathlib import Path
import hashlib,json,difflib
root=Path(__file__).resolve().parents[1]
p=root/'.venv-a6000-p2p/lib/python3.12/site-packages/afd_plugin/connectors/gpu/p2p.py'
before=p.read_text()
old='    def send_attn_output(\n'
new='''    def send_attn_output(
        self,
        hidden_states: torch.Tensor,
        context: AFDTransferContext,
        **kwargs: Any,
    ) -> None:
        """Queue outgoing payloads independently of incoming FFN results.

        DBO yields after each send. On one stream, send(stage 1) can block
        recv(stage 0), while the FFN's send(stage 0) blocks recv(stage 1).
        A dedicated outgoing stream breaks that cycle. Wait for the matching
        send event before reusing the input tensor as the receive buffer.
        """
        if not self.vllm_config.model_config.enforce_eager:
            return self._send_attn_output_on_current_stream(hidden_states, context, **kwargs)
        if not hasattr(self, "_attention_send_stream"):
            self._attention_send_stream = torch.cuda.Stream(device=hidden_states.device)
            self._attention_send_events = {}
        outgoing = self._attention_send_stream
        outgoing.wait_stream(torch.cuda.current_stream(hidden_states.device))
        with torch.cuda.stream(outgoing):
            self._send_attn_output_on_current_stream(hidden_states, context, **kwargs)
            self._attention_send_events[int(context.metadata.stage_idx)] = outgoing.record_event()
            hidden_states.record_stream(outgoing)
            for name in ("router_logits", "input_ids"):
                tensor = kwargs.get(name)
                if tensor is not None:
                    tensor.record_stream(outgoing)

    def _send_attn_output_on_current_stream(
'''
assert before.count(old)==1
after=before.replace(old,new)
old='        output = self._recv_hidden_states(\n'
new='''        event = getattr(self, "_attention_send_events", {}).pop(ubatch_idx, None)
        if event is not None:
            torch.cuda.current_stream(ref_tensor.device).wait_event(event)
        output = self._recv_hidden_states(
'''
assert after.count(old)==1;after=after.replace(old,new)
tmp=p.with_suffix('.repair-tmp');tmp.write_text(after);tmp.replace(p)
out=root/'runtime-patches/a6000-p2p';out.mkdir(exist_ok=True)
m={'reason':'Fix eager AFD DBO send/receive stream dependency cycle','files':{str(p):{'before_sha256':hashlib.sha256(before.encode()).hexdigest(),'after_sha256':hashlib.sha256(after.encode()).hexdigest()}}}
(out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
(out/'p2p.py.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/afd_plugin/connectors/gpu/p2p.py',tofile='b/afd_plugin/connectors/gpu/p2p.py')))
print(out/'manifest.json')
