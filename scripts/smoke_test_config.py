"""Quick smoke test for VasudhaConfig."""
from vasudha.models.config import VasudhaConfig

# Test debug config
c = VasudhaConfig.for_debug()
print("Config OK")
print(repr(c))
print(f"Layers: {c.num_hidden_layers}")
print(f"MoE layers: {c.moe_layers}")
print(f"Attn pattern: {[c.get_attention_type_for_layer(i) for i in range(4)]}")
p = c.estimate_parameters()
print(f"Params: {p['total']/1e6:.1f}M")

# Test Qwen3-4B config
c4b = VasudhaConfig.for_qwen3_4b()
p4b = c4b.estimate_parameters()
print(f"Qwen3-4B estimate: {p4b['total']/1e9:.2f}B params")
print(f"Qwen3-4B MoE layers count: {len(c4b.moe_layers)}")
print(f"Qwen3-4B use_moe: {c4b.use_moe}")

# Test is_moe_layer
assert not c.is_moe_layer(0), "Layer 0 should not be MoE"
assert c.is_moe_layer(1), "Layer 1 should be MoE"
assert not c.is_moe_layer(2), "Layer 2 should not be MoE"

# Test hybrid pattern
assert c.get_attention_type_for_layer(0) == "gla"
assert c.get_attention_type_for_layer(3) == "sdpa"

print("\nAll config smoke tests PASSED!")
