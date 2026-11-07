"""Parameter roles for the offline Flash Next execution recipe."""

import re


ROLES = {
    "attn_hyper_connection.block_inject_weight.weight": "hc_attn_inject.weight",
    "attn_hyper_connection.hc_norm.weight": "hc_attn_norm.weight",
    "attn_hyper_connection.input_mix_weight_down.weight": "hc_attn_down.weight",
    "attn_hyper_connection.input_mix_weight_up.weight": "hc_attn_up.weight",
    "mlp_hyper_connection.block_inject_weight.weight": "hc_ffn_inject.weight",
    "mlp_hyper_connection.hc_norm.weight": "hc_ffn_norm.weight",
    "mlp_hyper_connection.input_mix_weight_down.weight": "hc_ffn_down.weight",
    "mlp_hyper_connection.input_mix_weight_up.weight": "hc_ffn_up.weight",
    "linear_attn.A_log": "ssm_a_log",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "self_attn.indexer.index_qk_proj.weight": "index_qk.weight",
    "self_attn.indexer.q_layernorm.weight": "index_q_norm.weight",
    "self_attn.indexer.k_layernorm.weight": "index_k_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "mlp.experts.gate_up_proj": "ffn_gate_up_exps.weight",
    "mlp.experts.down_proj": "ffn_down_exps.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
    "ple.conv1d.weight": "ple_conv1d.weight",
    "ple.key_proj.weight": "ple_key.weight",
    "ple.norm_conv.weight": "ple_norm_conv.weight",
    "ple.norm_key.weight": "ple_norm_key.weight",
    "ple.norm_query.weight": "ple_norm_query.weight",
    "ple.value_proj.weight": "ple_value.weight",
}


def role(name):
    if name.startswith("mtp."):
        suffix = name[4:]
        if suffix in (
            "fc_embedding.weight",
            "fc_hidden.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
        ):
            return suffix
        if suffix.startswith("layers.0."):
            return role("model.language_model.layers.48." + suffix[len("layers.0.") :])
        if suffix.startswith("hyper_connection_mixer."):
            return role("model.language_model." + suffix)
        raise ValueError(f"Unsupported Flash Next MTP parameter: {name}")
    if ".ple_embedding." in name:
        return None  # CPU PLE metadata.
    if name == "model.language_model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "lm_head.weight":
        return "output.weight"
    prefix = "model.language_model.hyper_connection_mixer."
    if name.startswith(prefix):
        suffix = name[len(prefix) :]
        return {
            "hc_norm.weight": "output_hc_norm.weight",
            "input_mix_weight_down.weight": "output_hc_down.weight",
            "input_mix_weight_up.weight": "output_hc_up.weight",
        }[suffix]
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unexpected Flash Next text parameter: {name}")
    if match[2] not in ROLES:
        raise ValueError(f"Unsupported Flash Next parameter role: {name}")
    return f"blk.{int(match[1])}.{ROLES[match[2]]}"
