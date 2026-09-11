import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
from cp_transformer import RoFormerSymbolicTransformer, FramedDataset, fill_with_neg_inf
from cp_transformer_fine_tune import RoformerFineTune, PreprocessingParameters, RoFormerSymbolicTransformerInjected, train
import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from peft import LoraConfig, get_peft_model
from modules.yinyang_cross_attn import LowRankMultiheadAttention
from modules.hubert import _compute_mask
import sys
from generator_helper import end_generator
from yield_tags import Tags
import math

class RoformerYinyang(RoformerFineTune):

    def __init__(self, model_fp, train_task, mask_prob=None, mask_length=None, max_position_embeddings=768, n_skip=2,
                 compress_ratio_l=1, compress_ratio_r=1, lr=None, use_lora=True):
        super().__init__(compress_ratio_l=compress_ratio_l, compress_ratio_r=compress_ratio_r, lr=lr)
        self.save_hyperparameters()
        if not os.path.exists(model_fp):
            # Use relative path
            model_fp = os.path.join('ckpt', os.path.basename(model_fp))
        base_model = RoFormerSymbolicTransformerInjected.load_from_checkpoint(model_fp, max_position_embeddings=max_position_embeddings, strict=False)
        if use_lora:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["query", "value"],
                lora_dropout=0.1,
            )
            self.wrapped_model = get_peft_model(base_model, lora_config)
        else:
            self.wrapped_model = base_model
            self.wrapped_model.eval()
            self.wrapped_model.freeze()
        self.n_layers = base_model.num_layers
        self.n_skip = n_skip
        self.yinyang_attn, self.masked_embedding = self.initialize_trainable(base_model.hidden_size, max_position_embeddings)
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.yinyang_mask_ratio = 0.0
        self.preprocess_args = PreprocessingParameters(train_task)

    def initialize_trainable(self, hidden_size, max_position_embeddings):
        yinyang_attn = nn.ModuleList([
            LowRankMultiheadAttention(
                in_dim=hidden_size,
                embed_dim=256,
                num_heads=4,
                dropout=0.1,
                max_len=max_position_embeddings
            )
            for _ in range(self.n_layers // self.n_skip)
        ])
        masked_embedding = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.1, requires_grad=True)
        return yinyang_attn, masked_embedding

    def get_yinyang_attn(self, layer):
        return self.yinyang_attn[layer]

    def forward(self, x1, x2, indices1, indices2):
        gen1 = self.wrapped_model(x1)
        gen2 = self.wrapped_model(x2)
        data1 = next(gen1); assert data1[0] == Tags.SIMUNOTE_EMBEDDING
        data2 = next(gen2); assert data2[0] == Tags.SIMUNOTE_EMBEDDING
        if self.mask_prob > 0.0:
            if self.preprocess_args.left_mask:
                mask1 = _compute_mask((x1.size(0), x1.size(1)), self.mask_prob, self.mask_length, x1.device, 2)
                data1[1][mask1] = self.masked_embedding.to(data1[1].dtype)
            elif self.training:
                mask2 = _compute_mask((x2.size(0), x2.size(1)), self.mask_prob, self.mask_length, x2.device, 2)
                data2[1][mask2] = self.masked_embedding.to(data2[1].dtype)
        data1 = next(gen1); assert data1[0] == Tags.PE_POSITIONS
        data2 = next(gen2); assert data2[0] == Tags.PE_POSITIONS
        for layer in range(self.n_layers):
            data1 = next(gen1); assert data1[0] == Tags.HIDDEN_STATES
            data2 = next(gen2); assert data2[0] == Tags.HIDDEN_STATES
            if layer % self.n_skip == 0:
                h = data2[1]
                yinyang_weights = self.get_yinyang_attn(layer // self.n_skip)(h, data1[1], data1[1], None, indices_key=indices1, indices_query=indices2)
            data1 = next(gen1); assert data1[0] == Tags.PRENORM_OUTPUT
            data2 = next(gen2); assert data2[0] == Tags.PRENORM_OUTPUT
            if layer % self.n_skip == 0:
                if self.training and self.yinyang_mask_ratio > 0.0:
                    yinyang_mask = torch.rand(data2[1].shape[:2], device=data2[1].device) < self.yinyang_mask_ratio
                    yinyang_weights.masked_fill_(yinyang_mask.unsqueeze(-1), 0)
                data2[1] = data2[1] + yinyang_weights
        return end_generator(gen2)

    def _print_yinyang_diagnostic(
        self,
        attn_module,
        adapter_index,
        model_global_step,
        harmonic_step,
        melody_global_offset,
        base_state,
        yinyang_state,
        multiplier,
        sample_index=0,
    ):
        """
        Passive YinYang microscope.

        Reports:
          - where each head attends in the melody;
          - local / past / future attention mass;
          - attention entropy;
          - magnitude of the gated conditioning contribution.

        No model state or generation value is modified.
        """

        if not hasattr(
            attn_module,
            "_diag_last_attn_weights",
        ):
            return

        weights = (
            attn_module
            ._diag_last_attn_weights[
                sample_index
            ]
            .detach()
            .float()
            .cpu()
        )

        # [heads, key_len]
        num_heads, key_len = weights.shape

        global_key_positions = torch.arange(
            key_len,
            dtype=torch.long,
        ) + int(melody_global_offset)

        base = (
            base_state[
                sample_index,
                -1,
                :
            ]
            .detach()
            .float()
        )

        # yy = (
        #     yinyang_state[
        #         sample_index,
        #         -1,
        #         :
        #     ]
        #     .detach()
        #     .float()
        # )

        # if isinstance(multiplier, torch.Tensor):
        #     if multiplier.numel() == 1:
        #         mult_value = float(
        #             multiplier.detach().float().item()
        #         )
        #     else:
        #         mult_value = float(
        #             multiplier[
        #                 sample_index
        #             ]
        #             .detach()
        #             .float()
        #             .reshape(-1)[0]
        #             .item()
        #         )
        # else:
        #     mult_value = float(multiplier)

        # contribution = yy * mult_value
        yy_post_gate = (
            yinyang_state[
                sample_index,
                -1,
                :
            ]
            .detach()
            .float()
        )

        yy_pre_gate = (
            attn_module
            ._diag_last_output_pre_gate[
                sample_index
            ]
            .detach()
            .float()
        )

        yy_cached_post_gate = (
            attn_module
            ._diag_last_output_post_gate[
                sample_index
            ]
            .detach()
            .float()
        )

        if isinstance(multiplier, torch.Tensor):
            if multiplier.numel() == 1:
                mult_value = float(
                    multiplier.detach().float().item()
                )
            else:
                mult_value = float(
                    multiplier[
                        sample_index
                    ]
                    .detach()
                    .float()
                    .reshape(-1)[0]
                    .item()
                )
        else:
            mult_value = float(multiplier)

        contribution = (
            yy_post_gate * mult_value
        )
        base_norm = float(
            torch.linalg.vector_norm(base).item()
        )

        # yy_norm = float(
        #     torch.linalg.vector_norm(yy).item()
        # )

        # contribution_norm = float(
        #     torch.linalg.vector_norm(
        #         contribution
        #     ).item()
        # )
        pre_gate_norm = float(
            torch.linalg.vector_norm(
                yy_pre_gate
            ).item()
        )

        post_gate_norm = float(
            torch.linalg.vector_norm(
                yy_post_gate
            ).item()
        )

        cached_post_gate_norm = float(
            torch.linalg.vector_norm(
                yy_cached_post_gate
            ).item()
        )

        contribution_norm = float(
            torch.linalg.vector_norm(
                contribution
            ).item()
        )

        gate_effect_ratio = (
            post_gate_norm / pre_gate_norm
            if pre_gate_norm > 1e-12
            else float("nan")
        )

        cache_delta = float(
            torch.max(
                torch.abs(
                    yy_post_gate
                    - yy_cached_post_gate
                )
            ).item()
        )

        ratio = (
            contribution_norm / base_norm
            if base_norm > 1e-12
            else float("nan")
        )

        gate = float(
            attn_module
            .gates
            .detach()
            .float()
            .cpu()
            .item()
        )

        print()
        print(
            "=== YINYANG MICROSCOPE ==="
        )
        print(
            f"model step:       "
            f"{model_global_step}"
        )
        print(
            f"harmonic step:    "
            f"{harmonic_step}"
        )
        print(
            f"adapter:          "
            f"{adapter_index:02d}"
        )
        print(
            f"melody key range: "
            f"{melody_global_offset}:"
            f"{melody_global_offset + key_len}"
        )
        print(
            f"gate:             "
            f"{gate:+.8f}"
        )
        print(
            f"multiplier:       "
            f"{mult_value:.6f}"
        )
        print(
            f"base norm:        "
            f"{base_norm:.6f}"
        )
        # print(
        #     f"YY norm:          "
        #     f"{yy_norm:.6f}"
        # )
        # print(
        #     f"actual contrib:   "
        #     f"{contribution_norm:.6f}"
        # )
        # print(
        #     f"contrib/base:     "
        #     f"{ratio:.6f}"
        # )
        print(
            f"pre-gate norm:    "
            f"{pre_gate_norm:.6f}"
        )
        print(
            f"post-gate norm:   "
            f"{post_gate_norm:.6f}"
        )
        print(
            f"post/pre:         "
            f"{gate_effect_ratio:.6f}"
        )
        print(
            f"actual contrib:   "
            f"{contribution_norm:.6f}"
        )
        print(
            f"contrib/base:     "
            f"{ratio:.6f}"
        )
        print(
            f"cache delta:      "
            f"{cache_delta:.9e}"
        )
        for head in range(num_heads):
            w = weights[head]

            total = float(w.sum().item())

            if total <= 0.0:
                continue

            # Renormalize defensively.  In eval mode dropout is inactive,
            # but this makes the diagnostic numerically robust.
            w = w / w.sum()

            positions = (
                global_key_positions.float()
            )

            mean_position = float(
                (w * positions).sum().item()
            )

            eps = 1e-12

            entropy = float(
                -(
                    w
                    * torch.log(
                        w.clamp_min(eps)
                    )
                )
                .sum()
                .item()
            )

            if key_len > 1:
                normalized_entropy = (
                    entropy
                    / math.log(key_len)
                )
            else:
                normalized_entropy = 0.0

            past_mask = (
                global_key_positions
                < model_global_step
            )

            current_mask = (
                global_key_positions
                == model_global_step
            )

            future_mask = (
                global_key_positions
                > model_global_step
            )

            past_mass = float(
                w[past_mask].sum().item()
            )

            current_mass = float(
                w[current_mask].sum().item()
            )

            future_mass = float(
                w[future_mask].sum().item()
            )

            def local_mass(radius):
                mask = (
                    torch.abs(
                        global_key_positions
                        - model_global_step
                    )
                    <= radius
                )

                return float(
                    w[mask].sum().item()
                )

            k = min(8, key_len)

            top_values, top_indices = (
                torch.topk(
                    w,
                    k=k,
                )
            )

            top_pairs = []

            for value, index in zip(
                top_values.tolist(),
                top_indices.tolist(),
            ):
                global_pos = (
                    int(melody_global_offset)
                    + int(index)
                )

                top_pairs.append(
                    f"{global_pos}:{value:.4f}"
                )

            print(
                f"  head {head}: "
                f"mean={mean_position:7.2f}  "
                f"entropy={normalized_entropy:.3f}  "
                f"past={past_mass:.3f}  "
                f"here={current_mass:.3f}  "
                f"future={future_mass:.3f}"
            )

            print(
                "          "
                f"±4={local_mass(4):.3f}  "
                f"±8={local_mass(8):.3f}  "
                f"±16={local_mass(16):.3f}  "
                f"±32={local_mass(32):.3f}"
            )

            print(
                "          top: "
                + "  ".join(top_pairs)
            )

    def global_sampling(self, x1, x2, temperature=1.0, multiplier=1.0, sampling_func=None, diagnostic_steps=None, diagnostic_global_offset=0, diagnostic_prompt_length=0,):
        if not isinstance(multiplier, float):
            multiplier = torch.tensor(multiplier, dtype=torch.float32, device=x1.device)
            multiplier = multiplier[:, None, None]
        print('Yinyang Sampling')
        if diagnostic_steps is None:
            diagnostic_steps = set()
        else:
            diagnostic_steps = {
                int(x)
                for x in diagnostic_steps
            }
        batch_size, max_seq_len, subseq_len = x1.shape
        indices1 = torch.arange(max_seq_len, dtype=torch.long, device=x1.device) * self.compress_ratio_l
        max_seq_len = max_seq_len * self.compress_ratio_l // self.compress_ratio_r
        indices2 = torch.arange(max_seq_len, dtype=torch.long, device=x2.device) * self.compress_ratio_r
        seq_len = x2.shape[1]

        gen1 = self.wrapped_model(x1)
        gen1_hidden_states = []
        data1 = next(gen1)
        assert data1[0] == Tags.SIMUNOTE_EMBEDDING
        if self.preprocess_args.left_mask:
            mask1 = _compute_mask((x1.size(0), x1.size(1)), self.mask_prob, self.mask_length, x1.device, 2)
            data1[1][mask1] = self.masked_embedding.to(data1[1].dtype)
        data1 = next(gen1); assert data1[0] == Tags.PE_POSITIONS
        for layer in range(self.n_layers):
            data1 = next(gen1); assert data1[0] == Tags.HIDDEN_STATES
            gen1_hidden_states.append(data1[1])
            data1 = next(gen1); assert data1[0] == Tags.PRENORM_OUTPUT
        end_generator(gen1)
        gen2 = self.wrapped_model.global_sampling(x2, max_seq_len=max_seq_len, temperature=temperature, sampling_func=sampling_func)

        for i in range(seq_len, max_seq_len):
            data2 = next(gen2); assert data2[0] == Tags.GENERATION_STEP
            data2 = next(gen2); assert data2[0] == Tags.PE_POSITIONS
            for layer in range(self.n_layers):
                data2 = next(gen2); assert data2[0] == Tags.HIDDEN_STATES
                # if layer % self.n_skip == 0:
                #     h = data2[1]
                #     indices2_slice = indices2[i + 1 - h.shape[1]:i + 1]
                #     yinyang_weights = self.get_yinyang_attn(layer // self.n_skip)(h, gen1_hidden_states[layer], gen1_hidden_states[layer], None, indices_key=indices1, indices_query=indices2_slice)
                # data2 = next(gen2); assert data2[0] == Tags.PRENORM_OUTPUT
                # if layer % self.n_skip == 0:
                #     data2[1] = data2[1] + yinyang_weights * multiplier
                if layer % self.n_skip == 0:
                    h = data2[1]

                    indices2_slice = indices2[
                        i + 1 - h.shape[1]:
                        i + 1
                    ]

                    adapter_index = (
                        layer // self.n_skip
                    )

                    attn_module = (
                        self.get_yinyang_attn(
                            adapter_index
                        )
                    )

                    yinyang_weights = (
                        attn_module(
                            h,
                            gen1_hidden_states[layer],
                            gen1_hidden_states[layer],
                            None,
                            indices_key=indices1,
                            indices_query=indices2_slice,
                        )
                    )

                data2 = next(gen2)
                assert (
                    data2[0]
                    == Tags.PRENORM_OUTPUT
                )

                if layer % self.n_skip == 0:
                    model_global_step = (
                        int(
                            diagnostic_global_offset
                        )
                        + int(i)
                    )

                    if (
                        model_global_step
                        in diagnostic_steps
                    ):
                        harmonic_step = (
                            model_global_step
                            - int(
                                diagnostic_prompt_length
                            )
                        )

                        self._print_yinyang_diagnostic(
                            attn_module=attn_module,
                            adapter_index=adapter_index,
                            model_global_step=(
                                model_global_step
                            ),
                            harmonic_step=(
                                harmonic_step
                            ),
                            melody_global_offset=(
                                diagnostic_global_offset
                            ),
                            base_state=data2[1],
                            yinyang_state=(
                                yinyang_weights
                            ),
                            multiplier=multiplier,
                            sample_index=0,
                        )

                    data2[1] = (
                        data2[1]
                        + yinyang_weights
                        * multiplier
                    )
        return end_generator(gen2)

    def loss(self, x, pitch_shift):
        x1, x2, indices1, indices2 = self.preprocess(x, pitch_shift, preprocess_args=self.preprocess_args)
        y = self(x1, x2, indices1, indices2)
        return F.cross_entropy(y.view(-1, self.wrapped_model.tokenizer.n_tokens), x2.reshape(-1), ignore_index=self.wrapped_model.tokenizer.pad_token)

def main():
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--batch_size', type=int)
    args.add_argument('--fp_path', type=str, default='ckpt/cp_transformer_v0.42_size1_batch_48_schedule.epoch=00.fin.ckpt')
    args.add_argument('--dataset_name', type=str)
    args.add_argument('--train_task', type=str, default=None)
    args.add_argument('--weights_path', type=str, default=None)
    args.add_argument('--mask_prob', type=float, default=0.25)
    args.add_argument('--mask_length', type=int, default=10)
    args.add_argument('--compress_ratio_l', type=int, default=1)
    args.add_argument('--compress_ratio_r', type=int, default=1)
    args.add_argument('--train_length', type=int, default=384)
    args.add_argument('--lr', type=float, default=1e-4)
    args.add_argument('--use_lora', action='store_true', default=True)
    args.add_argument('--no_lora', action='store_false', dest='use_lora')
    args.add_argument('--n_skip', type=int, default=2)
    args.add_argument('--max_steps', type=int, default=60000)
    args.add_argument('--early_stopping_patience', type=int, default=1e10)
    args = args.parse_args()
    sample_step = max(args.compress_ratio_l, args.compress_ratio_r)
    train_length = args.train_length
    n_gpus = max(torch.cuda.device_count(), 1)
    lora_tag = '_lora' if args.use_lora else ''
    skip_tag = f'-skip{args.n_skip}' if args.n_skip != 2 else ''
    train_task = args.dataset_name if args.train_task is None else args.train_task
    model_name = f'cp_transformer_yinyang_v5.1{lora_tag}_batch_{args.batch_size * n_gpus}_{train_task}_mask{args.mask_prob}-{args.mask_length}-step{sample_step}{skip_tag}'
    if args.weights_path is not None:
        net = RoformerYinyang.load_from_checkpoint(args.weights_path, strict=False, lr=args.lr)
        print('Loaded from', args.weights_path)
    else:
        net = RoformerYinyang(args.fp_path, train_task=train_task, mask_prob=args.mask_prob,
                              mask_length=args.mask_length,
                              compress_ratio_l=args.compress_ratio_l, compress_ratio_r=args.compress_ratio_r,
                              lr=args.lr, use_lora=args.use_lora, n_skip=args.n_skip)
    train_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='train', sample_step=sample_step), batch_size=None, num_workers=1, persistent_workers=True)
    val_set_loader = DataLoader(FramedDataset(f'data/{args.dataset_name}.pt', train_length, args.batch_size, split='val', sample_step=sample_step), batch_size=None, num_workers=1, persistent_workers=True)
    train(net, model_name, args.early_stopping_patience, args.max_steps, train_set_loader, val_set_loader)


if __name__ == '__main__':
    main()