import os
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as L
from torch.utils.data import DataLoader, IterableDataset
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from peft import LoraConfig, get_peft_model

from cp_transformer import RoFormerSymbolicTransformer, fill_with_neg_inf
from cp_transformer_fine_tune import (
    RoformerFineTune,
    PreprocessingParameters,
    RoFormerSymbolicTransformerInjected,
)
from modules.yinyang_cross_attn import LowRankMultiheadAttention
from modules.hubert import _compute_mask
from generator_helper import end_generator
from yield_tags import Tags


# ============================================================
# TRAINING CONSTANTS
# ============================================================

TRAIN_LENGTH = 384

# 50% overlap:
#
#   window 1: 0   ........ 383
#   window 2: 192 ........ 575
#   window 3: 384 ........ 767
#
WINDOW_STEP = 192

MAX_STEPS = 60000


# ============================================================
# OVERLAPPING DATASET
# ============================================================

class OverlapFramedDataset(IterableDataset):
    """
    Generate systematic overlapping windows from complete songs.

    The underlying .pt dataset remains unchanged.

    Each eligible song produces:

        start = 0
        start = WINDOW_STEP
        start = 2 * WINDOW_STEP
        ...

    with one additional final window if necessary.

    The final window is forced to end exactly at the end of the song.

    Example:

        target_length = 384
        window_step   = 192
        song_length   = 650

        windows:

            0   -> 383
            192 -> 575
            266 -> 649

    The final stride is therefore allowed to be smaller than 192.
    This guarantees complete coverage of the song.

    Windows are shuffled as a group before each pass through the
    dataset.

    The dataset yields exactly the same object structure expected
    by the existing Yinyang training code:

        (
            batch_data,
            batch_pitch_shift
        )
    """

    def __init__(
        self,
        file_path,
        target_length,
        batch_size,
        split='all',
        split_ratio=10,
        window_step=192,
        random_order=True,
        repeat=True,
    ):
        super().__init__()

        if target_length <= 0:
            raise ValueError("target_length must be > 0")

        if window_step <= 0:
            raise ValueError("window_step must be > 0")

        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        if split not in ('all', 'train', 'val', 'test'):
            raise ValueError(
                f"Unknown split '{split}'. "
                "Expected 'all', 'train', 'val', or 'test'."
            )

        self.file_path = file_path
        self.target_length = target_length
        self.batch_size = batch_size
        self.split = split
        self.split_ratio = split_ratio
        self.window_step = window_step
        self.random_order = random_order
        self.repeat = repeat

        # ----------------------------------------------------
        # Load only metadata initially.
        # The actual data tensor is loaded lazily in __iter__.
        # ----------------------------------------------------

        self.length = torch.load(
            file_path[:-3] + '.length.pt',
            weights_only=True
        )

        # Absolute starting position of every song inside
        # the concatenated data tensor.
        self.start = (
            torch.cumsum(self.length, dim=0) - self.length
        )

        self.song_indices = torch.arange(len(self.start))

        # ----------------------------------------------------
        # Split songs exactly like the original FramedDataset.
        # ----------------------------------------------------

        is_valid = self.length >= target_length

        if split == 'all':
            self.valid_indices = self.song_indices[is_valid]

        elif split == 'train':
            self.valid_indices = self.song_indices[
                torch.logical_and(
                    self.song_indices % split_ratio > 1,
                    is_valid
                )
            ]

        elif split == 'val':
            self.valid_indices = self.song_indices[
                torch.logical_and(
                    self.song_indices % split_ratio == 1,
                    is_valid
                )
            ]

        elif split == 'test':
            self.valid_indices = self.song_indices[
                torch.logical_and(
                    self.song_indices % split_ratio == 0,
                    is_valid
                )
            ]

        self.valid_song_count = len(self.valid_indices)

        print(
            'Metadata for dataset',
            file_path,
            'split',
            split,
            'loaded.'
        )

        print(
            'Number of valid songs:',
            self.valid_song_count
        )

        print(
            'Window length:',
            target_length,
            'Window step:',
            window_step,
            'Overlap:',
            target_length - window_step
        )

        print(
            'First 20 valid song indices:',
            self.valid_indices[:20]
        )

        self.data = None
        self.pitch_shift_range = None

    # --------------------------------------------------------
    # Window generation
    # --------------------------------------------------------

    def _get_window_starts(self, song_length):
        """
        Return all window starts for one song.

        Normal starts use WINDOW_STEP.

        The final start is forced to:

            song_length - target_length

        if the regular sequence does not already reach it.
        """

        max_start = song_length - self.target_length

        starts = list(
            range(
                0,
                max_start + 1,
                self.window_step
            )
        )

        if starts[-1] != max_start:
            starts.append(max_start)

        return starts

    def _build_window_descriptors(self):
        """
        Build all windows for all songs in the selected split.

        Each descriptor is:

            (song_index, absolute_start)

        No actual tensor data is copied here.
        """

        windows = []

        for song_index in self.valid_indices.tolist():

            song_length = int(
                self.length[song_index].item()
            )

            song_start = int(
                self.start[song_index].item()
            )

            relative_starts = self._get_window_starts(
                song_length
            )

            for relative_start in relative_starts:

                absolute_start = (
                    song_start + relative_start
                )

                windows.append(
                    (
                        song_index,
                        absolute_start
                    )
                )

        return windows

    # --------------------------------------------------------
    # IterableDataset
    # --------------------------------------------------------

    def __iter__(self):

        # ----------------------------------------------------
        # Lazy loading.
        #
        # This preserves the behavior of the original dataset:
        # metadata is available to the parent process, while the
        # large tensor is loaded by the DataLoader worker.
        # ----------------------------------------------------

        if self.data is None:

            self.data = torch.load(
                self.file_path,
                weights_only=True
            )

            self.pitch_shift_range = torch.load(
                self.file_path[:-3] +
                '.pitch_shift_range.pt',
                weights_only=True
            ).reshape(-1, 2)

            # Same pitch-shift limits as original FramedDataset.
            self.pitch_shift_range[
                self.pitch_shift_range[:, 0] < -5,
                0
            ] = -5

            self.pitch_shift_range[
                self.pitch_shift_range[:, 1] > 6,
                1
            ] = 6

            # Validation/test must not receive random
            # pitch augmentation.
            if self.split in ('val', 'test'):
                self.pitch_shift_range = torch.zeros_like(
                    self.pitch_shift_range
                )

            print(
                'Data for dataset',
                self.file_path,
                'loaded.'
            )

        # ----------------------------------------------------
        # Repeat indefinitely unless repeat=False.
        # One complete pass contains EVERY window.
        # ----------------------------------------------------

        while True:

            windows = self._build_window_descriptors()

            if self.random_order:

                permutation = torch.randperm(
                    len(windows)
                )

                windows = [
                    windows[i]
                    for i in permutation.tolist()
                ]

            # ------------------------------------------------
            # Batch windows.
            # ------------------------------------------------

            for batch_start in range(
                0,
                len(windows),
                self.batch_size
            ):

                batch_windows = windows[
                    batch_start:
                    batch_start + self.batch_size
                ]

                batch_data = []
                batch_pitch_shift = []

                for song_index, absolute_start in batch_windows:

                    absolute_end = (
                        absolute_start +
                        self.target_length
                    )

                    # Exactly 384 steps.
                    window = self.data[
                        absolute_start:absolute_end
                    ]

                    if window.shape[0] != self.target_length:
                        raise RuntimeError(
                            "Internal error: generated window "
                            f"has length {window.shape[0]}, "
                            f"expected {self.target_length}. "
                            f"song_index={song_index}, "
                            f"absolute_start={absolute_start}"
                        )

                    batch_data.append(window)

                    # ------------------------------------------------
                    # Random pitch shift, exactly as original
                    # FramedDataset.
                    # ------------------------------------------------

                    shift_range = (
                        self.pitch_shift_range[song_index]
                    )

                    min_shift = int(
                        shift_range[0].item()
                    )

                    max_shift = int(
                        shift_range[1].item()
                    )

                    pitch_shift = int(
                        torch.randint(
                            min_shift,
                            max_shift + 1,
                            (1,)
                        ).item()
                    )

                    batch_pitch_shift.append(
                        pitch_shift
                    )

                yield (
                    torch.stack(batch_data, dim=0),
                    torch.tensor(
                        batch_pitch_shift,
                        dtype=torch.long
                    )
                )

            if not self.repeat:
                break


# ============================================================
# YINYANG MODEL
#
# This is intentionally copied from the existing
# cp_transformer_yinyang.py.
#
# The model itself is NOT modified.
# ============================================================

class RoformerYinyang(RoformerFineTune):

    def __init__(
        self,
        model_fp,
        train_task,
        mask_prob=None,
        mask_length=None,
        max_position_embeddings=768,
        n_skip=2,
        compress_ratio_l=1,
        compress_ratio_r=1,
        lr=None,
        use_lora=True
    ):

        super().__init__(
            compress_ratio_l=compress_ratio_l,
            compress_ratio_r=compress_ratio_r,
            lr=lr
        )

        self.save_hyperparameters()

        if not os.path.exists(model_fp):
            model_fp = os.path.join(
                'ckpt',
                os.path.basename(model_fp)
            )

        base_model = (
            RoFormerSymbolicTransformerInjected
            .load_from_checkpoint(
                model_fp,
                max_position_embeddings=max_position_embeddings,
                strict=False
            )
        )

        if use_lora:

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=[
                    "query",
                    "value"
                ],
                lora_dropout=0.1,
            )

            self.wrapped_model = get_peft_model(
                base_model,
                lora_config
            )

        else:

            self.wrapped_model = base_model
            self.wrapped_model.eval()
            self.wrapped_model.freeze()

        self.n_layers = base_model.num_layers
        self.n_skip = n_skip

        (
            self.yinyang_attn,
            self.masked_embedding
        ) = self.initialize_trainable(
            base_model.hidden_size,
            max_position_embeddings
        )

        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.yinyang_mask_ratio = 0.0

        self.preprocess_args = (
            PreprocessingParameters(train_task)
        )

    def initialize_trainable(
        self,
        hidden_size,
        max_position_embeddings
    ):

        yinyang_attn = nn.ModuleList([
            LowRankMultiheadAttention(
                in_dim=hidden_size,
                embed_dim=256,
                num_heads=4,
                dropout=0.1,
                max_len=max_position_embeddings
            )
            for _ in range(
                self.n_layers // self.n_skip
            )
        ])

        masked_embedding = nn.Parameter(
            torch.randn(
                1,
                1,
                hidden_size
            ) * 0.1,
            requires_grad=True
        )

        return yinyang_attn, masked_embedding

    def get_yinyang_attn(self, layer):
        return self.yinyang_attn[layer]

    def forward(
        self,
        x1,
        x2,
        indices1,
        indices2
    ):

        gen1 = self.wrapped_model(x1)
        gen2 = self.wrapped_model(x2)

        data1 = next(gen1)
        assert data1[0] == Tags.SIMUNOTE_EMBEDDING

        data2 = next(gen2)
        assert data2[0] == Tags.SIMUNOTE_EMBEDDING

        if self.mask_prob > 0.0:

            if self.preprocess_args.left_mask:

                mask1 = _compute_mask(
                    (
                        x1.size(0),
                        x1.size(1)
                    ),
                    self.mask_prob,
                    self.mask_length,
                    x1.device,
                    2
                )

                data1[1][mask1] = (
                    self.masked_embedding
                    .to(data1[1].dtype)
                )

            elif self.training:

                mask2 = _compute_mask(
                    (
                        x2.size(0),
                        x2.size(1)
                    ),
                    self.mask_prob,
                    self.mask_length,
                    x2.device,
                    2
                )

                data2[1][mask2] = (
                    self.masked_embedding
                    .to(data2[1].dtype)
                )

        data1 = next(gen1)
        assert data1[0] == Tags.PE_POSITIONS

        data2 = next(gen2)
        assert data2[0] == Tags.PE_POSITIONS

        for layer in range(self.n_layers):

            data1 = next(gen1)
            assert data1[0] == Tags.HIDDEN_STATES

            data2 = next(gen2)
            assert data2[0] == Tags.HIDDEN_STATES

            if layer % self.n_skip == 0:

                h = data2[1]

                yinyang_weights = (
                    self.get_yinyang_attn(
                        layer // self.n_skip
                    )(
                        h,
                        data1[1],
                        data1[1],
                        None,
                        indices_key=indices1,
                        indices_query=indices2
                    )
                )

            data1 = next(gen1)
            assert data1[0] == Tags.PRENORM_OUTPUT

            data2 = next(gen2)
            assert data2[0] == Tags.PRENORM_OUTPUT

            if layer % self.n_skip == 0:

                if (
                    self.training
                    and self.yinyang_mask_ratio > 0.0
                ):

                    yinyang_mask = (
                        torch.rand(
                            data2[1].shape[:2],
                            device=data2[1].device
                        )
                        < self.yinyang_mask_ratio
                    )

                    yinyang_weights.masked_fill_(
                        yinyang_mask.unsqueeze(-1),
                        0
                    )

                data2[1] = (
                    data2[1] +
                    yinyang_weights
                )

        return end_generator(gen2)

    def global_sampling(
        self,
        x1,
        x2,
        temperature=1.0,
        multiplier=1.0,
        sampling_func=None
    ):

        if not isinstance(multiplier, float):

            multiplier = torch.tensor(
                multiplier,
                dtype=torch.float32,
                device=x1.device
            )

            multiplier = multiplier[:, None, None]

        print('Yinyang Sampling')

        batch_size, max_seq_len, subseq_len = (
            x1.shape
        )

        indices1 = (
            torch.arange(
                max_seq_len,
                dtype=torch.long,
                device=x1.device
            )
            * self.compress_ratio_l
        )

        max_seq_len = (
            max_seq_len *
            self.compress_ratio_l //
            self.compress_ratio_r
        )

        indices2 = (
            torch.arange(
                max_seq_len,
                dtype=torch.long,
                device=x2.device
            )
            * self.compress_ratio_r
        )

        seq_len = x2.shape[1]

        gen1 = self.wrapped_model(x1)

        gen1_hidden_states = []

        data1 = next(gen1)
        assert data1[0] == Tags.SIMUNOTE_EMBEDDING

        if self.preprocess_args.left_mask:

            mask1 = _compute_mask(
                (
                    x1.size(0),
                    x1.size(1)
                ),
                self.mask_prob,
                self.mask_length,
                x1.device,
                2
            )

            data1[1][mask1] = (
                self.masked_embedding
                .to(data1[1].dtype)
            )

        data1 = next(gen1)
        assert data1[0] == Tags.PE_POSITIONS

        for layer in range(self.n_layers):

            data1 = next(gen1)
            assert data1[0] == Tags.HIDDEN_STATES

            gen1_hidden_states.append(
                data1[1]
            )

            data1 = next(gen1)
            assert data1[0] == Tags.PRENORM_OUTPUT

        end_generator(gen1)

        gen2 = self.wrapped_model.global_sampling(
            x2,
            max_seq_len=max_seq_len,
            temperature=temperature,
            sampling_func=sampling_func
        )

        for i in range(
            seq_len,
            max_seq_len
        ):

            data2 = next(gen2)
            assert data2[0] == Tags.GENERATION_STEP

            data2 = next(gen2)
            assert data2[0] == Tags.PE_POSITIONS

            for layer in range(self.n_layers):

                data2 = next(gen2)
                assert data2[0] == Tags.HIDDEN_STATES

                if layer % self.n_skip == 0:

                    h = data2[1]

                    indices2_slice = indices2[
                        i + 1 - h.shape[1]:
                        i + 1
                    ]

                    yinyang_weights = (
                        self.get_yinyang_attn(
                            layer // self.n_skip
                        )(
                            h,
                            gen1_hidden_states[layer],
                            gen1_hidden_states[layer],
                            None,
                            indices_key=indices1,
                            indices_query=indices2_slice
                        )
                    )

                data2 = next(gen2)
                assert data2[0] == Tags.PRENORM_OUTPUT

                if layer % self.n_skip == 0:

                    data2[1] = (
                        data2[1] +
                        yinyang_weights * multiplier
                    )

        return end_generator(gen2)

    def loss(self, x, pitch_shift):

        (
            x1,
            x2,
            indices1,
            indices2
        ) = self.preprocess(
            x,
            pitch_shift,
            preprocess_args=self.preprocess_args
        )

        y = self(
            x1,
            x2,
            indices1,
            indices2
        )

        return F.cross_entropy(
            y.view(
                -1,
                self.wrapped_model.tokenizer.n_tokens
            ),
            x2.reshape(-1),
            ignore_index=self.wrapped_model.tokenizer.pad_token
        )


# ============================================================
# TRAINING
# ============================================================

def train(
    net,
    model_name,
    early_stopping_patience,
    max_steps,
    train_set_loader,
    val_set_loader
):

    n_gpus = max(
        torch.cuda.device_count(),
        1
    )

    checkpoint_callback = (
        L.callbacks.ModelCheckpoint(
            monitor='val_loss',
            save_top_k=10,
            save_last=True,
            dirpath=f'ckpt/{model_name}',
            enable_version_counter=False,
            filename=(
                model_name +
                '.{epoch:02d}.{val_loss:.5f}'
            )
        )
    )

    early_stopping = (
        L.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=early_stopping_patience
        )
    )

    trainer = L.Trainer(
        devices=n_gpus,
        precision=(
            "bf16-mixed"
            if torch.cuda.is_available()
            else 32
        ),
        max_steps=max_steps,
        accelerator=(
            'gpu'
            if torch.cuda.is_available()
            else 'cpu'
        ),
        callbacks=[
            checkpoint_callback,
            early_stopping
        ],
        val_check_interval=500,
        limit_val_batches=25,
        check_val_every_n_epoch=None,
        logger=TensorBoardLogger(
            "tb_logs",
            name=model_name
        ),
        strategy=(
            'auto'
            if n_gpus == 1
            else 'ddp'
        )
    )

    net.strict_loading = False

    trainer.fit(
        net,
        train_set_loader,
        val_set_loader
    )

    shutil.copy(
        checkpoint_callback.best_model_path,
        f'ckpt/{model_name}.epoch=best.ckpt'
    )

    os.chmod(
        f'ckpt/{model_name}.epoch=best.ckpt',
        0o666
    )

    shutil.copy(
        f'ckpt/{model_name}/last.ckpt',
        f'ckpt/{model_name}.epoch=last.ckpt'
    )

    os.chmod(
        f'ckpt/{model_name}.epoch=last.ckpt',
        0o666
    )


# ============================================================
# MAIN
# ============================================================

def main():

    import argparse

    args = argparse.ArgumentParser()

    args.add_argument(
        '--batch_size',
        type=int,
        required=True
    )

    args.add_argument(
        '--fp_path',
        type=str,
        default=(
            'ckpt/'
            'cp_transformer_v0.42_size1_batch_48_schedule.'
            'epoch=00.fin.ckpt'
        )
    )

    args.add_argument(
        '--dataset_name',
        type=str,
        required=True
    )

    args.add_argument(
        '--train_task',
        type=str,
        default=None
    )

    args.add_argument(
        '--weights_path',
        type=str,
        default=None
    )

    args.add_argument(
        '--mask_prob',
        type=float,
        default=0.25
    )

    args.add_argument(
        '--mask_length',
        type=int,
        default=10
    )

    args.add_argument(
        '--compress_ratio_l',
        type=int,
        default=1
    )

    args.add_argument(
        '--compress_ratio_r',
        type=int,
        default=1
    )

    # --------------------------------------------------------
    # 384 IS INTENTIONAL AND FIXED.
    # --------------------------------------------------------

    args.add_argument(
        '--train_length',
        type=int,
        default=TRAIN_LENGTH
    )

    # --------------------------------------------------------
    # New parameter.
    #
    # Default = 192.
    #
    # It can be changed for experiments without modifying
    # the source.
    # --------------------------------------------------------

    args.add_argument(
        '--window_step',
        type=int,
        default=WINDOW_STEP
    )

    args.add_argument(
        '--lr',
        type=float,
        default=1e-4
    )

    args.add_argument(
        '--use_lora',
        action='store_true',
        default=True
    )

    args.add_argument(
        '--no_lora',
        action='store_false',
        dest='use_lora'
    )

    args.add_argument(
        '--n_skip',
        type=int,
        default=2
    )

    args.add_argument(
        '--max_steps',
        type=int,
        default=MAX_STEPS
    )

    args.add_argument(
        '--early_stopping_patience',
        type=int,
        default=10**10
    )

    args = args.parse_args()

    # --------------------------------------------------------
    # Safety check.
    #
    # We deliberately don't permit the new script to silently
    # become a different-context experiment.
    # --------------------------------------------------------

    if args.train_length != 384:

        raise ValueError(
            "This training script is intentionally fixed at "
            "train_length=384. "
            "The 384-step context must not be changed."
        )

    if args.window_step <= 0:

        raise ValueError(
            "window_step must be > 0"
        )

    sample_step = max(
        args.compress_ratio_l,
        args.compress_ratio_r
    )

    train_length = 384

    n_gpus = max(
        torch.cuda.device_count(),
        1
    )

    lora_tag = (
        '_lora'
        if args.use_lora
        else ''
    )

    skip_tag = (
        f'-skip{args.n_skip}'
        if args.n_skip != 2
        else ''
    )

    train_task = (
        args.dataset_name
        if args.train_task is None
        else args.train_task
    )

    model_name = (
        f'cp_transformer_yinyang_v5.1'
        f'{lora_tag}'
        f'_batch_{args.batch_size * n_gpus}'
        f'_{train_task}'
        f'_mask{args.mask_prob}-{args.mask_length}'
        f'-step{sample_step}'
        f'-window{train_length}'
        f'-overlap{train_length - args.window_step}'
        f'{skip_tag}'
    )

    print()
    print('=' * 72)
    print('OVERLAPPING-WINDOW YINYANG TRAINING')
    print('=' * 72)
    print('Dataset:       ', args.dataset_name)
    print('Task:          ', train_task)
    print('Train length:  ', train_length)
    print('Window step:   ', args.window_step)
    print('Overlap:       ', train_length - args.window_step)
    print('Batch size:    ', args.batch_size)
    print('GPUs:          ', n_gpus)
    print('Max steps:     ', args.max_steps)
    print('LoRA:          ', args.use_lora)
    print('Model name:    ', model_name)
    print('=' * 72)
    print()

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    if args.weights_path is not None:

        net = RoformerYinyang.load_from_checkpoint(
            args.weights_path,
            strict=False,
            lr=args.lr
        )

        print(
            'Loaded from',
            args.weights_path
        )

    else:

        net = RoformerYinyang(
            args.fp_path,
            train_task=train_task,
            mask_prob=args.mask_prob,
            mask_length=args.mask_length,
            compress_ratio_l=args.compress_ratio_l,
            compress_ratio_r=args.compress_ratio_r,
            lr=args.lr,
            use_lora=args.use_lora,
            n_skip=args.n_skip
        )

    # --------------------------------------------------------
    # NEW DATASET ONLY.
    #
    # The original cp_transformer_yinyang.py remains untouched.
    # --------------------------------------------------------

    train_dataset = OverlapFramedDataset(
        f'data/{args.dataset_name}.pt',
        train_length,
        args.batch_size,
        split='train',
        window_step=args.window_step,
        random_order=True,
        repeat=True
    )

    val_dataset = OverlapFramedDataset(
        f'data/{args.dataset_name}.pt',
        train_length,
        args.batch_size,
        split='val',
        window_step=args.window_step,
        random_order=True,
        repeat=True
    )

    train_set_loader = DataLoader(
        train_dataset,
        batch_size=None,
        num_workers=2,
        persistent_workers=True
    )

    val_set_loader = DataLoader(
        val_dataset,
        batch_size=None,
        num_workers=2,
        persistent_workers=True
    )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    train(
        net,
        model_name,
        args.early_stopping_patience,
        args.max_steps,
        train_set_loader,
        val_set_loader
    )


if __name__ == '__main__':
    main()
