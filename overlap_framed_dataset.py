import torch
from torch.utils.data import IterableDataset


class OverlapFramedDataset(IterableDataset):
    """
    Generate systematic overlapping fixed-length windows from a concatenated
    MIDI tensor dataset.

    Each song remains intact in the underlying .pt file. Windows are generated
    on the fly.

    Example with target_length=384 and window_step=192:

        Song length = 650

        windows:
            0   -> 383
            192 -> 575
            266 -> 649   # final window, forced to song end

    Thus:
        - every complete 192-step position is covered;
        - the final part of the song is never discarded;
        - no window crosses a song boundary;
        - windows are shuffled before batching when random_order=True.

    The dataset yields:
        (batch_tensor, batch_pitch_shift)

    exactly like the existing FramedDataset.
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

        self.file_path = file_path
        self.target_length = target_length
        self.batch_size = batch_size
        self.split = split
        self.split_ratio = split_ratio
        self.window_step = window_step
        self.random_order = random_order
        self.repeat = repeat

        # Song lengths in the concatenated tensor.
        self.length = torch.load(
            file_path[:-3] + '.length.pt',
            weights_only=True
        )

        # Absolute starting offset of every song.
        self.start = torch.cumsum(self.length, dim=0) - self.length

        self.song_indices = torch.arange(len(self.start))

        # A song must contain at least one complete 384-step window.
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

        else:
            raise ValueError(
                f"Unknown split '{split}'. "
                "Expected 'all', 'train', 'val', or 'test'."
            )

        self.valid_song_count = len(self.valid_indices)

        print(
            'Metadata for dataset',
            file_path,
            'split',
            split,
            'loaded.',
            'Number of valid songs:',
            self.valid_song_count,
            'first 20:',
            self.valid_indices[:20]
        )

        self.data = None
        self.pitch_shift_range = None

    def _window_starts(self, song_length):
        """
        Return relative window starts for one song.

        Regular windows advance by window_step.

        The final window is forced to end exactly at the end of the song,
        unless the regular sequence already reaches that position.

        Example:

            target_length = 384
            window_step = 192
            song_length = 650

            max_start = 266

            result:
                [0, 192, 266]
        """

        max_start = song_length - self.target_length

        starts = list(range(0, max_start + 1, self.window_step))

        if not starts:
            # Should never happen because invalid songs were excluded,
            # but keep this defensive guard.
            return []

        if starts[-1] != max_start:
            starts.append(max_start)

        return starts

    def _build_window_index(self, song_index):
        """
        Build absolute [start, end) window descriptors for one song.
        """

        song_length = int(self.length[song_index].item())
        song_start = int(self.start[song_index].item())

        relative_starts = self._window_starts(song_length)

        return [
            (
                song_index,
                song_start + relative_start
            )
            for relative_start in relative_starts
        ]

    def _build_all_windows(self):
        """
        Build the complete list of windows for the selected songs.

        Each item is:
            (song_index, absolute_start)
        """

        windows = []

        for song_index in self.valid_indices.tolist():
            windows.extend(self._build_window_index(song_index))

        return windows

    def __iter__(self):
        # Each DataLoader worker gets its own dataset object/state.
        if self.data is None:
            self.data = torch.load(
                self.file_path,
                weights_only=True
            )

            self.pitch_shift_range = torch.load(
                self.file_path[:-3] + '.pitch_shift_range.pt',
                weights_only=True
            ).reshape(-1, 2)

            # Same pitch-shift limits as the original FramedDataset.
            self.pitch_shift_range[
                self.pitch_shift_range[:, 0] < -5, 0
            ] = -5

            self.pitch_shift_range[
                self.pitch_shift_range[:, 1] > 6, 1
            ] = 6

            # No pitch augmentation during validation/test.
            if self.split == 'val' or self.split == 'test':
                self.pitch_shift_range = torch.zeros_like(
                    self.pitch_shift_range
                )

            print(
                'Data for dataset',
                self.file_path,
                'loaded.'
            )

        while True:
            windows = self._build_all_windows()

            if self.random_order:
                permutation = torch.randperm(len(windows))
                windows = [windows[i] for i in permutation.tolist()]

            # Construct batches from the shuffled window list.
            for i in range(0, len(windows), self.batch_size):
                batch_windows = windows[i:i + self.batch_size]

                batch_data = []
                batch_pitch_shift = []

                for song_index, absolute_start in batch_windows:
                    absolute_end = absolute_start + self.target_length

                    batch_data.append(
                        self.data[absolute_start:absolute_end]
                    )

                    shift_range = self.pitch_shift_range[song_index]

                    min_shift = int(shift_range[0].item())
                    max_shift = int(shift_range[1].item())

                    if min_shift == max_shift:
                        pitch_shift = min_shift
                    else:
                        pitch_shift = int(
                            torch.randint(
                                min_shift,
                                max_shift + 1,
                                (1,)
                            ).item()
                        )

                    batch_pitch_shift.append(pitch_shift)

                yield (
                    torch.stack(batch_data, dim=0),
                    torch.tensor(
                        batch_pitch_shift,
                        dtype=torch.long
                    )
                )

            if not self.repeat:
                break
