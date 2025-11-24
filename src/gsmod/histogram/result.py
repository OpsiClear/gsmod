"""Histogram result dataclass with analysis methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass
class HistogramResult:
    """Histogram computation results with analysis methods.

    Attributes:
        counts: Histogram bin counts, shape [n_bins] or [n_channels, n_bins]
        bin_edges: Bin edges, shape [n_bins + 1]
        mean: Mean value(s), scalar or [n_channels]
        std: Standard deviation(s), scalar or [n_channels]
        min_val: Minimum value(s), scalar or [n_channels]
        max_val: Maximum value(s), scalar or [n_channels]
        n_samples: Total number of samples

    Example:
        >>> result = data.histogram_colors()
        >>> print(f"Mean RGB: {result.mean}")
        >>> print(f"Peak bin: {result.bin_centers[result.counts[0].argmax()]}")
    """

    counts: np.ndarray
    bin_edges: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    min_val: np.ndarray
    max_val: np.ndarray
    n_samples: int

    @property
    def bin_centers(self) -> np.ndarray:
        """Get bin center values.

        :return: Array of bin centers, shape [n_bins]
        """
        return (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

    @property
    def n_bins(self) -> int:
        """Get number of bins.

        :return: Number of histogram bins
        """
        return len(self.bin_edges) - 1

    @property
    def n_channels(self) -> int:
        """Get number of channels (1 for single property, 3 for RGB).

        :return: Number of channels
        """
        if self.counts.ndim == 1:
            return 1
        return self.counts.shape[0]

    def percentile(self, p: float, channel: int = 0) -> float:
        """Compute percentile from histogram.

        :param p: Percentile value (0-100)
        :param channel: Channel index for multi-channel histograms
        :return: Value at percentile
        """
        counts = self.counts if self.counts.ndim == 1 else self.counts[channel]
        cumsum = np.cumsum(counts)
        total = cumsum[-1]
        if total == 0:
            return float(self.bin_centers[0])

        target = total * (p / 100.0)
        idx = np.searchsorted(cumsum, target)
        idx = min(idx, len(self.bin_centers) - 1)
        return float(self.bin_centers[idx])

    def mode(self, channel: int = 0) -> float:
        """Get most frequent value (mode).

        :param channel: Channel index for multi-channel histograms
        :return: Bin center of most frequent bin
        """
        counts = self.counts if self.counts.ndim == 1 else self.counts[channel]
        idx = np.argmax(counts)
        return float(self.bin_centers[idx])

    def entropy(self, channel: int = 0) -> float:
        """Compute entropy of distribution.

        :param channel: Channel index for multi-channel histograms
        :return: Shannon entropy in bits
        """
        counts = self.counts if self.counts.ndim == 1 else self.counts[channel]
        total = counts.sum()
        if total == 0:
            return 0.0

        probs = counts / total
        # Avoid log(0)
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log2(probs)))

    def dynamic_range(self, percentile_low: float = 1.0, percentile_high: float = 99.0) -> float:
        """Compute dynamic range using percentiles.

        :param percentile_low: Low percentile (default 1%)
        :param percentile_high: High percentile (default 99%)
        :return: Dynamic range (high - low)
        """
        if self.n_channels == 1:
            low = self.percentile(percentile_low)
            high = self.percentile(percentile_high)
        else:
            # Use average across channels
            low = np.mean([self.percentile(percentile_low, c) for c in range(self.n_channels)])
            high = np.mean([self.percentile(percentile_high, c) for c in range(self.n_channels)])
        return float(high - low)

    def to_color_values(self, profile: str = "neutral"):
        """Generate ColorValues to optimize histogram for target profile.

        Analyzes current distribution and suggests adjustments.

        Profiles:
        - "neutral": Flatten histogram toward uniform distribution
        - "vibrant": Boost saturation, increase mid-tone contrast
        - "dramatic": Strong contrast, darker shadows
        - "bright": Shift distribution higher
        - "dark": Shift distribution lower

        :param profile: Target histogram profile
        :return: ColorValues to apply

        Example:
            >>> result = data.histogram_colors()
            >>> adjustment = result.to_color_values("vibrant")
            >>> data.color(adjustment)
        """
        from gsmod.config.values import ColorValues

        # Analyze current distribution
        if self.n_channels == 3:
            avg_mean = float(np.mean(self.mean))
            avg_std = float(np.mean(self.std))
        else:
            avg_mean = float(self.mean) if np.isscalar(self.mean) else float(self.mean[0])
            avg_std = float(self.std) if np.isscalar(self.std) else float(self.std[0])

        # Current contrast proxy (std normalized to [0,1] range)
        current_contrast = avg_std / 0.289  # 0.289 = std of uniform [0,1]
        current_brightness = avg_mean / 0.5  # 0.5 = mean of uniform [0,1]

        if profile == "neutral":
            # Flatten toward uniform - adjust gamma and contrast
            # If too bright, lower gamma; if too dark, raise gamma
            target_mean = 0.5
            gamma = 1.0
            if avg_mean > 0.01:
                # Approximate gamma to shift mean toward target
                gamma = np.log(target_mean) / np.log(avg_mean)
                gamma = max(0.5, min(2.0, gamma))

            # Adjust contrast toward normal
            contrast = 1.0 / current_contrast if current_contrast > 0.1 else 1.0
            contrast = max(0.5, min(2.0, contrast))

            return ColorValues(gamma=gamma, contrast=contrast)

        elif profile == "vibrant":
            # Boost saturation and mid-tone presence
            saturation = 1.3
            vibrance = 1.2
            contrast = 1.05
            # Slight gamma adjustment to emphasize mid-tones
            gamma = 0.95 if avg_mean > 0.5 else 1.05

            return ColorValues(
                saturation=saturation,
                vibrance=vibrance,
                contrast=contrast,
                gamma=gamma,
            )

        elif profile == "dramatic":
            # Strong contrast, darker shadows
            contrast = 1.3
            shadows = -0.15
            highlights = 0.05
            saturation = 1.1

            return ColorValues(
                contrast=contrast,
                shadows=shadows,
                highlights=highlights,
                saturation=saturation,
            )

        elif profile == "bright":
            # Shift distribution higher
            brightness = 1.3 / current_brightness if current_brightness > 0.1 else 1.3
            brightness = max(1.0, min(2.0, brightness))
            gamma = 0.9  # Lift shadows

            return ColorValues(brightness=brightness, gamma=gamma)

        elif profile == "dark":
            # Shift distribution lower
            brightness = 0.7 / current_brightness if current_brightness > 0.1 else 0.7
            brightness = max(0.5, min(1.0, brightness))
            gamma = 1.1  # Deepen shadows

            return ColorValues(brightness=brightness, gamma=gamma)

        else:
            raise ValueError(
                f"Unknown profile: {profile}. "
                f"Available: neutral, vibrant, dramatic, bright, dark"
            )

    def learn_from(
        self,
        source_colors: "torch.Tensor",
        params: list[str] | None = None,
        n_epochs: int = 100,
        lr: float = 0.02,
        verbose: bool = False,
    ):
        """Learn ColorValues to match this histogram distribution.

        Uses gradient descent to optimize color adjustment parameters
        that transform source_colors to match this target histogram.

        :param source_colors: Source RGB colors as PyTorch tensor [N, 3]
        :param params: Parameters to learn (default: brightness, contrast, gamma, saturation)
        :param n_epochs: Number of optimization epochs
        :param lr: Learning rate
        :param verbose: Print progress
        :return: Learned ColorValues

        Example:
            >>> # Get target histogram from reference
            >>> target = reference_data.histogram_colors()
            >>>
            >>> # Learn color grading to match
            >>> source_colors = torch.tensor(source_data.sh0, device='cuda')
            >>> learned = target.learn_from(source_colors, n_epochs=200)
            >>>
            >>> # Apply learned values
            >>> source_data.color(learned)
        """
        import torch
        import torch.optim as optim

        from gsmod.config.values import ColorValues
        from gsmod.histogram.loss import MomentMatchingLoss
        from gsmod.torch.learn import LearnableColor

        # Ensure source is on correct device
        if not isinstance(source_colors, torch.Tensor):
            source_colors = torch.tensor(source_colors, dtype=torch.float32)

        device = source_colors.device
        source_colors = source_colors.detach()

        # Create learnable model
        if params is None:
            params = ['brightness', 'contrast', 'gamma', 'saturation']

        model = LearnableColor.from_values(ColorValues(), params).to(device)

        # Use moment matching loss (fastest and most accurate for parameter recovery)
        loss_fn = MomentMatchingLoss().to(device)

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Training loop
        best_loss = float('inf')
        best_values = None

        for epoch in range(n_epochs):
            optimizer.zero_grad()

            # Apply color adjustment
            adjusted = model(source_colors)

            # Compute loss
            loss = loss_fn(adjusted, self)

            # Backprop
            loss.backward()
            optimizer.step()

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_values = model.to_values()

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}: loss = {loss.item():.6f}")

        if verbose:
            print(f"Final loss: {best_loss:.6f}")

        return best_values

    @classmethod
    def empty(cls, n_bins: int = 256, n_channels: int = 1) -> HistogramResult:
        """Create empty histogram result.

        :param n_bins: Number of bins
        :param n_channels: Number of channels
        :return: Empty HistogramResult
        """
        if n_channels == 1:
            counts = np.zeros(n_bins, dtype=np.int64)
            mean = np.array(0.0, dtype=np.float64)
            std = np.array(0.0, dtype=np.float64)
            min_val = np.array(0.0, dtype=np.float64)
            max_val = np.array(0.0, dtype=np.float64)
        else:
            counts = np.zeros((n_channels, n_bins), dtype=np.int64)
            mean = np.zeros(n_channels, dtype=np.float64)
            std = np.zeros(n_channels, dtype=np.float64)
            min_val = np.zeros(n_channels, dtype=np.float64)
            max_val = np.zeros(n_channels, dtype=np.float64)

        return cls(
            counts=counts,
            bin_edges=np.linspace(0.0, 1.0, n_bins + 1),
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            n_samples=0,
        )
