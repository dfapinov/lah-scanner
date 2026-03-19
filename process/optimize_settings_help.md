### The Purpose
The purpose of this script is purely to find the **optimal solver settings** in terms of the maximum expansion order ($N$), regularization thresholds (dB), and the damping factor ($\lambda$).

### Finding the Optimal Order ($N$)
The key to finding the optimal order $N$ lies in the fact that the highest frequency range analyzed is within the **RFT (Reflection Free Time)**, where the sound field is intrinsically anechoic.

* **High Spatial Detail:** The highest frequencies hold the greatest potential for high spatial angular detail because the wavelengths are short. Capturing this detail requires the highest possible order $N$ for an accurate fit.
* **Anechoic Field:** Because the field is anechoic, when fit correctly, the internal source (DUT) energy should be high, while the external source (the room) energy should be minimal.

If the chosen order $N$ is stable and well-conditioned, the internal-to-external field energy should have a large ratio, such as **20 dB or greater**. However, if the order $N$ is too high for the provided dataset, the matrix becomes ill-conditioned, meaning it begins fitting noise or spatial aliasing artifacts of limited measurement grid resolution.

This manifests as the solver no longer knowing if energy should be placed in the internal or external field coefficients, causing the ratio of internal to external energy to drop. This usually happens in a sudden way (a **"tipping point"** in the curve), which helps us detect the highest order $N$ that returns maximum spatial detail while remaining stable and well-conditioned.

---

### The Role of Regularization
The order $N$ increases in discrete steps. While order 6 might be totally stable, order 7 might be "on the edge" of stability—holding a mixture of real, valuable detail alongside artifacts of poor conditioning.

Regularization—particularly the **thresholded regularization** employed here—helps stabilize this conditioning by applying stronger damping to large, unstable coefficients than to small, stable ones. To understand why this works, we have to look at how **Spherical Harmonics (SH)** fit data:

1.  A smooth, coherent acoustic field is easily described by physically reasonable, stable coefficients.
2.  Conversely, when the solver attempts to fit tiny, incoherent spatial patterns (like chaotic noise or aliasing), it is forced to assign massive, non-physical values to high-order coefficients.
3.  It does this so that these complex shapes destructively interfere, canceling each other out at the measurement points to match the tiny noise anomalies.

It is key to understand that while the total field fit might actually be improved by this destructive interference, the coefficients describing the internal and external fields are artificially large. When sound field separation is applied—isolating the internal and external fields—these coefficients no longer cancel each other out. As a result, we see **massive SPL (Sound Pressure Level) values** in both fields that are not based on any physical reality.

By applying a **damping factor ($\lambda$)** on the denominator, these massive, non-physical coefficients are heavily penalized and limited, while the smaller, stable coefficients of the true physical field are left largely unaffected. The end result is the successful damping of ill-conditioning.

---

### Thresholding and Sweeps
The difficulty lies in applying this damping without throwing away desired information. This is where the **dB thresholds** come in. To find the optimal threshold, the script initially applies strong damping so that its effects are obvious.

If this damping is applied too high up and affects the desirable clean signals, the internal-to-external energy ratio drops (which is undesirable). However, as the script sweeps the threshold lower, it identifies the exact point where the damping no longer interferes with those strong signals. Naturally, everything below this threshold is the noise responsible for ill-conditioning.

The final step is to test for the ideal strength of the damping ($\lambda$) to apply only below that threshold. Finding this combination is the **sweet spot**: it allows us to extract a little extra spatial detail while safely avoiding ill-conditioning.

Ultimately, the effect of this regularization is to soften the onset of instability as the order $N$ increases. It is not strictly required for a solve, but it acts as a safety net to make the pipeline more robust.