import Foundation
import MLX

/// GPU memory telemetry for a host app's "local runtime" status display. Surfaces MLX's process-global
/// GPU allocator counters — the same ones `OwnedRenderBench` samples — as plain `Int` bytes so a module
/// that does NOT (and must not) link MLX can read them. Orbital uses this from `orbital-loop` (the only
/// MLX-linked process) to publish what the on-device runtime occupies into the room store for its MLX-free
/// GUI. All counters are whole-process / whole-device, not per-model.
public enum MLXRuntimeStats {
    /// Bytes of GPU memory in active use right now (live model weights + KV cache).
    public static var activeBytes: Int { MLX.GPU.activeMemory }
    /// Bytes held in the recycled buffer pool (cached, not currently in active use).
    public static var cacheBytes: Int { MLX.GPU.cacheMemory }
    /// High-water active bytes since process start.
    public static var peakBytes: Int { MLX.GPU.peakMemory }
    /// The device's recommended maximum working-set size (the natural % denominator); 0 if unavailable.
    public static var recommendedWorkingSetBytes: Int { MLX.GPU.maxRecommendedWorkingSetBytes() ?? 0 }
}
