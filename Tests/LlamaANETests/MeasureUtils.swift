import Foundation
import Testing

/// A container type representing a time value that is suitable for storage,
/// conversion, encoding, and decoding.
///
/// This type models time values as durations. When representing a timestamp, an
/// instance of this type represents that timestamp as an offset from an epoch
/// such as the January 1, 1970 POSIX epoch or the system's boot time; which
/// epoch depends on the calling code.
///
/// This type is not part of the public interface of the testing library. Time
/// values exposed to clients of the testing library should generally be
/// represented as instances of ``Test/Clock/Instant`` or a type from the Swift
/// standard library like ``Duration``.
struct TimeValue: Sendable {
  /// The number of whole seconds represented by this instance.
  var seconds: Int64

  /// The number of attoseconds (that is, the subsecond part) represented by
  /// this instance.
  var attoseconds: Int64

  /// The amount of time represented by this instance as a tuple.
  var components: (seconds: Int64, attoseconds: Int64) {
    (seconds, attoseconds)
  }

  init(_ components: (seconds: Int64, attoseconds: Int64)) {
    (seconds, attoseconds) = components
  }

#if !SWT_NO_TIMESPEC
  init(_ timespec: timespec) {
    self.init((Int64(timespec.tv_sec), Int64(timespec.tv_nsec) * 1_000_000_000))
  }
#endif

  init(_ duration: Duration) {
    self.init(duration.components)
  }

  init(_ instant: SuspendingClock.Instant) {
    self.init(unsafeBitCast(instant, to: Duration.self))
  }
}

// MARK: - Equatable, Hashable, Comparable

extension TimeValue: Equatable, Hashable, Comparable {
  static func <(lhs: Self, rhs: Self) -> Bool {
    if lhs.seconds != rhs.seconds {
      return lhs.seconds < rhs.seconds
    }
    return lhs.attoseconds < rhs.attoseconds
  }
}

// MARK: - Codable

extension TimeValue: Codable {}

// MARK: - CustomStringConvertible

extension TimeValue: CustomStringConvertible {
  var description: String {
#if os(WASI)
    // BUG: https://github.com/swiftlang/swift/issues/72398
    return String(describing: Duration(self))
#else
    let (secondsFromAttoseconds, attosecondsRemaining) = attoseconds.quotientAndRemainder(dividingBy: 1_000_000_000_000_000_000)
    let seconds = seconds + secondsFromAttoseconds
    var milliseconds = attosecondsRemaining / 1_000_000_000_000_000
    if seconds == 0 && milliseconds == 0 && attosecondsRemaining > 0 {
      milliseconds = 1
    }

    return withUnsafeTemporaryAllocation(of: CChar.self, capacity: 512) { buffer in
      withVaList([CLongLong(seconds), CInt(milliseconds)]) { args in
        _ = vsnprintf(buffer.baseAddress!, buffer.count, "%lld.%03d seconds", args)
      }
      return String(cString: buffer.baseAddress!)
    }
#endif
  }
}

// MARK: -

extension Duration {
  init(_ timeValue: TimeValue) {
    self.init(secondsComponent: timeValue.seconds, attosecondsComponent: timeValue.attoseconds)
  }
}

extension SuspendingClock.Instant {
  init(_ timeValue: TimeValue) {
    self = unsafeBitCast(Duration(timeValue), to: SuspendingClock.Instant.self)
  }
}

#if !SWT_NO_TIMESPEC
extension timespec {
  init(_ timeValue: TimeValue) {
    self.init(tv_sec: .init(timeValue.seconds), tv_nsec: .init(timeValue.attoseconds / 1_000_000_000))
  }
}
#endif

extension FloatingPoint {
  /// Initialize this floating-point value with the total number of seconds
  /// (including the subsecond part) represented by an instance of
  /// ``TimeValue``.
  ///
  /// - Parameters:
  ///   - timeValue: The instance of ``TimeValue`` to convert.
  ///
  /// The resulting value may have less precision than `timeValue` as most
  /// floating-point types are unable to represent a time value's
  /// ``TimeValue/attoseconds`` property exactly.
  init(_ timeValue: TimeValue) {
    self = Self(timeValue.seconds) + (Self(timeValue.attoseconds) / (1_000_000_000_000_000_000 as Self))
  }
}

@_spi(Experimental) @_spi(ForToolsIntegrationOnly)
extension Test {
  /// A clock used to track time when events occur during testing.
  ///
  /// This clock tracks time using both the [suspending clock](https://developer.apple.com/documentation/swift/suspendingclock)
  /// and the wall clock. Only the suspending clock is used for comparing and
  /// calculating; the wall clock is used for presentation when needed.
  public struct Clock: Sendable {
    /// An instant on the testing clock.
    public struct Instant: Sendable {
      /// The suspending-clock time corresponding to this instant.
      fileprivate(set) var suspending: TimeValue = {
#if !SWT_NO_TIMESPEC && SWT_TARGET_OS_APPLE
        // The testing library's availability on Apple platforms is earlier than
        // that of the Swift Clock API, so we don't use `SuspendingClock`
        // directly on them and instead derive a value from platform-specific
        // API. SuspendingClock corresponds to CLOCK_UPTIME_RAW on Darwin.
        // SEE: https://github.com/swiftlang/swift/blob/main/stdlib/public/Concurrency/Clock.cpp
        var uptime = timespec()
        _ = clock_gettime(CLOCK_UPTIME_RAW, &uptime)
        return TimeValue(uptime)
#else
        /// The corresponding suspending-clock time.
        TimeValue(SuspendingClock.Instant.now)
#endif
      }()

#if !SWT_NO_UTC_CLOCK
      /// The wall-clock time corresponding to this instant.
      fileprivate(set) var wall: TimeValue = {
#if !SWT_NO_TIMESPEC
        var wall = timespec()
#if os(Android)
        // Android headers recommend `clock_gettime` over `timespec_get` which
        // is available with API Level 29+ for `TIME_UTC`.
        clock_gettime(CLOCK_REALTIME, &wall)
#else
        timespec_get(&wall, TIME_UTC)
#endif
        return TimeValue(wall)
#else
#warning("Platform-specific implementation missing: UTC time unavailable (no timespec)")
        return TimeValue((0, 0))
#endif
      }()
#endif

      /// The current time according to the testing clock.
      public static var now: Self {
        Self()
      }
    }

    public init() {}
  }
}

// MARK: -

@_spi(Experimental) @_spi(ForToolsIntegrationOnly)
extension SuspendingClock.Instant {
  /// Initialize this instant to the equivalent of the same instant on the
  /// testing library's clock.
  ///
  /// - Parameters:
  ///   - testClockInstant: The equivalent instant on ``Test/Clock``.
  public init(_ testClockInstant: Test.Clock.Instant) {
    self.init(testClockInstant.suspending)
  }
}

extension Test.Clock.Instant {
#if !SWT_NO_UTC_CLOCK
  /// The duration since 1970 represented by this instance as a tuple of seconds
  /// and attoseconds.
  ///
  /// The value of this property is the equivalent of `self` on the wall clock.
  /// It is suitable for display to the user, but not for fine timing
  /// calculations.
  public var timeComponentsSince1970: (seconds: Int64, attoseconds: Int64) {
    wall.components
  }

  /// The duration since 1970 represented by this instance.
  ///
  /// The value of this property is the equivalent of `self` on the wall clock.
  /// It is suitable for display to the user, but not for fine timing
  /// calculations.
  public var durationSince1970: Duration {
    Duration(wall)
  }
#endif

  /// Get the number of nanoseconds from this instance to another.
  ///
  /// - Parameters:
  ///   - other: The later instant.
  ///
  /// - Returns: The number of nanoseconds between `self` and `other`. If
  ///   `other` is ordered before this instance, the result is negative.
  func nanoseconds(until other: Self) -> Int64 {
    if other < self {
      return -other.nanoseconds(until: self)
    }
    let otherNanoseconds = (other.suspending.seconds * 1_000_000_000) + (other.suspending.attoseconds / 1_000_000_000)
    let selfNanoseconds = (suspending.seconds * 1_000_000_000) + (suspending.attoseconds / 1_000_000_000)
    return otherNanoseconds - selfNanoseconds
  }
}

// MARK: - Sleeping

extension Test.Clock {
  /// Suspend the current task for the given duration.
  ///
  /// - Parameters:
  ///   - duration: How long to suspend for.
  ///
  /// - Throws: `CancellationError` if the current task was cancelled while it
  ///   was sleeping.
  ///
  /// This function is not part of the public interface of the testing library.
  /// It is primarily used by the testing library's own tests. External clients
  /// can use ``sleep(for:tolerance:)`` or ``sleep(until:tolerance:)`` instead.
  static func sleep(for duration: Duration) async throws {
#if !SWT_NO_UNSTRUCTURED_TASKS
    return try await SuspendingClock().sleep(for: duration)
#elseif !SWT_NO_TIMESPEC
    let timeValue = TimeValue(duration)
    var ts = timespec(timeValue)
    var tsRemaining = ts
    while 0 != nanosleep(&ts, &tsRemaining) {
      try Task.checkCancellation()
      ts = tsRemaining
    }
#else
#warning("Platform-specific implementation missing: task sleep unavailable")
#endif
  }
}

// MARK: - Clock

extension Test.Clock: _Concurrency.Clock {
  public typealias Duration = SuspendingClock.Duration

  public var now: Instant {
    .now
  }

  public var minimumResolution: Duration {
#if SWT_TARGET_OS_APPLE
    var res = timespec()
    _ = clock_getres(CLOCK_UPTIME_RAW, &res)
    return Duration(TimeValue(res))
#else
    SuspendingClock().minimumResolution
#endif
  }

  public func sleep(until deadline: Instant, tolerance: Duration?) async throws {
    let duration = Instant.now.duration(to: deadline)
#if SWT_NO_UNSTRUCTURED_TASKS
    try await Self.sleep(for: duration)
#else
    try await SuspendingClock().sleep(for: duration, tolerance: tolerance)
#endif
  }
}

// MARK: - Equatable, Hashable, Comparable

extension Test.Clock.Instant: Equatable, Hashable, Comparable {
  public static func ==(lhs: Self, rhs: Self) -> Bool {
    lhs.suspending == rhs.suspending
  }

  public func hash(into hasher: inout Hasher) {
    hasher.combine(suspending)
  }

  public static func <(lhs: Self, rhs: Self) -> Bool {
    lhs.suspending < rhs.suspending
  }
}

// MARK: - InstantProtocol

extension Test.Clock.Instant: InstantProtocol {
  public typealias Duration = Swift.Duration

  public func advanced(by duration: Duration) -> Self {
    var result = self

    result.suspending = TimeValue(Duration(result.suspending) + duration)
#if !SWT_NO_UTC_CLOCK
    result.wall = TimeValue(Duration(result.wall) + duration)
#endif

    return result
  }

  public func duration(to other: Test.Clock.Instant) -> Duration {
    Duration(other.suspending) - Duration(suspending)
  }
}

// MARK: - Duration descriptions

extension Test.Clock.Instant {
  /// Get a description of the duration between this instance and another.
  ///
  /// - Parameters:
  ///   - other: The later instant.
  ///
  /// - Returns: A string describing the duration between `self` and `other`,
  ///   up to millisecond accuracy.
  func descriptionOfDuration(to other: Test.Clock.Instant) -> String {
#if SWT_TARGET_OS_APPLE
    let (seconds, nanosecondsRemaining) = nanoseconds(until: other).quotientAndRemainder(dividingBy: 1_000_000_000)
    return String(describing: TimeValue((seconds, nanosecondsRemaining * 1_000_000_000)))
#else
    return String(describing: TimeValue(Duration(other.suspending) - Duration(suspending)))
#endif
  }
}
