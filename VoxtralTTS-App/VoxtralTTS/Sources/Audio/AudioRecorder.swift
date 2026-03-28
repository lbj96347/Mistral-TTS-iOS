import AVFoundation
import Foundation

/// Records audio from the microphone using AVAudioEngine.
/// Enforces 3-30 second duration and provides real-time level metering.
class AudioRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var recordingDuration: TimeInterval = 0
    @Published var audioLevel: Float = 0  // 0.0 - 1.0 for UI metering

    /// Minimum recording duration in seconds
    static let minDuration: TimeInterval = 3.0
    /// Maximum recording duration in seconds
    static let maxDuration: TimeInterval = 30.0

    private var audioEngine: AVAudioEngine?
    private var recordedBuffers: [AVAudioPCMBuffer] = []
    private var recordingFormat: AVAudioFormat?
    private var durationTimer: Timer?
    private var recordingStartTime: Date?

    enum RecorderError: LocalizedError {
        case microphonePermissionDenied
        case engineStartFailed(Error)
        case noAudioRecorded
        case tooShort(TimeInterval)
        case conversionFailed

        var errorDescription: String? {
            switch self {
            case .microphonePermissionDenied:
                return "Microphone permission is required to record audio."
            case .engineStartFailed(let error):
                return "Failed to start recording: \(error.localizedDescription)"
            case .noAudioRecorded:
                return "No audio was recorded."
            case .tooShort(let duration):
                return String(format: "Recording too short (%.1fs). Minimum is %.0fs.", duration, AudioRecorder.minDuration)
            case .conversionFailed:
                return "Failed to process recorded audio."
            }
        }
    }

    // MARK: - Permission

    func requestPermission() async -> Bool {
        await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    // MARK: - Recording

    func startRecording() throws {
        guard !isRecording else { return }

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        recordedBuffers = []
        recordingFormat = inputFormat
        recordingDuration = 0
        audioLevel = 0

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { [weak self] buffer, _ in
            guard let self = self else { return }
            self.recordedBuffers.append(buffer)

            // Calculate RMS level for metering
            let level = Self.rmsLevel(buffer: buffer)
            DispatchQueue.main.async {
                self.audioLevel = level
            }
        }

        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true)
        #endif

        do {
            try engine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            throw RecorderError.engineStartFailed(error)
        }

        self.audioEngine = engine
        self.isRecording = true
        self.recordingStartTime = Date()

        // Timer for duration tracking and auto-stop at max
        durationTimer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            guard let self = self, let start = self.recordingStartTime else { return }
            let elapsed = Date().timeIntervalSince(start)
            DispatchQueue.main.async {
                self.recordingDuration = elapsed
            }
            if elapsed >= Self.maxDuration {
                DispatchQueue.main.async {
                    _ = try? self.stopRecording()
                }
            }
        }
    }

    /// Stops recording and returns the combined audio buffer.
    /// Throws if recording is too short (< 3s) or no audio was captured.
    @discardableResult
    func stopRecording() throws -> AVAudioPCMBuffer {
        durationTimer?.invalidate()
        durationTimer = nil

        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil

        isRecording = false
        audioLevel = 0

        guard !recordedBuffers.isEmpty, let format = recordingFormat else {
            throw RecorderError.noAudioRecorded
        }

        let duration = recordingDuration
        if duration < Self.minDuration {
            recordedBuffers = []
            throw RecorderError.tooShort(duration)
        }

        // Combine all buffers into one
        let totalFrames = recordedBuffers.reduce(0) { $0 + Int($1.frameLength) }
        guard let combined = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(totalFrames)) else {
            throw RecorderError.conversionFailed
        }
        combined.frameLength = AVAudioFrameCount(totalFrames)

        var offset = 0
        for buffer in recordedBuffers {
            let frames = Int(buffer.frameLength)
            for ch in 0 ..< Int(format.channelCount) {
                let src = buffer.floatChannelData![ch]
                let dst = combined.floatChannelData![ch]
                memcpy(dst.advanced(by: offset), src, frames * MemoryLayout<Float>.size)
            }
            offset += frames
        }

        recordedBuffers = []

        #if os(iOS)
        // Reset audio session for playback
        try? AVAudioSession.sharedInstance().setCategory(.playback)
        #endif

        return combined
    }

    func cancelRecording() {
        durationTimer?.invalidate()
        durationTimer = nil
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isRecording = false
        audioLevel = 0
        recordedBuffers = []
    }

    // MARK: - Level Metering

    private static func rmsLevel(buffer: AVAudioPCMBuffer) -> Float {
        guard let channelData = buffer.floatChannelData else { return 0 }
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return 0 }

        let samples = channelData[0]
        var sum: Float = 0
        for i in 0 ..< frames {
            sum += samples[i] * samples[i]
        }
        let rms = sqrt(sum / Float(frames))
        // Normalize to 0-1 range (typical speech RMS is 0.01-0.3)
        return min(rms * 5.0, 1.0)
    }

    deinit {
        cancelRecording()
    }
}
