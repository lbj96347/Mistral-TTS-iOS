import AVFoundation
import Foundation

/// Preprocesses audio for voice cloning: resamples to 24kHz mono, normalizes, trims silence, encodes as WAV.
enum AudioPreprocessor {
    static let targetSampleRate: Double = 24000
    static let minDuration: Double = 3.0
    static let maxDuration: Double = 30.0

    enum PreprocessError: LocalizedError {
        case invalidFormat
        case resampleFailed
        case tooShort(Double)
        case tooLong(Double)
        case emptyAfterTrim
        case fileLoadFailed(Error)

        var errorDescription: String? {
            switch self {
            case .invalidFormat:
                return "Unsupported audio format."
            case .resampleFailed:
                return "Failed to resample audio to 24kHz."
            case .tooShort(let d):
                return String(format: "Audio too short (%.1fs). Minimum is %.0fs.", d, minDuration)
            case .tooLong(let d):
                return String(format: "Audio too long (%.1fs). Maximum is %.0fs.", d, maxDuration)
            case .emptyAfterTrim:
                return "Audio is silent — no speech detected."
            case .fileLoadFailed(let error):
                return "Failed to load audio file: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - Public API

    /// Process a recorded buffer into WAV data suitable for the Mistral API.
    static func process(buffer: AVAudioPCMBuffer) throws -> Data {
        let mono = try convertToMono(buffer: buffer)
        let resampled = try resample(buffer: mono, targetRate: targetSampleRate)
        let trimmed = trimSilence(buffer: resampled)

        let duration = Double(trimmed.frameLength) / targetSampleRate
        if duration < minDuration {
            throw PreprocessError.tooShort(duration)
        }
        if duration > maxDuration {
            throw PreprocessError.tooLong(duration)
        }

        let normalized = normalize(buffer: trimmed)
        return encodeWAV(buffer: normalized)
    }

    /// Load and process an audio file from disk.
    static func process(fileURL: URL) throws -> Data {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: fileURL)
        } catch {
            throw PreprocessError.fileLoadFailed(error)
        }

        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: file.fileFormat.sampleRate,
            channels: file.fileFormat.channelCount,
            interleaved: false
        ) else {
            throw PreprocessError.invalidFormat
        }

        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw PreprocessError.invalidFormat
        }

        try file.read(into: buffer)
        return try process(buffer: buffer)
    }

    // MARK: - Mono Conversion

    private static func convertToMono(buffer: AVAudioPCMBuffer) throws -> AVAudioPCMBuffer {
        let channels = Int(buffer.format.channelCount)
        if channels == 1 { return buffer }

        guard let monoFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: buffer.format.sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw PreprocessError.invalidFormat
        }

        let frames = Int(buffer.frameLength)
        guard let mono = AVAudioPCMBuffer(pcmFormat: monoFormat, frameCapacity: buffer.frameCapacity) else {
            throw PreprocessError.invalidFormat
        }
        mono.frameLength = buffer.frameLength

        let dst = mono.floatChannelData![0]
        // Average all channels
        for i in 0 ..< frames {
            var sum: Float = 0
            for ch in 0 ..< channels {
                sum += buffer.floatChannelData![ch][i]
            }
            dst[i] = sum / Float(channels)
        }

        return mono
    }

    // MARK: - Resampling

    private static func resample(buffer: AVAudioPCMBuffer, targetRate: Double) throws -> AVAudioPCMBuffer {
        let sourceRate = buffer.format.sampleRate
        if abs(sourceRate - targetRate) < 1.0 { return buffer }

        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: targetRate,
            channels: 1,
            interleaved: false
        ) else {
            throw PreprocessError.resampleFailed
        }

        guard let converter = AVAudioConverter(from: buffer.format, to: targetFormat) else {
            throw PreprocessError.resampleFailed
        }

        let ratio = targetRate / sourceRate
        let estimatedFrames = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 100
        guard let output = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: estimatedFrames) else {
            throw PreprocessError.resampleFailed
        }

        var error: NSError?
        var consumed = false
        let status = converter.convert(to: output, error: &error) { _, outStatus in
            if consumed {
                outStatus.pointee = .noDataNow
                return nil
            }
            consumed = true
            outStatus.pointee = .haveData
            return buffer
        }

        if status == .error, let error = error {
            throw PreprocessError.fileLoadFailed(error)
        }

        return output
    }

    // MARK: - Silence Trimming

    private static func trimSilence(buffer: AVAudioPCMBuffer, threshold: Float = 0.01) -> AVAudioPCMBuffer {
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return buffer }
        let samples = buffer.floatChannelData![0]

        // Find first non-silent frame (use windows of 480 samples = 20ms at 24kHz)
        let windowSize = 480
        var startFrame = 0
        var endFrame = frames

        // Trim leading silence
        for i in stride(from: 0, to: frames - windowSize, by: windowSize) {
            var energy: Float = 0
            for j in i ..< min(i + windowSize, frames) {
                energy += samples[j] * samples[j]
            }
            energy = sqrt(energy / Float(windowSize))
            if energy > threshold {
                startFrame = max(0, i - windowSize)  // Keep a little padding
                break
            }
        }

        // Trim trailing silence
        for i in stride(from: frames - windowSize, through: startFrame, by: -windowSize) {
            var energy: Float = 0
            for j in i ..< min(i + windowSize, frames) {
                energy += samples[j] * samples[j]
            }
            energy = sqrt(energy / Float(windowSize))
            if energy > threshold {
                endFrame = min(frames, i + 2 * windowSize)  // Keep a little padding
                break
            }
        }

        let trimmedLength = endFrame - startFrame
        guard trimmedLength > 0 else { return buffer }

        guard let trimmed = AVAudioPCMBuffer(pcmFormat: buffer.format, frameCapacity: AVAudioFrameCount(trimmedLength)) else {
            return buffer
        }
        trimmed.frameLength = AVAudioFrameCount(trimmedLength)
        memcpy(trimmed.floatChannelData![0], samples.advanced(by: startFrame), trimmedLength * MemoryLayout<Float>.size)
        return trimmed
    }

    // MARK: - Normalization

    private static func normalize(buffer: AVAudioPCMBuffer) -> AVAudioPCMBuffer {
        let frames = Int(buffer.frameLength)
        guard frames > 0 else { return buffer }
        let samples = buffer.floatChannelData![0]

        var maxAmp: Float = 0
        for i in 0 ..< frames {
            maxAmp = max(maxAmp, abs(samples[i]))
        }

        if maxAmp > 0 && maxAmp != 1.0 {
            let scale = 0.95 / maxAmp
            for i in 0 ..< frames {
                samples[i] *= scale
            }
        }

        return buffer
    }

    // MARK: - WAV Encoding

    /// Encode a 24kHz mono Float32 buffer as WAV data.
    static func encodeWAV(buffer: AVAudioPCMBuffer) -> Data {
        let frames = Int(buffer.frameLength)
        let sampleRate = Int(buffer.format.sampleRate)
        let bytesPerSample = 2  // 16-bit PCM
        let dataSize = frames * bytesPerSample

        var data = Data()
        data.reserveCapacity(44 + dataSize)

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        appendUInt32(&data, UInt32(36 + dataSize))
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        appendUInt32(&data, 16)                          // chunk size
        appendUInt16(&data, 1)                           // PCM format
        appendUInt16(&data, 1)                           // mono
        appendUInt32(&data, UInt32(sampleRate))          // sample rate
        appendUInt32(&data, UInt32(sampleRate * bytesPerSample))  // byte rate
        appendUInt16(&data, UInt16(bytesPerSample))      // block align
        appendUInt16(&data, 16)                          // bits per sample

        // data chunk
        data.append(contentsOf: "data".utf8)
        appendUInt32(&data, UInt32(dataSize))

        // Convert Float32 to Int16
        let samples = buffer.floatChannelData![0]
        for i in 0 ..< frames {
            let clamped = max(-1.0, min(1.0, samples[i]))
            let int16 = Int16(clamped * 32767.0)
            appendInt16(&data, int16)
        }

        return data
    }

    // MARK: - Binary Helpers

    private static func appendUInt32(_ data: inout Data, _ value: UInt32) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 4))
    }

    private static func appendUInt16(_ data: inout Data, _ value: UInt16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }

    private static func appendInt16(_ data: inout Data, _ value: Int16) {
        var v = value.littleEndian
        data.append(Data(bytes: &v, count: 2))
    }
}
