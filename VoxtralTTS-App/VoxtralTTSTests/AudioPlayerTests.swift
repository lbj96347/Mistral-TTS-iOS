import XCTest
import MLX
import AVFoundation

/// Tests for AudioPlayer using synthetic data -- no model needed.
final class AudioPlayerTests: XCTestCase {

    // MARK: - Load Audio

    func testLoadAudioDuration() {
        let player = AudioPlayer(sampleRate: 24000)
        // 24000 samples at 24kHz = exactly 1 second
        let samples = MLXArray(Array(repeating: Float(0), count: 24000))
        player.loadAudio(from: samples)

        XCTAssertEqual(player.duration, 1.0, accuracy: 0.001,
                       "24000 samples at 24kHz should be 1.0 second")
    }

    func testLoadAudioVariousDurations() {
        let player = AudioPlayer(sampleRate: 24000)

        let testCases: [(sampleCount: Int, expectedDuration: Double)] = [
            (12000, 0.5),
            (48000, 2.0),
            (2400, 0.1),
            (240000, 10.0),
        ]

        for (sampleCount, expectedDuration) in testCases {
            let samples = MLXArray(Array(repeating: Float(0), count: sampleCount))
            player.loadAudio(from: samples)

            XCTAssertEqual(player.duration, expectedDuration, accuracy: 0.001,
                           "\(sampleCount) samples should be \(expectedDuration)s")
        }
    }

    @MainActor
    func testLoadAudioResetsCurrentTime() async throws {
        let player = AudioPlayer(sampleRate: 24000)
        let samples = MLXArray(Array(repeating: Float(0), count: 24000))
        player.loadAudio(from: samples)

        // Give the DispatchQueue.main.async inside loadAudio time to execute
        try await Task.sleep(nanoseconds: 200_000_000)

        XCTAssertEqual(player.currentTime, 0, accuracy: 0.001)
    }

    // MARK: - Save to File

    func testSaveToFile() throws {
        let player = AudioPlayer(sampleRate: 24000)

        // Generate a 0.5s sine wave to have non-trivial content
        let sampleCount = 12000
        var samples = [Float]()
        for i in 0..<sampleCount {
            let t = Float(i) / 24000.0
            samples.append(sin(2.0 * .pi * 440.0 * t) * 0.5) // 440Hz sine
        }
        player.loadAudio(from: MLXArray(samples))

        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_output_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        try player.saveToFile(url: tempURL)

        XCTAssertTrue(FileManager.default.fileExists(atPath: tempURL.path),
                       "Saved WAV file should exist")

        let attributes = try FileManager.default.attributesOfItem(atPath: tempURL.path)
        let fileSize = attributes[.size] as? Int ?? 0
        XCTAssertGreaterThan(fileSize, 0, "Saved WAV file should have non-zero size")

        // WAV header is 44 bytes minimum; file should be larger than just headers
        XCTAssertGreaterThan(fileSize, 100, "WAV file should contain audio data beyond header")
    }

    func testSaveWithoutAudioThrows() {
        let player = AudioPlayer(sampleRate: 24000)
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("test_empty_\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: tempURL) }

        do {
            try player.saveToFile(url: tempURL)
            XCTFail("Saving without loaded audio should throw")
        } catch {
            // Expected -- saveToFile throws when no audio is loaded
        }
    }

    // MARK: - Time Formatting

    func testTimeFormatting() {
        let cases: [(TimeInterval, String)] = [
            (0.0, "0:00"),
            (5.0, "0:05"),
            (65.0, "1:05"),
            (125.0, "2:05"),
        ]
        for (interval, expected) in cases {
            XCTAssertEqual(interval.formattedTime, expected,
                           "\(interval)s should format as '\(expected)'")
        }
    }
}
