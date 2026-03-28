import AVFoundation
import Foundation
import MLX

class AudioPlayer: ObservableObject {
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0

    private var audioEngine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var audioBuffer: AVAudioPCMBuffer?
    private var displayTimer: Timer?

    let sampleRate: Double

    init(sampleRate: Int = 24000) {
        self.sampleRate = Double(sampleRate)
    }

    func loadAudio(from mlxArray: MLXArray) {
        // Convert MLX array to Float32 array
        let count = mlxArray.dim(0)
        let floatArray = mlxArray.asType(.float32)

        // Create audio format: mono, 24kHz, float32
        let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        )!

        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(count)) else {
            return
        }

        buffer.frameLength = AVAudioFrameCount(count)

        // Copy samples from MLX to buffer, normalizing to [-1, 1]
        let channelData = buffer.floatChannelData![0]
        let np = floatArray.asArray(Float.self)
        let maxAmp = np.reduce(0) { max($0, abs($1)) }
        let scale: Float = maxAmp > 1.0 ? (0.95 / maxAmp) : 1.0
        for i in 0 ..< count {
            channelData[i] = np[i] * scale
        }

        self.audioBuffer = buffer
        self.duration = Double(count) / sampleRate

        DispatchQueue.main.async {
            self.currentTime = 0
        }
    }

    func play() {
        guard let buffer = audioBuffer else { return }

        stop()

        let engine = AVAudioEngine()
        let player = AVAudioPlayerNode()

        engine.attach(player)
        engine.connect(player, to: engine.mainMixerNode, format: buffer.format)

        do {
            try engine.start()
        } catch {
            print("Audio engine failed to start: \(error)")
            return
        }

        player.scheduleBuffer(buffer) { [weak self] in
            DispatchQueue.main.async {
                self?.isPlaying = false
                self?.stopTimer()
            }
        }

        player.play()

        self.audioEngine = engine
        self.playerNode = player
        self.isPlaying = true

        startTimer()
    }

    func pause() {
        playerNode?.pause()
        isPlaying = false
        stopTimer()
    }

    func resume() {
        playerNode?.play()
        isPlaying = true
        startTimer()
    }

    func stop() {
        playerNode?.stop()
        audioEngine?.stop()
        audioEngine = nil
        playerNode = nil
        isPlaying = false
        stopTimer()
        DispatchQueue.main.async {
            self.currentTime = 0
        }
    }

    func togglePlayback() {
        if isPlaying {
            pause()
        } else if playerNode != nil {
            resume()
        } else {
            play()
        }
    }

    // MARK: - Save to File

    func saveToFile(url: URL) throws {
        guard let buffer = audioBuffer else {
            throw NSError(domain: "AudioPlayer", code: 1, userInfo: [NSLocalizedDescriptionKey: "No audio loaded"])
        }

        let file = try AVAudioFile(
            forWriting: url,
            settings: buffer.format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )

        try file.write(from: buffer)
    }

    // MARK: - Timer

    private func startTimer() {
        stopTimer()
        displayTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self = self, let player = self.playerNode,
                  let nodeTime = player.lastRenderTime,
                  let playerTime = player.playerTime(forNodeTime: nodeTime) else { return }

            DispatchQueue.main.async {
                self.currentTime = Double(playerTime.sampleTime) / playerTime.sampleRate
            }
        }
    }

    private func stopTimer() {
        displayTimer?.invalidate()
        displayTimer = nil
    }

    deinit {
        stop()
    }
}

// MARK: - Time Formatting

extension TimeInterval {
    var formattedTime: String {
        let mins = Int(self) / 60
        let secs = Int(self) % 60
        return String(format: "%d:%02d", mins, secs)
    }
}
