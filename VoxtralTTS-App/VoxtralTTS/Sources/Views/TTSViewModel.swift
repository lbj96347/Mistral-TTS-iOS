import SwiftUI
import MLX
import Foundation

@MainActor
class TTSViewModel: ObservableObject {
    // UI State
    @Published var inputText = ""
    @Published var selectedVoice = ""
    @Published var isGenerating = false
    @Published var isLoadingModel = false
    @Published var modelLoaded = false
    @Published var hasAudio = false
    @Published var errorMessage: String?
    @Published var loadingStatus = ""
    @Published var generationProgress: Double = 0
    @Published var currentFrame = 0
    @Published var maxFrames = 2048
    @Published var generationStats: GenerationStats?
    @Published var showingDirectoryPicker = false

    // Model
    private var model: VoxtralTTSModel?
    private var tokenizer: VoxtralTokenizer?
    private var modelConfig: ModelConfig?
    private var modelPath: URL?

    // Generation task (for cancellation)
    private var generationTask: Task<Void, Never>?

    // Audio
    let audioPlayer = AudioPlayer(sampleRate: 24000)

    // Persistence
    private let modelPathKey = "VoxtralTTS.modelPath"

    var lastModelPath: String? {
        UserDefaults.standard.string(forKey: modelPathKey)
    }

    var availableVoices: [String] {
        guard let path = modelPath else { return [] }
        let voiceDir = path.appendingPathComponent("voice_embedding")
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: voiceDir, includingPropertiesForKeys: nil
        ) else { return [] }

        return files
            .filter { $0.pathExtension == "safetensors" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
    }

    var playbackProgress: Double {
        guard audioPlayer.duration > 0 else { return 0 }
        return audioPlayer.currentTime / audioPlayer.duration
    }

    // MARK: - Auto-Load from Launch Argument

    func checkAutoLoadArgument() {
        let args = ProcessInfo.processInfo.arguments
        if let idx = args.firstIndex(of: "--model-path"), idx + 1 < args.count {
            let path = args[idx + 1]
            print("[TTS] Auto-loading model from launch argument: \(path)")
            let url = URL(fileURLWithPath: path)
            loadModel(from: url)
        }
    }

    // MARK: - Model Loading

    func loadModel(from url: URL) {
        isLoadingModel = true
        errorMessage = nil
        loadingStatus = "Loading config..."
        print("[TTS] loadModel called with path: \(url.path)")

        // Start security-scoped access for sandboxed apps
        let accessing = url.startAccessingSecurityScopedResource()
        print("[TTS] Security-scoped access: \(accessing)")

        Task {
            let startTime = CFAbsoluteTimeGetCurrent()
            do {
                loadingStatus = "Loading config..."
                print("[TTS] Loading model config and weights...")
                let (loadedModel, config) = try loadVoxtralModel(from: url)
                let modelElapsed = CFAbsoluteTimeGetCurrent() - startTime
                print("[TTS] Model loaded in \(String(format: "%.2f", modelElapsed))s — dim=\(config.dim), layers=\(config.nLayers)")

                loadingStatus = "Loading tokenizer..."
                print("[TTS] Loading tokenizer...")
                let tokenizerStart = CFAbsoluteTimeGetCurrent()
                let loadedTokenizer = try await VoxtralTokenizer(modelPath: url)
                let tokenizerElapsed = CFAbsoluteTimeGetCurrent() - tokenizerStart
                print("[TTS] Tokenizer loaded in \(String(format: "%.2f", tokenizerElapsed))s")

                let totalElapsed = CFAbsoluteTimeGetCurrent() - startTime
                print("[TTS] ✅ Total load time: \(String(format: "%.2f", totalElapsed))s")

                await MainActor.run {
                    self.model = loadedModel
                    self.tokenizer = loadedTokenizer
                    self.modelConfig = config
                    self.modelPath = url
                    self.modelLoaded = true
                    self.isLoadingModel = false

                    // Save path for next launch
                    UserDefaults.standard.set(url.path, forKey: modelPathKey)
                }
            } catch {
                print("[TTS] ❌ Failed to load model: \(error.localizedDescription)")
                print("[TTS] ❌ Full error: \(error)")
                await MainActor.run {
                    self.errorMessage = "Failed to load model: \(error.localizedDescription)"
                    self.isLoadingModel = false
                }
            }

            if accessing {
                url.stopAccessingSecurityScopedResource()
            }
        }
    }

    // MARK: - Frame Estimation

    private func estimateMaxFrames(for text: String) -> Int {
        let wordCount = text.split(whereSeparator: { $0.isWhitespace || $0.isNewline }).count
        // ~5 frames/word at 12.5Hz, 3x safety multiplier
        let estimated = wordCount * 15
        return min(max(estimated, 50), 2048)
    }

    // MARK: - Cancel

    func cancelGeneration() {
        generationTask?.cancel()
    }

    // MARK: - Generation

    func generate() {
        guard let model = model, let tokenizer = tokenizer else { return }
        guard !inputText.isEmpty else { return }

        let estimatedFrames = estimateMaxFrames(for: inputText)

        isGenerating = true
        errorMessage = nil
        generationProgress = 0
        currentFrame = 0
        maxFrames = estimatedFrames
        generationStats = nil

        generationTask = Task.detached { [weak self, inputText, selectedVoice, estimatedFrames] in
            guard let self = self else { return }

            do {
                // Tokenize text only (no BOS/EOS — prompt building adds control tokens)
                let tokenIds = tokenizer.encode(inputText)
                let textTokenIds = tokenIds.map { Int32($0) }

                // Load voice embedding if selected
                var voiceEmb: MLXArray? = nil
                if !selectedVoice.isEmpty, let modelPath = await self.modelPath {
                    let voicePath = modelPath
                        .appendingPathComponent("voice_embedding")
                        .appendingPathComponent("\(selectedVoice).safetensors")
                    if FileManager.default.fileExists(atPath: voicePath.path) {
                        voiceEmb = model.loadVoiceEmbedding(from: voicePath)
                    }
                }

                // Generate with correct TTS prompt template
                let result = model.generate(
                    textTokenIds: textTokenIds,
                    voiceEmbedding: voiceEmb,
                    temperature: 0.0,
                    topP: 1.0,
                    maxAudioFrames: estimatedFrames,
                    progressCallback: { frame, total in
                        Task { @MainActor in
                            self.currentFrame = frame
                            self.maxFrames = total
                            self.generationProgress = Double(frame) / Double(total)
                        }
                    },
                    shouldCancel: {
                        Task.isCancelled
                    }
                )

                if let result = result {
                    await MainActor.run {
                        self.audioPlayer.loadAudio(from: result.audio)
                        self.hasAudio = true
                        self.generationStats = GenerationStats(
                            audioDuration: result.audioDuration,
                            realTimeFactor: result.realTimeFactor,
                            tokenCount: result.tokenCount,
                            processingTime: result.processingTimeSeconds
                        )
                        self.audioPlayer.play()
                    }
                }
            } catch {
                await MainActor.run {
                    self.errorMessage = "Generation failed: \(error.localizedDescription)"
                }
            }

            await MainActor.run {
                self.isGenerating = false
            }
        }
    }

    // MARK: - Save Audio

    @Published var showingSaveDialog = false

    func saveAudio() {
        #if os(macOS)
        // macOS: Use NSSavePanel
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = "voxtral_output.wav"

        if panel.runModal() == .OK, let url = panel.url {
            do {
                try audioPlayer.saveToFile(url: url)
            } catch {
                errorMessage = "Failed to save: \(error.localizedDescription)"
            }
        }
        #else
        // iOS: Save to Documents directory
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let fileURL = documentsURL.appendingPathComponent("voxtral_output.wav")
        do {
            try audioPlayer.saveToFile(url: fileURL)
            errorMessage = nil
        } catch {
            errorMessage = "Failed to save: \(error.localizedDescription)"
        }
        #endif
    }
}
