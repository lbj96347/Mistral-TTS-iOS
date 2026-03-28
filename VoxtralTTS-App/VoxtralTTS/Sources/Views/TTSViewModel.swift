import SwiftUI
import MLX
import Foundation
import Combine

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
    @Published var currentChunk = 0
    @Published var totalChunks = 1

    // Voice cloning
    @Published var showingAPIKeySettings = false

    // Model
    private var model: VoxtralTTSModel?
    private var tokenizer: VoxtralTokenizer?
    private var modelConfig: ModelConfig?
    private(set) var modelPath: URL?
    private var securityScopedURL: URL?  // Keep security-scoped access alive on iOS

    // Generation task (for cancellation)
    private var generationTask: Task<Void, Never>?

    // Audio
    let audioPlayer = AudioPlayer(sampleRate: 24000)
    private var audioPlayerCancellable: AnyCancellable?

    init() {
        audioPlayerCancellable = audioPlayer.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in
                self?.objectWillChange.send()
            }
    }

    // Persistence
    private let modelBookmarkKey = "VoxtralTTS.modelBookmark"
    private let apiKeyKey = "VoxtralTTS.mistralAPIKey"

    var apiKey: String {
        get { UserDefaults.standard.string(forKey: apiKeyKey) ?? "" }
        set { UserDefaults.standard.set(newValue, forKey: apiKeyKey) }
    }

    var lastModelPath: String? {
        // Resolve bookmark to get display path
        guard let bookmarkData = UserDefaults.standard.data(forKey: modelBookmarkKey) else { return nil }
        var isStale = false
        guard let url = try? URL(resolvingBookmarkData: bookmarkData, bookmarkDataIsStale: &isStale) else { return nil }
        return url.path
    }

    /// Resolve saved bookmark into a security-scoped URL for reloading
    func resolveBookmark() -> URL? {
        guard let bookmarkData = UserDefaults.standard.data(forKey: modelBookmarkKey) else { return nil }
        var isStale = false
        guard let url = try? URL(resolvingBookmarkData: bookmarkData, bookmarkDataIsStale: &isStale) else { return nil }
        if isStale {
            // Re-save fresh bookmark
            saveBookmark(for: url)
        }
        return url
    }

    private func saveBookmark(for url: URL) {
        do {
            let bookmarkData = try url.bookmarkData(options: .minimalBookmark, includingResourceValuesForKeys: nil, relativeTo: nil)
            UserDefaults.standard.set(bookmarkData, forKey: modelBookmarkKey)
        } catch {
            print("[TTS] Failed to save bookmark: \(error)")
        }
    }

    @Published var cachedVoices: [String] = []
    @Published var cachedCustomVoices: [String] = []

    var availableVoices: [String] {
        cachedVoices
    }

    /// Preset voices with on-device embeddings (.safetensors)
    var presetVoices: [String] {
        cachedVoices.filter { !cachedCustomVoices.contains($0) }
    }

    /// Custom cloned voices (.wav reference audio)
    var customVoices: [String] {
        cachedCustomVoices
    }

    /// Whether the selected voice is a custom cloned voice (requires API)
    var isCustomVoiceSelected: Bool {
        guard !selectedVoice.isEmpty, let path = modelPath else { return false }
        return cachedCustomVoices.contains(selectedVoice)
            && !VoiceEmbeddingStore.hasEmbedding(name: selectedVoice, in: path)
    }

    private func loadAvailableVoices(from path: URL) -> [String] {
        VoiceEmbeddingStore.listAll(in: path)
    }

    private func loadCustomVoices(from path: URL) -> [String] {
        VoiceEmbeddingStore.listCustom(in: path)
    }

    func refreshVoices() {
        guard let path = modelPath else { return }
        cachedVoices = loadAvailableVoices(from: path)
        cachedCustomVoices = loadCustomVoices(from: path)
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

        // Release previous security-scoped access if any
        if let prev = securityScopedURL {
            prev.stopAccessingSecurityScopedResource()
            securityScopedURL = nil
        }

        // Start security-scoped access — keep alive for voice embedding access
        let accessing = url.startAccessingSecurityScopedResource()
        print("[TTS] Security-scoped access: \(accessing)")
        if accessing {
            securityScopedURL = url
        }

        Task {
            let startTime = CFAbsoluteTimeGetCurrent()
            do {
                loadingStatus = "Loading config..."
                print("[TTS] Loading model config and weights...")
                let (loadedModel, config) = try loadVoxtralModel(from: url)
                let modelElapsed = CFAbsoluteTimeGetCurrent() - startTime
                print("[TTS] Model loaded in \(String(format: "%.2f", modelElapsed))s — dim=\(config.dim), layers=\(config.nLayers)")

                // Set memory limits for iOS to prevent jetsam kills
                #if os(iOS)
                Memory.cacheLimit = 20 * 1024 * 1024  // 20MB buffer cache
                #endif

                loadingStatus = "Loading tokenizer..."
                print("[TTS] Loading tokenizer...")
                let tokenizerStart = CFAbsoluteTimeGetCurrent()
                let loadedTokenizer = try await VoxtralTokenizer(modelPath: url)
                let tokenizerElapsed = CFAbsoluteTimeGetCurrent() - tokenizerStart
                print("[TTS] Tokenizer loaded in \(String(format: "%.2f", tokenizerElapsed))s")

                // Cache voice list while we have access
                let voices = self.loadAvailableVoices(from: url)
                let custom = self.loadCustomVoices(from: url)
                print("[TTS] Found \(voices.count) voice(s): \(voices), \(custom.count) custom")

                let totalElapsed = CFAbsoluteTimeGetCurrent() - startTime
                print("[TTS] ✅ Total load time: \(String(format: "%.2f", totalElapsed))s")

                await MainActor.run {
                    self.model = loadedModel
                    self.tokenizer = loadedTokenizer
                    self.modelConfig = config
                    self.modelPath = url
                    self.cachedVoices = voices
                    self.cachedCustomVoices = custom
                    self.modelLoaded = true
                    self.isLoadingModel = false

                    // Save bookmark for next launch (works across iOS relaunches)
                    self.saveBookmark(for: url)
                }
            } catch {
                print("[TTS] ❌ Failed to load model: \(error.localizedDescription)")
                print("[TTS] ❌ Full error: \(error)")
                await MainActor.run {
                    self.errorMessage = "Failed to load model: \(error.localizedDescription)"
                    self.isLoadingModel = false
                }
                // Release access on failure
                if accessing {
                    url.stopAccessingSecurityScopedResource()
                    self.securityScopedURL = nil
                }
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

    // MARK: - Voice Management

    func onVoiceCreated(_ name: String) {
        refreshVoices()
        selectedVoice = name
    }

    func deleteCustomVoice(_ name: String) {
        guard let path = modelPath else { return }
        try? VoiceEmbeddingStore.delete(name: name, from: path)
        if selectedVoice == name {
            selectedVoice = ""
        }
        refreshVoices()
    }

    // MARK: - Generation

    func generate() {
        guard !inputText.isEmpty else { return }

        // Route to API generation for custom cloned voices
        if isCustomVoiceSelected {
            generateWithAPI()
            return
        }

        generateOnDevice()
    }

    /// API-based generation for custom cloned voices
    private func generateWithAPI() {
        guard let modelPath = modelPath else { return }
        guard !apiKey.isEmpty else {
            showingAPIKeySettings = true
            errorMessage = "API key required for cloned voices."
            return
        }

        guard let refAudioData = VoiceEmbeddingStore.loadCustomVoiceWAV(
            name: selectedVoice, from: modelPath
        ) else {
            errorMessage = "Reference audio not found for voice '\(selectedVoice)'."
            return
        }

        isGenerating = true
        errorMessage = nil
        generationProgress = 0
        generationStats = nil

        generationTask = Task { [weak self, inputText, apiKey] in
            guard let self = self else { return }
            let startTime = CFAbsoluteTimeGetCurrent()

            do {
                let client = MistralAPIClient(apiKey: apiKey)
                let audioBuffer = try await client.generateWithVoiceClone(
                    text: inputText,
                    referenceAudioData: refAudioData
                )

                let elapsed = Float(CFAbsoluteTimeGetCurrent() - startTime)
                let samples = Int(audioBuffer.frameLength)
                let audioDurationSec = Float(samples) / Float(audioBuffer.format.sampleRate)
                let rtf = audioDurationSec / elapsed

                let hours = Int(audioDurationSec / 3600)
                let mins = Int((audioDurationSec.truncatingRemainder(dividingBy: 3600)) / 60)
                let secs = Int(audioDurationSec.truncatingRemainder(dividingBy: 60))
                let ms = Int((audioDurationSec.truncatingRemainder(dividingBy: 1)) * 1000)

                self.audioPlayer.loadAudio(buffer: audioBuffer)
                self.hasAudio = true
                self.generationStats = GenerationStats(
                    audioDuration: String(format: "%02d:%02d:%02d.%03d", hours, mins, secs, ms),
                    realTimeFactor: rtf,
                    tokenCount: inputText.count,
                    processingTime: elapsed
                )
                self.audioPlayer.play()
            } catch {
                self.errorMessage = "API generation failed: \(error.localizedDescription)"
            }

            self.isGenerating = false
        }
    }

    /// On-device generation for preset voices
    private func generateOnDevice() {
        guard let model = model, let tokenizer = tokenizer else { return }
        guard !inputText.isEmpty else { return }

        let estimatedFrames = estimateMaxFrames(for: inputText)
        let chunks = splitTextIntoChunks(inputText)

        isGenerating = true
        errorMessage = nil
        generationProgress = 0
        currentFrame = 0
        maxFrames = estimatedFrames
        currentChunk = 0
        totalChunks = chunks.count
        generationStats = nil

        generationTask = Task.detached { [weak self, inputText, selectedVoice, estimatedFrames, chunks] in
            guard let self = self else { return }

            do {
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

                let result: GenerationResult?

                if chunks.count > 1 {
                    // Multi-chunk: generate each chunk separately, concatenate audio
                    result = model.generateChunked(
                        text: inputText,
                        tokenizer: tokenizer,
                        voiceEmbedding: voiceEmb,
                        temperature: 0.0,
                        topP: 1.0,
                        maxAudioFrames: estimatedFrames,
                        progressCallback: { chunkIdx, chunkCount, frame, total in
                            Task { @MainActor in
                                self.currentChunk = chunkIdx
                                self.totalChunks = chunkCount
                                self.currentFrame = frame
                                self.maxFrames = total
                                self.generationProgress = (Double(chunkIdx) + Double(frame) / Double(max(total, 1))) / Double(chunkCount)
                            }
                        },
                        shouldCancel: {
                            Task.isCancelled
                        }
                    )
                } else {
                    // Single chunk: use existing direct path
                    let tokenIds = tokenizer.encode(inputText, addSpecialTokens: false)
                    let textTokenIds = tokenIds.map { Int32($0) }

                    result = model.generate(
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
                }

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

    @Published var shareURL: URL?

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
        // iOS: Save to temp directory, then present share sheet
        let tempDir = FileManager.default.temporaryDirectory
        let fileURL = tempDir.appendingPathComponent("voxtral_output.wav")
        try? FileManager.default.removeItem(at: fileURL)
        do {
            try audioPlayer.saveToFile(url: fileURL)
            errorMessage = nil
            shareURL = fileURL
        } catch {
            errorMessage = "Failed to save: \(error.localizedDescription)"
        }
        #endif
    }
}
