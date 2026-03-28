import SwiftUI
import MLX
import MLXNN
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

    // Voice cloning
    @Published var showingAPIKeySettings = false

    // Model
    private var model: VoxtralTTSModel?
    private var tokenizer: VoxtralTokenizer?
    private var modelConfig: ModelConfig?
    private(set) var modelPath: URL?

    // Generation task (for cancellation)
    private var generationTask: Task<Void, Never>?

    // Audio
    let audioPlayer = AudioPlayer(sampleRate: 24000)

    // Persistence keys
    private let modelPathKey = "VoxtralTTS.modelPath"
    private let apiKeyKey = "VoxtralTTS.mistralAPIKey"

    var lastModelPath: String? {
        UserDefaults.standard.string(forKey: modelPathKey)
    }

    var apiKey: String {
        get { UserDefaults.standard.string(forKey: apiKeyKey) ?? "" }
        set { UserDefaults.standard.set(newValue, forKey: apiKeyKey) }
    }

    // MARK: - Voice Lists

    /// All available voices: preset (.safetensors) + custom (.wav)
    var availableVoices: [String] {
        guard let path = modelPath else { return [] }
        return VoiceEmbeddingStore.listAll(in: path)
    }

    /// Preset voices with on-device embeddings
    var presetVoices: [String] {
        guard let path = modelPath else { return [] }
        return VoiceEmbeddingStore.listPreset(in: path)
    }

    /// Custom cloned voices (reference audio)
    var customVoices: [String] {
        guard let path = modelPath else { return [] }
        return VoiceEmbeddingStore.listCustom(in: path)
    }

    /// Whether the selected voice is a custom cloned voice (requires API)
    var isCustomVoiceSelected: Bool {
        guard !selectedVoice.isEmpty, let path = modelPath else { return false }
        return VoiceEmbeddingStore.isCustomVoice(name: selectedVoice)
            && VoiceEmbeddingStore.hasReferenceAudio(name: selectedVoice, in: path)
            && !VoiceEmbeddingStore.hasEmbedding(name: selectedVoice, in: path)
    }

    var playbackProgress: Double {
        guard audioPlayer.duration > 0 else { return 0 }
        return audioPlayer.currentTime / audioPlayer.duration
    }

    // MARK: - Model Loading

    func loadModel(from url: URL) {
        isLoadingModel = true
        errorMessage = nil
        loadingStatus = "Loading config..."

        // Start security-scoped access for sandboxed apps
        let accessing = url.startAccessingSecurityScopedResource()

        Task {
            do {
                loadingStatus = "Loading config..."
                let (loadedModel, config) = try loadVoxtralModel(from: url)

                loadingStatus = "Loading tokenizer..."
                let loadedTokenizer = try await VoxtralTokenizer(modelPath: url)

                // Set memory limits for iOS to prevent jetsam kills
                #if os(iOS)
                Memory.cacheLimit = 20 * 1024 * 1024  // 20MB buffer cache
                #endif

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
        let estimated = wordCount * 15
        return min(max(estimated, 50), 2048)
    }

    // MARK: - Cancel

    func cancelGeneration() {
        generationTask?.cancel()
    }

    // MARK: - Generation

    func generate() {
        guard !inputText.isEmpty else { return }

        // Route to API generation for custom cloned voices
        if isCustomVoiceSelected {
            generateWithAPI()
            return
        }

        // On-device generation for preset voices
        generateOnDevice()
    }

    /// On-device generation using MLX model (preset voices with .safetensors embeddings)
    private func generateOnDevice() {
        guard let model = model, let tokenizer = tokenizer else { return }

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
                let tokenIds = tokenizer.encode(inputText, addSpecialTokens: false)
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

    /// API-based generation for custom cloned voices (sends ref_audio to Mistral API)
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

    // MARK: - Voice Management

    func onVoiceCreated(_ name: String) {
        selectedVoice = name
        objectWillChange.send()
    }

    func deleteCustomVoice(_ name: String) {
        guard let path = modelPath else { return }
        try? VoiceEmbeddingStore.delete(name: name, from: path)
        if selectedVoice == name {
            selectedVoice = ""
        }
        objectWillChange.send()
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
