import Foundation

/// Manages voice files: preset voices (.safetensors embeddings) and custom cloned voices (.wav reference audio).
///
/// - Preset voices: stored as `voice_embedding/{name}.safetensors` — used with on-device MLX generation
/// - Custom voices: stored as `voice_embedding/{name}.wav` — used with Mistral API (ref_audio)
enum VoiceEmbeddingStore {
    /// Known preset voice names that ship with the model (not deletable).
    static let presetVoices: Set<String> = [
        "jessica", "emma", "allison", "nicole", "vivian",
        "alma", "zephyr", "eric", "brian", "luca",
        "aiden", "dylan", "ono_anna", "ryan", "serena",
        "sohee", "uncle_fu"
    ]

    // MARK: - Save Custom Voice

    /// Save reference audio WAV for a custom cloned voice.
    static func saveCustomVoice(wavData: Data, name: String, to modelPath: URL) throws {
        let voiceDir = modelPath.appendingPathComponent("voice_embedding")

        if !FileManager.default.fileExists(atPath: voiceDir.path) {
            try FileManager.default.createDirectory(at: voiceDir, withIntermediateDirectories: true)
        }

        let filePath = voiceDir.appendingPathComponent("\(sanitizedName(name)).wav")
        try wavData.write(to: filePath)
    }

    /// Load reference audio WAV data for a custom cloned voice.
    static func loadCustomVoiceWAV(name: String, from modelPath: URL) -> Data? {
        let filePath = modelPath
            .appendingPathComponent("voice_embedding")
            .appendingPathComponent("\(name).wav")
        return FileManager.default.contents(atPath: filePath.path)
    }

    // MARK: - Delete

    /// Delete a custom voice. Refuses to delete preset voices.
    static func delete(name: String, from modelPath: URL) throws {
        guard isCustomVoice(name: name) else { return }

        let voiceDir = modelPath.appendingPathComponent("voice_embedding")
        // Remove .wav file
        let wavPath = voiceDir.appendingPathComponent("\(name).wav")
        if FileManager.default.fileExists(atPath: wavPath.path) {
            try FileManager.default.removeItem(at: wavPath)
        }
    }

    // MARK: - List

    /// List all available voice names (both preset and custom).
    static func listAll(in modelPath: URL) -> [String] {
        let voiceDir = modelPath.appendingPathComponent("voice_embedding")
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: voiceDir, includingPropertiesForKeys: nil
        ) else { return [] }

        var names = Set<String>()
        for file in files {
            let ext = file.pathExtension
            if ext == "safetensors" || ext == "wav" {
                names.insert(file.deletingPathExtension().lastPathComponent)
            }
        }
        return names.sorted()
    }

    /// List preset voices (have .safetensors embeddings).
    static func listPreset(in modelPath: URL) -> [String] {
        let voiceDir = modelPath.appendingPathComponent("voice_embedding")
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: voiceDir, includingPropertiesForKeys: nil
        ) else { return [] }

        return files
            .filter { $0.pathExtension == "safetensors" }
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
    }

    /// List custom (user-created) voices (have .wav reference audio).
    static func listCustom(in modelPath: URL) -> [String] {
        let voiceDir = modelPath.appendingPathComponent("voice_embedding")
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: voiceDir, includingPropertiesForKeys: nil
        ) else { return [] }

        return files
            .filter { $0.pathExtension == "wav" && !presetVoices.contains($0.deletingPathExtension().lastPathComponent.lowercased()) }
            .map { $0.deletingPathExtension().lastPathComponent }
            .sorted()
    }

    // MARK: - Query

    /// Check if a voice is a custom (user-created) voice vs a preset.
    static func isCustomVoice(name: String) -> Bool {
        !presetVoices.contains(name.lowercased())
    }

    /// Check if a voice has a .safetensors embedding (can be used on-device).
    static func hasEmbedding(name: String, in modelPath: URL) -> Bool {
        let filePath = modelPath
            .appendingPathComponent("voice_embedding")
            .appendingPathComponent("\(name).safetensors")
        return FileManager.default.fileExists(atPath: filePath.path)
    }

    /// Check if a voice has reference audio (can be used with API).
    static func hasReferenceAudio(name: String, in modelPath: URL) -> Bool {
        let filePath = modelPath
            .appendingPathComponent("voice_embedding")
            .appendingPathComponent("\(name).wav")
        return FileManager.default.fileExists(atPath: filePath.path)
    }

    // MARK: - Helpers

    /// Sanitize a voice name for use as a filename.
    static func sanitizedName(_ name: String) -> String {
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "_-"))
        return String(name.unicodeScalars.filter { allowed.contains($0) })
            .trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: " ", with: "_")
            .lowercased()
    }
}
