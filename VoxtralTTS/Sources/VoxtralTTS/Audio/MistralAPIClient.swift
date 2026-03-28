import AVFoundation
import Foundation

/// Client for Mistral's Voxtral TTS API.
/// Used for voice cloning: sends text + reference audio, receives generated speech.
class MistralAPIClient {
    static let endpoint = URL(string: "https://api.mistral.ai/v1/audio/speech")!
    static let model = "voxtral-mini-tts-2603"

    private let apiKey: String

    init(apiKey: String) {
        self.apiKey = apiKey
    }

    enum APIError: LocalizedError {
        case noAPIKey
        case invalidResponse(Int, String)
        case decodingFailed
        case networkError(Error)

        var errorDescription: String? {
            switch self {
            case .noAPIKey:
                return "Mistral API key is required. Get a free key at console.mistral.ai."
            case .invalidResponse(let code, let message):
                return "API error (\(code)): \(message)"
            case .decodingFailed:
                return "Failed to decode API response."
            case .networkError(let error):
                return "Network error: \(error.localizedDescription)"
            }
        }
    }

    // MARK: - TTS with Voice Cloning

    /// Generate speech from text using a reference audio clip for voice cloning.
    /// - Parameters:
    ///   - text: The text to synthesize
    ///   - referenceAudioData: WAV data of the reference voice (3-30s)
    /// - Returns: Audio samples as AVAudioPCMBuffer (24kHz mono)
    func generateWithVoiceClone(text: String, referenceAudioData: Data) async throws -> AVAudioPCMBuffer {
        guard !apiKey.isEmpty else { throw APIError.noAPIKey }

        let base64Audio = referenceAudioData.base64EncodedString()

        let body: [String: Any] = [
            "model": Self.model,
            "input": text,
            "ref_audio": base64Audio,
            "response_format": "wav"
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: body)

        var request = URLRequest(url: Self.endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIError.networkError(error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.decodingFailed
        }

        guard httpResponse.statusCode == 200 else {
            let message = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw APIError.invalidResponse(httpResponse.statusCode, message)
        }

        // Parse response — API returns JSON with base64 audio_data
        return try decodeResponse(data: data)
    }

    /// Generate speech using a preset voice (no reference audio needed).
    func generateWithPresetVoice(text: String, voiceId: String) async throws -> AVAudioPCMBuffer {
        guard !apiKey.isEmpty else { throw APIError.noAPIKey }

        let body: [String: Any] = [
            "model": Self.model,
            "input": text,
            "voice_id": voiceId,
            "response_format": "wav"
        ]

        let jsonData = try JSONSerialization.data(withJSONObject: body)

        var request = URLRequest(url: Self.endpoint)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = jsonData

        let (data, response): (Data, URLResponse)
        do {
            (data, response) = try await URLSession.shared.data(for: request)
        } catch {
            throw APIError.networkError(error)
        }

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.decodingFailed
        }

        guard httpResponse.statusCode == 200 else {
            let message = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw APIError.invalidResponse(httpResponse.statusCode, message)
        }

        return try decodeResponse(data: data)
    }

    // MARK: - Response Decoding

    private func decodeResponse(data: Data) throws -> AVAudioPCMBuffer {
        // Try JSON response first ({"audio_data": "base64..."})
        if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let audioBase64 = json["audio_data"] as? String,
           let audioData = Data(base64Encoded: audioBase64) {
            return try decodeWAVData(audioData)
        }

        // Fallback: response might be raw audio bytes
        if data.count > 44, String(data: data[0..<4], encoding: .ascii) == "RIFF" {
            return try decodeWAVData(data)
        }

        throw APIError.decodingFailed
    }

    private func decodeWAVData(_ data: Data) throws -> AVAudioPCMBuffer {
        // Write to temp file and read with AVAudioFile
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        try data.write(to: tempURL)
        defer { try? FileManager.default.removeItem(at: tempURL) }

        let file = try AVAudioFile(forReading: tempURL)
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: file.fileFormat.sampleRate,
            channels: file.fileFormat.channelCount,
            interleaved: false
        ) else {
            throw APIError.decodingFailed
        }

        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw APIError.decodingFailed
        }

        try file.read(into: buffer)
        return buffer
    }
}
